"""卒業研究の類似度マップ用データを生成する。

1. 2020-2024 要旨集PDF / 2025 Web記事 / 2026 中間発表概要 からテキストを収集
2. 2020-2024 は要旨全文を約250字の概要に要約（2025・2026と粒度をそろえるため）
3. 全件に短縮ラベル（10字程度）を付与
4. text-embedding-3-large でベクトル化 → コサイン類似度 → 上位3件をエッジ化
5. K-means でグループ化し、各グループに名前を付与
6. graph.json を出力

中間結果は cache/ に保存し、再実行時は API を呼ばない。
"""
import html
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)

CHAT_MODEL = "gpt-5.4-mini"
EMBED_MODEL = "text-embedding-3-large"
TOP_K = 3
N_CLUSTERS = 8

# 要旨集PDFのページ番号 → タイトル（抽出時に崩れるため手で確定）
TITLES = {
    2020: {
        2: "ジャンル別に映画作品を提案するLINE Bot",
        3: "個人感覚に基づく音楽ジャンルの3Dの可視化",
        4: "LINE Messaging APIを用いた作業効率化アプリの開発",
        5: "スケジュール情報の3D表現",
        6: "消耗品の残量をLINE Botで通知するシステムの開発",
        7: "ポーズで動画を再生するインタラクティブメディアの制作",
        8: "気温に応じたレシピを提案するLINE Botの開発",
        9: "飲食店検索におけるバリアフリー情報の共有機能の実現",
        10: "転移学習を用いた猫用の玩具の制作",
        11: "転移学習を用いたWeb会議のための居眠り検知機能の開発",
        12: "姿勢推定モデルによる行動改善アプリの制作",
    },
    2021: {
        2: "LINE Beaconを用いた時間管理システムの開発",
        3: "OCRを用いた食品の消費期限管理システム",
        4: "買い物を支援する価格情報共有LINE Botの制作",
        5: "ジェスチャによる非接触操作システムの開発と評価",
        6: "ビデオチャットにおける顔への落書きツールの開発",
        7: "摘粒支援のためのブドウ果粒検出モデルの構築",
        8: "Reactを用いた直感的レーダーチャートの作成",
        9: "3D Avatarを設置するReact Componentの実装",
        10: "Three.jsを用いたVRサウンドビジュアライザの開発",
        11: "画像の代表色をイラストに自動着色するシステム",
        12: "湿度に応える光のオブジェの制作",
        13: "音声認識を用いた照明による単語の視覚化",
        14: "ダストセンサーを用いた掃除管理LINE Botアプリ",
    },
    2022: {
        2: "VRに対応したNFTの売買システムの比較",
        3: "NFT発行アプリの開発とその流通に関する調査",
        4: "センサーの値を可視化する花のオブジェの制作",
        5: "PoseNetを用いた姿勢の可視化",
        6: "BLE電波強度を用いた自転車間距離の通知デバイスの開発",
        7: "BLE電波強度を用いた顧客の滞在時間の推定",
        8: "NFCタグで複数端末を連携させるサイネージデバイスの制作",
        9: "Handposeと360度カメラを用いた写真撮影ツールの制作",
        10: "画像認識を使ったペットのトイレの状態推定",
    },
    2023: {
        2: "ハンドトラッキングを使ったバーチャル生け花",
        3: "バーチャルタップダンスのための足の接地判定",
        4: "AIとダンサーの協調作業による音楽生成",
        5: "AIを用いたp5.jsコードの生成による図形の描画",
        6: "音楽の特徴を用いた画像生成AIによるキャラクターの生成",
        7: "NFTで学習成果を認定するeラーニングシステム",
        8: "NFCを利用したコミュニケーション促進Webアプリの開発",
        9: "多数のNFCタグを活用したインタラクション",
        10: "華展における反応型インスタレーションの制作",
    },
    2024: {
        2: "GPTモデルを用いた音楽歌詞の解析によるキャラクター生成システム",
        3: "聴覚障がい者のための環境音認識",
        4: "生成AIを用いた画像からのシャインマスカット重さ推定",
        5: "目の温度測定システムの開発とその疲労度推定への応用",
        6: "読者の感情を反映したNFTを発行する読書体験システム",
        7: "NFCチップとNFTの連携による真贋判定システム",
        8: "NFT活用による学習意欲の向上とコミュニティ活性化LMS",
    },
}

client = OpenAI()


def cached(name, fn):
    path = CACHE / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = fn()
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return value


def chat(prompt):
    res = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return res.choices[0].message.content.strip()


def pdf_page_text(pdf, page):
    text = subprocess.run(
        ["pdftotext", "-f", str(page), "-l", str(page), str(pdf), "-"],
        capture_output=True, text=True, check=True,
    ).stdout
    text = re.sub(r"[​‪-‮]", "", text)
    # 本文は「１．はじめに」以降（タイトル・著者名は含めない）
    m = re.search(r"１\s*．\s*はじめに", text)
    body = text[m.start():] if m else text
    body = re.sub(r"\s+", "", body)
    return body


def collect_pdf_years():
    works = []
    for year, pages in TITLES.items():
        pdf = ROOT / f"卒研要旨集fujimura{year}.pdf"
        for page, title in pages.items():
            works.append({
                "id": f"{year}-{page:02d}",
                "year": year,
                "title": title,
                "fulltext": pdf_page_text(pdf, page),
            })
    return works


def collect_2025():
    # macOS の python.org 版 Python は証明書未設定のことがあるため curl で取得
    src = subprocess.run(["curl", "-sL", "https://web.fujimura.com/blog/archives/1338/"],
                         capture_output=True, text=True, check=True).stdout
    works = []
    for i, li in enumerate(re.findall(r"<li(?:\s[^>]*)?>(.*?)</li>", src, re.S), 1):
        m = re.search(r"<strong>(.*?)</strong>", li)
        if not m:
            continue
        body = html.unescape(re.sub(r"<[^>]+>", "", li.split("</strong>", 1)[1])).strip()
        works.append({"id": f"2025-{i:02d}", "year": 2025,
                      "title": html.unescape(m.group(1)).strip(), "summary": body})
    return works


def collect_2026():
    md = (ROOT / "卒研進捗報告fujimura2026" / "中間発表概要2026.md").read_text(encoding="utf-8")
    works = []
    for i, block in enumerate(re.split(r"^### ", md, flags=re.M)[1:], 1):
        title, _, body = block.partition("\n")
        body = re.split(r"^## ", body, flags=re.M)[0]
        works.append({"id": f"2026-{i:02d}", "year": 2026,
                      "title": title.strip(), "summary": re.sub(r"\s+", "", body)})
    return works


SUMMARY_PROMPT = """次の卒業研究の要旨を、250字程度の日本語の概要にまとめてください。
「本研究では」で始め、目的・手法（使用技術）・成果を含め、「〜ました」調の敬体で書いてください。
個人名は含めないでください。概要の本文だけを出力してください。

タイトル：{title}
要旨：
{text}"""

LABEL_PROMPT = """次の卒業研究のタイトルと概要から、類似度マップ上に表示する短いラベルを作ってください。
- 日本語で全角10字以内（英字は半角2字で全角1字と数える）
- 研究の対象と特徴が一目で分かる名詞句（例：「映画提案LINE Bot」「ブドウ果粒検出」「推し活NFT」）
- ラベルだけを出力

タイトル：{title}
概要：{summary}"""

CLUSTER_PROMPT = """次の卒業研究のタイトル群は、内容の類似度でまとめた1つのグループです。
このグループの共通テーマを表す名前を、日本語で全角12字以内で付けてください。名前だけを出力してください。

{titles}"""


def main():
    works = cached("works_raw.json", lambda: collect_pdf_years() + collect_2025() + collect_2026())

    summaries = cached("summaries.json", lambda: {
        w["id"]: w.get("summary") or chat(SUMMARY_PROMPT.format(title=w["title"], text=w["fulltext"]))
        for w in works
    })
    for w in works:
        w["summary"] = summaries[w["id"]]

    labels = cached("labels.json", lambda: {
        w["id"]: chat(LABEL_PROMPT.format(title=w["title"], summary=w["summary"])) for w in works
    })
    # 手で直したいラベルは labels_override.json に {"id": "ラベル"} で書く
    override_path = OUT / "labels_override.json"
    overrides = json.loads(override_path.read_text(encoding="utf-8")) if override_path.exists() else {}
    for w in works:
        w["label"] = overrides.get(w["id"], labels[w["id"]]).strip("「」 ")

    def embed():
        inputs = [f"タイトル：{w['title']}\n概要：{w['summary']}" for w in works]
        res = client.embeddings.create(model=EMBED_MODEL, input=inputs)
        return {w["id"]: d.embedding for w, d in zip(works, res.data)}

    emb = cached("embeddings.json", embed)
    X = np.array([emb[w["id"]] for w in works])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    sim = X @ X.T

    # 各ノードの類似上位 TOP_K 件をエッジに（重複は1本にまとめる）
    edges = {}
    for i in range(len(works)):
        order = np.argsort(-sim[i])
        for j in [j for j in order if j != i][:TOP_K]:
            key = tuple(sorted((i, j)))
            edges[key] = float(sim[i, j])

    groups = kmeans(X, N_CLUSTERS, seed=0)
    cluster_names = cached(f"cluster_names_k{N_CLUSTERS}.json", lambda: {
        str(c): chat(CLUSTER_PROMPT.format(
            titles="\n".join(f"- {w['title']}" for w, g in zip(works, groups) if g == c)))
        for c in range(N_CLUSTERS)
    })

    # 2次元の初期配置（コサイン距離の t-SNE）。表示側ではこの位置へ引き寄せつつ重なりを解消する
    from sklearn.manifold import TSNE
    xy = TSNE(n_components=2, metric="cosine", perplexity=10, init="pca", random_state=0).fit_transform(X)
    xy = (xy - xy.mean(0)) / np.abs(xy - xy.mean(0)).max()

    neighbors = {}
    for i, w in enumerate(works):
        order = [j for j in np.argsort(-sim[i]) if j != i][:5]
        neighbors[w["id"]] = [{"id": works[j]["id"], "sim": round(float(sim[i, j]), 3)} for j in order]

    graph = {
        "meta": {"embedModel": EMBED_MODEL, "topK": TOP_K, "clusters": cluster_names},
        "nodes": [{
            "id": w["id"], "year": w["year"], "title": w["title"], "label": w["label"],
            "summary": w["summary"], "group": int(g), "neighbors": neighbors[w["id"]],
            "px": round(float(p[0]), 4), "py": round(float(p[1]), 4),
        } for w, g, p in zip(works, groups, xy)],
        "links": [{"source": works[i]["id"], "target": works[j]["id"], "value": round(v, 3)}
                  for (i, j), v in edges.items()],
    }
    (OUT / "graph.json").write_text(json.dumps(graph, ensure_ascii=False, indent=1), encoding="utf-8")
    # index.html を file:// で直接開けるよう、同じデータを JS としても出力
    (OUT / "graph.js").write_text("window.GRAPH = " + json.dumps(graph, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"{len(graph['nodes'])} nodes, {len(graph['links'])} links")
    for c, name in cluster_names.items():
        print(c, name, sum(1 for g in groups if g == int(c)))


def kmeans(X, k, seed=0, n_init=20, iters=100):
    """コサイン距離の K-means（k-means++ 初期化、最良の初期値を採用）。"""
    rng = np.random.default_rng(seed)
    best, best_inertia = None, np.inf
    for _ in range(n_init):
        centers = [X[rng.integers(len(X))]]
        for _ in range(k - 1):
            d = np.min([1 - X @ c for c in centers], axis=0).clip(0)
            centers.append(X[rng.choice(len(X), p=d / d.sum())])
        C = np.array(centers)
        for _ in range(iters):
            labels = np.argmax(X @ C.T, axis=1)
            newC = np.array([X[labels == c].mean(0) if np.any(labels == c) else C[c] for c in range(k)])
            newC /= np.linalg.norm(newC, axis=1, keepdims=True)
            if np.allclose(newC, C):
                break
            C = newC
        inertia = np.sum(1 - np.max(X @ C.T, axis=1))
        if inertia < best_inertia:
            best, best_inertia = labels, inertia
    # グループ番号を大きい順に振り直す（色の割り当てを安定させる）
    order = np.argsort(-np.bincount(best, minlength=k))
    remap = {old: new for new, old in enumerate(order)}
    return [remap[g] for g in best]


if __name__ == "__main__":
    main()
