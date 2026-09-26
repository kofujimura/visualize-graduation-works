"""卒研概要からキーワードを付与し、概念の抽象度にもとづくキーワードツリーを作る。

1. 全75件の概要をまとめて LLM に渡し、各研究に 3〜5 個のキーワードを付与
   （研究をまたいで同じ概念には同じ語を使わせ、語彙を約100語にそろえる）
2. キーワード一覧から、上位概念 → 下位概念 の木構造を LLM に作らせる
   （適切な親がない場合に限り、補助的な上位概念ノードの追加を許す）
3. 検証（全キーワードが1回ずつ現れるか・親が存在するか・循環がないか）して tree.js を出力

入力は ../similarity-map/graph.json（タイトルと概要）。中間結果は cache/ に保存する。
"""
import json
from collections import Counter
from pathlib import Path

from openai import OpenAI

OUT = Path(__file__).resolve().parent
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)
SRC = OUT.parent / "similarity-map" / "graph.json"

MODEL = "gpt-5.5"
ROOT_NAME = "卒業研究"

client = OpenAI()


def cached(name, fn):
    path = CACHE / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = fn()
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return value


def chat_json(prompt):
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(res.choices[0].message.content)


KEYWORD_PROMPT = """以下は、あるゼミの卒業研究{n}件のタイトルと概要です。
各研究に、内容をよく表すキーワードを3〜5個付けてください。

条件：
- 研究全体で使うキーワードの種類（異なり語数）が90〜110語程度になるようにする
- 同じ概念には研究をまたいで必ず同じ表記を使う（例：「LINE Bot」と「LINEボット」を混在させない）
- 1つの研究のキーワードには、抽象度の異なる語を混ぜる
  （例：分野を表す語「ブロックチェーン」、技術を表す語「NFT」、用途を表す語「学習支援」）
- 1件の研究にしか現れない細かすぎる語は避け、できるだけ複数の研究で共有される語を選ぶ。
  ただし、その研究の特徴を表すのに必要な固有の語は使ってよい
- 簡潔な名詞句にする。一般的な英字表記（NFT、LLM、IoT など）はそのまま使ってよい
- 研究IDはそのまま使う

出力はJSONのみ： {{"works": {{"<研究ID>": ["キーワード", ...], ...}}}}

{works}"""

TREE_PROMPT = """以下は、卒業研究に付けたキーワードの一覧です（括弧内は使われた研究の件数）。
これらのキーワードを、概念の抽象度にもとづいて「上位概念 → 下位概念」の木構造に整理してください。

条件：
- 根は「{root}」とする
- 一覧のキーワードは、すべて木の中にちょうど1回ずつ現れること（表記を変えない）
- 各キーワードの親は、そのキーワードを包含するより上位の概念にする。
  一覧の中に適切な上位概念のキーワードがあれば、それを親にする
- 一覧の中に適切な親がない場合に限り、上位概念のノードを追加してよい（"added": true とする）。
  追加するノードは必要最小限（15個程度まで）にし、一覧にない語であることを確認する
- 根の直下は5〜8個程度の大分類にする
- 木の深さ（根からの段数）は最大5段程度

出力はJSONのみ：
{{"nodes": [{{"name": "ノード名", "parent": "親ノード名", "added": true/false}}, ...]}}
根ノード自身は nodes に含めない。

キーワード一覧：
{keywords}"""


def main():
    graph = json.loads(SRC.read_text(encoding="utf-8"))
    works = graph["nodes"]

    kw = cached("keywords.json", lambda: chat_json(KEYWORD_PROMPT.format(
        n=len(works),
        works="\n\n".join(f"[{w['id']}] {w['title']}\n{w['summary']}" for w in works),
    )))["works"]

    missing = [w["id"] for w in works if w["id"] not in kw]
    assert not missing, f"キーワードのない研究: {missing}"

    counts = Counter(k for ks in kw.values() for k in ks)
    tree = cached("tree.json", lambda: chat_json(TREE_PROMPT.format(
        root=ROOT_NAME,
        keywords="\n".join(f"- {k}（{c}）" for k, c in counts.most_common()),
    )))["nodes"]

    # 親を手で直したいノードは tree_override.json に {"ノード名": "新しい親"} で書く
    override_path = OUT / "tree_override.json"
    overrides = json.loads(override_path.read_text(encoding="utf-8")) if override_path.exists() else {}
    tree = [{**n, "parent": overrides.get(n["name"], n["parent"])} for n in tree]

    nodes = validate(tree, counts)

    works_by_kw = {}
    for w in works:
        for k in kw[w["id"]]:
            works_by_kw.setdefault(k, []).append(w["id"])
    promote_redundant(nodes, works_by_kw)

    data = {
        "root": ROOT_NAME,
        "nodes": [{**n, "works": works_by_kw.get(n["name"], [])} for n in nodes],
        "works": {w["id"]: {"title": w["title"], "year": w["year"], "label": w["label"],
                            "summary": w["summary"], "keywords": kw[w["id"]]} for w in works},
    }
    (OUT / "tree.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "tree.js").write_text("window.TREE = " + json.dumps(data, ensure_ascii=False) + ";\n",
                                 encoding="utf-8")

    added = [n["name"] for n in nodes if n["added"]]
    print(f"研究 {len(works)} 件 / キーワード {len(counts)} 語 / 1件あたり平均 "
          f"{sum(len(v) for v in kw.values()) / len(kw):.1f} 語")
    print(f"ツリー: ノード {len(nodes)}（うち追加した上位概念 {len(added)}）: {added}")


def promote_redundant(nodes, works_by_kw):
    """1件の研究にしか付いていない語で、親の語もその同じ1件にしか付いていないものを1つ上の層へ移す。

    親子が同じ研究1件だけを指していると、階層が深くなるだけで情報が増えないため。
    移した語には promotedFrom（元の親）を記録する。
    """
    by_name = {n["name"]: n for n in nodes}
    changed = True
    while changed:
        changed = False
        for n in nodes:
            p = by_name.get(n["parent"])
            mine = works_by_kw.get(n["name"], [])
            if p and len(mine) == 1 and works_by_kw.get(p["name"], []) == mine:
                n.setdefault("promotedFrom", p["name"])
                n["parent"] = p["parent"]
                changed = True
                print(f"[格上げ] {n['name']}：{p['name']} の下 → {p['parent']} の下（研究 {mine[0]}）")


def validate(tree, counts):
    """LLM の出力を検証し、問題があれば補正して報告する。"""
    nodes, seen = [], set()
    for n in tree:
        name = n["name"].strip()
        if name in seen or name == ROOT_NAME:
            print(f"[補正] 重複ノードを除外: {name}")
            continue
        seen.add(name)
        nodes.append({"name": name, "parent": n["parent"].strip(),
                      "added": name not in counts})
    names = seen | {ROOT_NAME}

    for n in nodes:
        if n["parent"] not in names:
            print(f"[補正] 親「{n['parent']}」が存在しないため根に接続: {n['name']}")
            n["parent"] = ROOT_NAME

    for k in counts:
        if k not in seen:
            print(f"[補正] ツリーに無いキーワードを根に接続: {k}")
            nodes.append({"name": k, "parent": ROOT_NAME, "added": False})

    parent = {n["name"]: n["parent"] for n in nodes}
    for n in nodes:
        path, cur = set(), n["name"]
        while cur != ROOT_NAME:
            if cur in path:
                print(f"[補正] 循環を検出したため根に接続: {n['name']}")
                n["parent"] = ROOT_NAME
                parent[n["name"]] = ROOT_NAME
                break
            path.add(cur)
            cur = parent[cur]
    return nodes


if __name__ == "__main__":
    main()
