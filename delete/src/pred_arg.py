# -*- coding: utf-8 -*-
import os
import re
import json
import time
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime

# ========== 0. 事前設定 ==========
# ========== 0. 事前設定 ==========
from pathlib import Path

input_csv      = Path('../results/extract_pred_arg_pair/target_datas/test_target_data/test_target_data_extract_po_pair.csv')
onto_json_path = Path('../ontology/1_movie_ontology_trans_ja.json')

# 入力CSVと同じディレクトリ配下に出力用フォルダを作成
out_root  = input_csv.parent / 'llm_outputs'   # 任意名
log_dir   = out_root / 'logs'
out_root.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

# ファイル名は入力CSVのstemを使って自動命名
stem = input_csv.stem
output_json = out_root / f'{stem}_rel_arg_labeled.json'
output_csv  = out_root / f'{stem}_rel_arg_labeled.csv'
triple_csv  = out_root / f'{stem}_triples_auto.csv'

MODEL_NAME = "gemini-2.0-flash"
RETRY_MAX  = 3
RPM_LIMIT  = 15
SLEEP_SEC  = 60 / RPM_LIMIT + 0.2


# ====== Gemini SDK (google.genai) 周り ======
from google import genai
from google.genai import types
import google.api_core.exceptions as gexc

# ====== 環境変数 ======
from dotenv import load_dotenv
load_dotenv()
API_KEY_GEMINI = os.getenv("GEMINI_API_KEY")
assert API_KEY_GEMINI, "GEMINI_API_KEY が環境変数に設定されていません。"

client = genai.Client(api_key=API_KEY_GEMINI)

# ========== 1. ユーティリティ ==========
def unique_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

_json_pat = re.compile(r"```(?:json)?\s*({.*?})\s*```", re.S)
def _clean_json(text: str) -> str:
    m = _json_pat.search(text) or re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("JSON block not found")
    return m.group(1) if m.lastindex else m.group(0)

# ========== 2. Gemini 応答スキーマ ==========
response_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "relations": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "text":     types.Schema(type=types.Type.STRING),
                    "pid":      types.Schema(type=types.Type.STRING),
                    "label_ja": types.Schema(type=types.Type.STRING),
                },
                required=["text", "pid", "label_ja"]
            )
        ),
        "concepts": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "text":     types.Schema(type=types.Type.STRING),
                    "qid":      types.Schema(type=types.Type.STRING),
                    "label_ja": types.Schema(type=types.Type.STRING),
                },
                required=["text", "qid", "label_ja"]
            )
        ),
    },
    required=["relations", "concepts"]
)

def ask_gemini(prompt: str) -> dict:
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        temperature=0.0,
        max_output_tokens=2048,
    )

    for _ in range(RETRY_MAX):
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=cfg,
            )
            if getattr(resp, "parsed", None):
                return resp.parsed
            try:
                return json.loads(resp.text)
            except json.JSONDecodeError:
                return json.loads(_clean_json(resp.text))

        except gexc.ResourceExhausted:
            print("Rate‑limit…待機"); time.sleep(SLEEP_SEC)
        except (json.JSONDecodeError, ValueError):
            print("JSON 抽出失敗…再試行"); time.sleep(SLEEP_SEC)
        except Exception as e:
            print(f"Gemini Error: {e} …再試行"); time.sleep(SLEEP_SEC)

    return {"relations": [], "concepts": [], "error": "max‑retries"}

# ========== 3. プロンプト生成 ==========
def build_prompt(rel_list, arg_list, onto):
    rel_str = "\n".join([f"- {t}" for t in rel_list])
    arg_str = "\n".join([f"- {t}" for t in arg_list])

    rel_candidates = "\n".join([f"- {r['pid']}: {r['label_ja']}" for r in onto.get('relations', [])])
    con_candidates = "\n".join([f"- {c['qid']}: {c['label_ja']}" for c in onto.get('concepts', [])])

    prompt = f"""
あなたは日本語語句を、以下の映画オントロジーに定義された relation / concept に対応付けるタスクを行います。

### 重要条件
- 「rel_ja」リストの各要素は relation に対応付けます。
- 「arg_ja」リストの各要素は concept に対応付けます。
- 意味的に適切な対応がある場合のみ、その要素を出力して下さい。対応がない語句は出力に含めないでください。
- 助詞などによる表記揺れはある程度許容します。
- 出力は必ず JSON 形式のみ（以下の形式）で、説明文や余計なテキストは不要です。

### 対象リスト
[REL_JA]
{rel_str}

[ARG_JA]
{arg_str}

### オントロジー: RELATION 候補 (pid: 日本語ラベル)
{rel_candidates}

### オントロジー: CONCEPT 候補 (qid: 日本語ラベル)
{con_candidates}

### 出力形式
{{
  "relations": [
    {{"text":"監督","pid":"P57","label_ja":"監督"}}
  ],
  "concepts": [
    {{"text":"千と千尋の神隠し","qid":"Q11424","label_ja":"映画"}}
  ]
}}
"""
    return prompt.strip()

# ========== 4. リスト構築（品詞フィルタなし） ==========
def build_id2lists(csv_path: str):
    """
    CSVからidごとの rel_ja/arg_ja 重複なしリストを作る。
    ★品詞フィルタなし★
    """
    df = pd.read_csv(csv_path)

    id2lists = {}
    for id_, g in df.groupby('id'):
        rel_list = unique_preserve_order(g['rel_ja'])
        arg_list = unique_preserve_order(g['arg_ja'])
        id2lists[id_] = {'rel_ja': rel_list, 'arg_ja': arg_list}

    return id2lists, df

# ========== 5. ラベリング実行 ==========
def run_labeling(csv_path: str, onto_path: str):
    id2lists, df = build_id2lists(csv_path)
    id2sent = df.groupby('id')['sentence'].first().to_dict()

    with open(onto_path, encoding='utf-8') as f:
        onto = json.load(f)

    now = datetime.now()
    log_filename = os.path.join(log_dir, now.strftime("label_log_%Y%m%d_%H%M%S.json"))
    log_data = {
        "start_time": now.isoformat(),
        "input_file": csv_path,
        "ontology_file": onto_path,
        "results": [],
        "errors": []
    }

    id2mapped = {}
    for id_, lists in tqdm(id2lists.items(), desc="LLM labeling"):
        rel_list = lists['rel_ja']
        arg_list = lists['arg_ja']
        prompt = build_prompt(rel_list, arg_list, onto)
        result = ask_gemini(prompt)

        relations_out = result.get("relations", [])
        concepts_out  = result.get("concepts",  [])

        rel_map = {item["text"]: {"label_ja": item["label_ja"], "pid": item["pid"]}
                   for item in relations_out if "text" in item}
        arg_map = {item["text"]: {"label_ja": item["label_ja"], "qid": item["qid"]}
                   for item in concepts_out  if "text" in item}

        id2mapped[id_] = {
            "rel_ja_list": rel_list,
            "arg_ja_list": arg_list,
            "relations": relations_out,
            "concepts":  concepts_out,
            "rel_map":   rel_map,
            "arg_map":   arg_map,
            "sentence":  id2sent.get(id_, "")
        }

        log_data["results"].append({
            "id": id_,
            "rel_ja_input": rel_list,
            "arg_ja_input": arg_list,
            "relations_out": relations_out,
            "concepts_out":  concepts_out
        })
        if "error" in result:
            log_data["errors"].append({"id": id_, "error": result["error"]})

        time.sleep(SLEEP_SEC)

    log_data["end_time"] = datetime.now().isoformat()
    with open(log_filename, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2, default=str)
    return id2mapped, onto

# ========== 6. Triple生成 ==========
def make_triples(id2mapped: dict, onto: dict):
    pid2rel = {r['pid']: r for r in onto.get('relations', [])}

    rows = []
    for id_, bundle in id2mapped.items():
        sent = bundle.get("sentence", "")
        concepts = bundle.get("concepts", [])
        qid2texts = {}
        for c in concepts:
            qid2texts.setdefault(c['qid'], []).append(c['text'])

        for rel in bundle.get("relations", []):
            pid      = rel["pid"]
            rel_label= rel["label_ja"]
            rel_text = rel["text"]

            onto_rel = pid2rel.get(pid)
            if not onto_rel:
                rows.append({
                    "id": id_, "sentence": sent,
                    "subject": None, "relation": rel_label, "object": None
                })
                continue

            domain_qid = onto_rel.get("domain", "")
            range_qid  = onto_rel.get("range", "")

            subj_cands = list(qid2texts.get(domain_qid, [])) if domain_qid else [None]
            obj_cands  = list(qid2texts.get(range_qid,  [])) if range_qid  else [None]

            if not subj_cands:
                subj_cands = [None]
            if not obj_cands:
                obj_cands  = [None]

            for s in subj_cands:
                for o in obj_cands:
                    rows.append({
                        "id": id_,
                        "sentence": sent,
                        "subject": s,
                        "relation": rel_label,
                        "object":  o
                    })

    # 重複排除
    dedup = { (r["id"], r["sentence"], r["subject"], r["relation"], r["object"]) : r
              for r in rows }
    return list(dedup.values())

# ========== 7. 実行 ==========
if __name__ == "__main__":
    id2mapped, onto = run_labeling(input_csv, onto_json_path)

    # # 確認
    # for i, (id_, d) in enumerate(id2mapped.items()):
    #     print(f"[{i}] id: {id_}")
    #     print("  relations:", d["relations"])
    #     print("  concepts :", d["concepts"])
    #     print()

    # 保存
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(id2mapped, f, ensure_ascii=False, indent=2)

    rows_flat = []
    for id_, d in id2mapped.items():
        for r in d["relations"]:
            rows_flat.append({
                "id": id_,
                "type": "relation",
                "text": r["text"],
                "label_ja": r["label_ja"],
                "pid": r["pid"]
            })
        for c in d["concepts"]:
            rows_flat.append({
                "id": id_,
                "type": "concept",
                "text": c["text"],
                "label_ja": c["label_ja"],
                "qid": c["qid"]
            })
    if rows_flat:
        pd.DataFrame(rows_flat).to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"ラベル付与結果CSV: {output_csv}")

    triple_rows = make_triples(id2mapped, onto)
    pd.DataFrame(triple_rows).to_csv(triple_csv, index=False, encoding="utf-8-sig")
    print(f"Triple出力: {triple_csv}")
