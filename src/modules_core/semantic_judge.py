import pandas as pd
import time
import json
import os
import ast
import re

from google import genai                     # ★ 新しい import
from google.genai import types              # ★ 型ヒント用
import google.api_core.exceptions as gexc   # SDK が内部で使用
import openai
from openai import OpenAI, DefaultHttpxClient

from datetime import datetime
from dotenv import load_dotenv


load_dotenv()

# --- 設定 ---
API_KEY_GEMINI = os.getenv("GEMINI_API_KEY")
API_KEY_OPENAI = os.getenv("OPENAI_API_KEY")

input_dir = "../data/parallels"
output_dir = "../results"
log_dir = "../results/logs"

# ===== モデル切り替え =====
# MODEL_NAME = "gpt-4.1-nano"
# MODEL_NAME = "gpt-4.1-mini"
# MODEL_NAME = "gpt-4o-mini"
MODEL_NAME = "gemini-2.0-flash"

INPUT_CSV = os.path.join(input_dir, f"parallel_test.csv")
OUTPUT_CSV = os.path.join(output_dir, f"parallel_test_{MODEL_NAME}.csv")


RPM_LIMIT = 15
RETRY_MAX = 3
SLEEP_SEC = 60 / RPM_LIMIT + 0.2




def build_alternation_prompt(sentence, parallel_elements):
    # 並列要素をそれぞれ「」で囲み、改行で連結する
    elements_str = ",".join(f"「{e}」" for e in parallel_elements)
    prompt = (
        f"""目的：
        対象文における並列要素について次の2点を厳密に確認し、「True」または「False」をjson形式で出力してください。

        【入力情報】
        - 対象文: 「{sentence}」
        - 並列要素: {elements_str}（例: 「"A"」,「"B"」,「"C"」）
        """
        """
        【判定基準】
        1. 類似性の確認：
        - 各並列要素の主要語（中心語）が**同じ品詞**であること（例：全て名詞、または全て動詞）。
        - 句構造もできる限り揃っていること（例：全て「名詞＋助詞」の形など）。

        2. 可換性の確認：
        - 並列要素の順序を入れ替えた場合でも、**日本語として自然な文になること**。
        - 並列要素のみを入れ替えても、文全体の意味が大きく変わらないこと（主要な役割・意味構造が維持される）。

        【判定方法】
        - 両方の基準を満たした場合のみ「True」、いずれか一つでも満たさなければ「False」。
        - 出力は必ず以下のJSON形式のみ。判定理由や補足コメントは**絶対に付けない**こと。

        json
        {
        "input": "太郎と花子が学校に行った。",
        "parallel_elements": ["太郎", "花子"],
        "judge_result": "True"
        }
        """
    )
    return prompt



now = datetime.now()
log_filename = os.path.join(log_dir, now.strftime("gpt_log_%Y%m%d_%H%M%S.json"))
log_data = {
    "start_time": now.isoformat(),
    "input_file": INPUT_CSV,
    "output_file": OUTPUT_CSV,
    "step2_prompt":build_alternation_prompt("dummy_sentence", "dummy_element"),
    "results": [],
    "errors": []
}


client = OpenAI(
    # This is the default and can be omitted
    api_key=API_KEY_OPENAI,
    # base_url="http://my.test.server.example.com:8083/v1",
    http_client=DefaultHttpxClient(
    proxy="http://wwwproxy.osakac.ac.jp:8080",
    # transport=httpx.HTTPTransport(local_address="0.0.0.0"),
    ),
)

def ask_gpt(prompt):
    # client = openai.OpenAI(api_key=API_KEY_OPENAI)
    messages = [
        {"role": "system", "content": "あなたは日本語の意味についての専門家です。"},
        {"role": "user", "content": prompt}
    ]
    for attempt in range(RETRY_MAX):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                # response_format={"type": "json_object"},
                temperature=0,
                max_tokens=512
            )
            result_json = response.choices[0].message.content
            return json.loads(result_json)
        except Exception as e:
            if "Rate limit" in str(e):
                print("APIレート制限に達しました。10秒後に再試行します…")
                time.sleep(10)
            else:
                print(e)
        except json.JSONDecodeError:
            print("JSON解析エラー。5秒後に再試行します…")
            time.sleep(5)
        except Exception as e:
            print(f"APIエラー: {e}（{attempt+1}回目）")
            time.sleep(10)
    return {"judge_result": "", "error": "API failed or JSON error"}


client = genai.Client()
# ── JSON スキーマ ────────────────────────
response_schema = {
    "type": "object",
    "properties": {
        "input":             {"type": "string"},
        "parallel_elements": {"type": "array", "items": {"type": "string"}},
        "judge_result":      {"type": "string"},
    },
    "required": ["input", "parallel_elements", "judge_result"],
    "propertyOrdering": ["input", "parallel_elements", "judge_result"],
}

# ── 余計な前置き／コードブロックを除去するヘルパ ─────────
_json_pat = re.compile(r"```(?:json)?\s*({.*?})\s*```", re.S)
def _clean_json(text: str) -> str:
    m = _json_pat.search(text) or re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("JSON block not found")
    return m.group(1) if m.lastindex else m.group(0)

# ── メイン関数 ───────────────────────────
def ask_gemini(prompt: str) -> dict:
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        temperature=0.0,
        max_output_tokens=512,
    )

    for _ in range(RETRY_MAX):
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,          # ← `contents=` が正式
                config=cfg,
            )

            # ① schema が効いている場合
            if resp.parsed:
                return resp.parsed

            # ② テキストから JSON を抽出
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

    return {"error": "max‑retries"}



def ask_model(prompt):
    if MODEL_NAME.startswith("gpt"):
        return ask_gpt(prompt)
    elif MODEL_NAME.startswith("gemini"):
        return ask_gemini(prompt)
    else:
        raise ValueError("未対応モデル名")
    

def to_list(x):
    """
    CSV の文字列を Python のリストに変換
    例："['a','b','c']", '["a","b","c"]', "[a,b,c]" に対応
    """
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not (s.startswith('[') and s.endswith(']')):
        return []
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        inner = s[1:-1]
        items = []
        for item in inner.split(','):
            it = item.strip().strip("'\"")
            if it:
                items.append(it)
        return items

 
# --- メイン・利用例 ---
def judge_parallel(sentence: str, parallel_elements: list) -> bool:
    """
    与えられた文と並列要素リストについて可換性・類似性を判定しTrue/Falseを返す
    """
    prompt = build_alternation_prompt(sentence, parallel_elements)
    result = ask_model(prompt)
    # 結果をそのまま返す場合は return result
    judge = result.get("judge_result", "").strip()
    if judge.lower() == "true":
        return True
    elif judge.lower() == "false":
        return False
    else:
        print("不正な返答:", result)
        return None  # or raise Exception

# sentence = "エンジニアがPythonとJavaを学ぶ"
# parallel_elements = ["Python", "Java"]
# prompt = build_alternation_prompt(sentence, parallel_elements)
# print(ask_model(prompt))
# --- CSV読み込み ---
# if os.path.exists(OUTPUT_CSV):
#     df = pd.read_csv(OUTPUT_CSV, dtype=str)
#     print(f"既存のoutput.csvを再利用します。未処理行のみ再度リクエストします。")
# else:
#     df = pd.read_csv(INPUT_CSV, dtype=str)
#     if "pattern" not in df.columns:
#         df["pattern"] = ""

# # ─── ここで parallel_elements 列を to_list で変換 ───
# df["parallel_elements"] = df["parallel_elements"].apply(to_list)


# # --- メインループ ---
# for idx, row in df.iterrows():
#     sentence = str(row["sentence"])
#     parallel_elements = row["parallel_elements"]
#     judge_result = str(row.get("judge_result", ""))
#     log_entry = {"id": row.get("id", idx), "input": sentence}

#     if judge_result and judge_result != "nan":
#         print(f"[{idx}] スキップ: judge_result既存-> {judge_result}")
#         continue

#     print(f"[{idx}] 送信: {sentence}, {parallel_elements}")
    
#     alternation_prompt = build_alternation_prompt(sentence, parallel_elements)
#     result = ask_model(alternation_prompt)
#     judge_result = result.get("judge_result", "")
#     error = result.get("error", "")
#     df.at[idx, "judge_result"] = judge_result
    

#     log_entry.update({
#         "pattern": judge_result,
#         "error": error,
#         "timestamp": datetime.now().isoformat()
#     })
#     log_data["results"].append(log_entry)
#     if error:
#         log_data["errors"].append(log_entry)

#     df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
#     time.sleep(SLEEP_TIME)

# log_data["end_time"] = datetime.now().isoformat()
# with open(log_filename, "w", encoding="utf-8") as f:
#     json.dump(log_data, f, ensure_ascii=False, indent=2)

# print(f"処理完了！出力: {OUTPUT_CSV}, ログ: {log_filename}")
