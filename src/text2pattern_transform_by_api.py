import pandas as pd
import time
import json
import os

import google.generativeai as genai
import google.api_core.exceptions
import openai
from openai import OpenAI, DefaultHttpxClient

from datetime import datetime
from dotenv import load_dotenv


load_dotenv()

# --- 設定 ---
API_KEY_GEMINI = os.getenv("GEMINI_API_KEY")
API_KEY_OPENAI = os.getenv("OPENAI_API_KEY")

input_dir = "./data"
output_dir = "./results"
log_dir = "./logs"

# ===== モデル切り替え =====
MODEL_NAME = "gpt-4o-mini"

INPUT_CSV = os.path.join(input_dir, f"text_to_pattern_eval_20_ver4.0.csv")
OUTPUT_CSV = os.path.join(output_dir, f"result_text_to_pattern_eval_20_{MODEL_NAME}_ver4.5.csv")


RPM_LIMIT = 15
RETRY_MAX = 3
SLEEP_TIME = 60 / RPM_LIMIT + 0.2



pattern_manual = """
【パターン記述法】
・[Xi]：目的語主辞要素
・[Yi]：述語主辞要素
・[Mi]：修飾語
    ◻︎　それぞれ（1 ≦ i）で１からi番目の要素
    ◻︎　リテラル以外の要素は[]で必ず囲む
・＊n：文節単位の連体修飾をn回されたもの
・#n：文節単位の連用修飾をn回されたもの
 *もしくは#の後には数値（修飾回数）もしくは、構造を示す()が必ず出現する
・[Xi] & [Xi+1]&...&[Xi+n]：並列要素[Xi],[Xi+1],...[Xi+n]が並列構造をなす
    ◻︎　& ∈ {と｜、｜や｜および｜または}
    ◻︎　i ：並列要素数(1 ≦ i )
・-{指定品詞}
    ◻︎　ex. [X-サ変名詞]
    ◻︎ 使える品詞辞書(動詞,名詞,サ変名詞,形容詞)
・[*([〇〇1]&[△△2])]：並列要素〇〇1,△△2に対する修飾
・[*([M1]&[M2])〇〇]：要素〇〇に対する並列修飾
・[...]を, [〇〇-サ変]する：リテラルとしての文字列（を/する/などの助詞）
"""

clause_defintion = """
【文節の定義】
日本語における文節の定義は「自立語＋付属語」とします。
・「接頭辞＋自立語＋付属語・接尾辞」
・0個以上の接頭辞、付属語・接尾辞
・複合名詞などでは自立語が複数の場合もある
"""

fewshot_examples = """
【text->パターンへの変換例】
※ ｜は文節の定義に沿った区切り。実際の入力文に文節の区切りは記載されていない。
- 「入力文」->(述語,目的語),..., -> パターン
【正例】
- 【正例】「仕事を分担する」->(分担する,仕事)-> [X1-名詞]を[Y1-サ変]する
- 【正例】「明日の予定を見る」->(見る,明日の予定)-> [*1X1]を[Y1-動詞]
- 【正例】「部長の重要な仕事を行う」->(部長の重要な仕事,行う)-> [*2X1]を[Y1-動詞]
- 【正例】「果物をゆっくりと切る」->(ゆっくりと切る,果物)-> [X1]を[#1Y1-動詞]
- 【正例】「危険と安心を確認する」->(確認する,危険),(確認する,安心)-> [X1]&[X2]を[Y1-サ変]する
- 【正例】「責任を整理、確認する」->(整理する,責任),(確認する,責任)-> [X1]を[Y1-サ変]&[Y2-サ変]する
- 【正例】「会社の給料や通勤時間を比べる」->(比べる,会社の給料),(比べる,通勤時間)-> [*1X1]&[X2]を[Y1]
- 【正例】「エンジニアと営業の責任を知る」->(知る,エンジニアの責任),(知る,営業の責任)-> [*([M1]&[M2])X1]を[Y1]
- 【正例】「会社のエンジニアと営業の責任を知る」->(知る,会社のエンジニアの責任),(知る,会社の営業の責任)-> [*1*([M1]&[M2])X1]を[Y1]
- 【正例】「会社のエンジニアと営業の仕事の責任を知る」->(知る,会社のエンジニアの仕事の責任),(知る,会社の営業の仕事の責任)-> [*1*([M1]&[M2])*1X1]を[Y1]
- 【正例】「ハチは花の蜜、花粉、樹液を集め、幼虫を育てる。」->(集める,花の蜜),(集める,花粉),(集める,樹液),(育てる,幼虫)->[*1X1]&[X2]&[X3]を[Y1-動詞],[X4]を[Y2-動詞]
【負例】
- 【負例】「明日の予定を見る」->(見る,明日の予定)-> *1[X1]を[Y1-動詞]【負例理由】*1が[]に入っていない
- 【負例】「部長の重要な仕事を行う」->(部長の重要な仕事,行う)-> *2X1をY1【負例理由】[]で囲まれていない
- 【負例】「部長の重要な仕事を行う」->(部長の重要な仕事,行う)-> [*1*1X1]を[Y1]【負例理由】*1*1が*2にまとまっていない
- 【負例】「果物をゆっくりと切る」->(ゆっくりと切る,果物)-> [X]を[#Y-動詞]【負例理由】変数X,Yに数値がついていない
- 【負例】「エンジニアと営業の責任を知る」->(知る,エンジニアの責任),(知る,営業の責任)-> *([M1]&[M2])X1を[Y1]【負例理由】*()が[]で囲まれていない
- 【負例】「会社のエンジニアと営業の仕事の責任を知る」->(知る,会社のエンジニアの仕事の責任),(知る,会社の営業の仕事の責任)-> *1*(M1&M2)[*1X1]をY1【負例理由】*1*()が[]で囲まれていない
"""


def build_prompt(sentence, target_triple):
    prompt = (
        "あなたは日本語テキストから指定された知識グラフ (KG) の述語・目的語ペアを抽出し、以下で定義する【パターン記述法】に従って文を記号パターンへ変換する専門家です。\n"
        "【入力文】と入力文から【抽出したい知識グラフTriple】を提示するので、パターンに変換してください。"
        + pattern_manual + "\n"
        + clause_defintion + "\n"
        + fewshot_examples + "\n"
        "【注意・確認点】：変数, 修飾子, 品詞指定, リテラル以外のまとまりは必ず[]で囲むこと\n"
        f"【入力文】：\n{sentence}\n"
        f"【抽出したい知識グラフTriple】：\n{target_triple}\n"
        "【出力】\n"
        "以下のJSON形式で出力してください：\n"
        "{\n"
        '  "input": "入力文",\n'
        '  "target_triple": "抽出したいトリプル",\n'
        '  "pattern": "変換されたパターン"\n'
        "}\n"
    )
    return prompt


now = datetime.now()
log_filename = os.path.join(log_dir, now.strftime("gpt_log_%Y%m%d_%H%M%S.json"))
log_data = {
    "start_time": now.isoformat(),
    "input_file": INPUT_CSV,
    "output_file": OUTPUT_CSV,
    "prompt":build_prompt("dummy_sentence", "dummy_triple"),
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

def ask_gpt(sentence, target_triple):
    # client = openai.OpenAI(api_key=API_KEY_OPENAI)
    prompt = build_prompt(sentence, target_triple)
    messages = [
        {"role": "system", "content": "あなたは日本語の知識グラフ抽出パターン変換の専門家です。"},
        {"role": "user", "content": prompt}
    ]
    for attempt in range(RETRY_MAX):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"},
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
    return {"pattern": "", "error": "API failed or JSON error"}


response_schema = {
    "input": {"type": "STRING"},
    "target_triple": {"type": "STRING"},
    "pattern": {"type": "STRING"}
}

def ask_gemini(sentence, target_triple):
    prompt = build_prompt(sentence, target_triple)
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.0,       # 出力の一貫性重視
            max_output_tokens=512 # 出力トークン上限
        )
    )
    for attempt in range(RETRY_MAX):
        try:
            # Geminiのgenerate_contentは通常、文字列またはJSONを返す
            response = model.generate_content(prompt)
            # GeminiではJSONは .text または .candidates[0].text で取得
            response_text = getattr(response, "text", None)
            if response_text is None and hasattr(response, "candidates"):
                response_text = response.candidates[0].text
            return json.loads(response_text)
        except google.api_core.exceptions.ResourceExhausted:
            print("APIレート制限に達しました。10秒後に再試行します…")
            time.sleep(10)
        except json.JSONDecodeError:
            print("JSON解析エラー。5秒後に再試行します…")
            time.sleep(5)
        except Exception as e:
            print(f"APIエラー: {e}（{attempt+1}回目）")
            time.sleep(10)
    return {"pattern": "", "error": "API failed or JSON error"}


def ask_model(sentence, target_triple):
    if MODEL_NAME.startswith("gpt"):
        return ask_gpt(sentence, target_triple)
    elif MODEL_NAME.startswith("gemini"):
        return ask_gemini(sentence, target_triple)
    else:
        raise ValueError("未対応モデル名")
    

# --- CSV読み込み ---
if os.path.exists(OUTPUT_CSV):
    df = pd.read_csv(OUTPUT_CSV, dtype=str)
    print(f"既存のoutput.csvを再利用します。未処理行のみ再度リクエストします。")
else:
    df = pd.read_csv(INPUT_CSV, dtype=str)
    if "pattern" not in df.columns:
        df["pattern"] = ""

# --- メインループ ---
for idx, row in df.iterrows():
    sentence = str(row["sentence"])
    target_triple = str(row["target_triple"])
    pattern_existing = str(row.get("pattern", ""))
    log_entry = {"id": row.get("id", idx), "input": sentence}

    if pattern_existing and pattern_existing != "nan":
        print(f"[{idx}] スキップ: pattern既存-> {pattern_existing}")
        continue

    print(f"[{idx}] 送信: {sentence}, {target_triple}")
    result = ask_model(sentence, target_triple)
    pattern = result.get("pattern", "")
    error = result.get("error", "")

    df.at[idx, "pattern"] = pattern

    log_entry.update({
        "pattern": pattern,
        "error": error,
        "timestamp": datetime.now().isoformat()
    })
    log_data["results"].append(log_entry)
    if error:
        log_data["errors"].append(log_entry)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    time.sleep(SLEEP_TIME)

log_data["end_time"] = datetime.now().isoformat()
with open(log_filename, "w", encoding="utf-8") as f:
    json.dump(log_data, f, ensure_ascii=False, indent=2)

print(f"処理完了！出力: {OUTPUT_CSV}, ログ: {log_filename}")
