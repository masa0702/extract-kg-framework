import pandas as pd
import time
import json
import os
import glob
from tqdm import tqdm


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
# MODEL_NAME = "gpt-4.1-nano"
# MODEL_NAME = "gpt-4.1-mini"
MODEL_NAME = "gpt-4o-mini"
# MODEL_NAME = "gemini-2.0-flash"

INPUT_CSV = os.path.join(input_dir, f"test.csv")
OUTPUT_CSV = os.path.join(output_dir, f"test_{MODEL_NAME}.csv")


RPM_LIMIT = 15
RETRY_MAX = 3
SLEEP_TIME = 60 / RPM_LIMIT + 0.2


def get_ontology_concepts(ontology):
    ont_concepts = ""
    for onto in ontology['concepts']:
        ont_concepts += onto['label'] + "/" + onto['label_ja'] + ", "
    return ont_concepts[0:-1]


def get_concept_label(ontology, concept):
    for onto in ontology['concepts']:
        if onto['qid'] == concept:
            return onto['label']
        
        
def get_ontology_relations(ontology):
    ont_rels = ""
    onto_rel_strings = list()
    for onto in ontology['relations']:
        ont_rel = onto['label'] + "/" + onto['label_ja']
        ont_rel = ont_rel.replace(" ", "_")
        ont_dom = onto['domain']
        ont_range = onto['range']
        ont_domain = get_concept_label(ontology, ont_dom)
        ont_range = get_concept_label(ontology, ont_range)

        if ont_rel == None:
            continue
        if ont_domain == None:
            ont_domain = ""
        if ont_range == None:
            ont_range = ""

        onto_rel_strings.append(f"{ont_rel} ({ont_range})\n")

        ont_rels += ont_rel + "(" + ont_range + "), "

    return ont_rels[0:-2]


def get_sent_prompt(sentence):
    test_prompt = "\n\nInput Sentence: " + sentence
    test_prompt += """
                    \nOutput json: 
                    {\n
                        "input": "input sentence",\n'
                        "target_ontlogy": "ontlogy name",\n'
                        "pattern": "pattern converted from sentence"\n'
                    }\n
                    """
    return test_prompt


def listup_by_ontlogy_id(csv_path, target_ontlogy_id):
    # CSV読み込み
    df = pd.read_csv(csv_path, sep=',')
    # 条件で抽出
    matched = df[df['ontlogy_id'] == target_ontlogy_id]
    # 各リストに保存
    sentences = matched['sentence'].tolist()
    bunsetus = matched['bunsetu'].tolist()
    patterns = matched['pattern'].tolist()
    return sentences, bunsetus, patterns


def get_clause_defintion(sentence_list:list, bunsetu_list:list) -> str:
    examples = f"""
        【文節の定義】
        文節の定義は「接頭辞＋自立語＋付属語・接尾辞」とします。
        ・0個以上の接頭辞、付属語・接尾辞
        ・複合名詞などでは自立語が複数の場合もある
        上記に従って、GiNZAの文節境界を採用します。
        ###GOOD_EXAMPLE（|は文節区切り）
        # 1「{sentence_list[0]}」->{bunsetu_list[0]}
        # 2「{sentence_list[1]}」->{bunsetu_list[1]}
        # 3「{sentence_list[2]}」->{bunsetu_list[2]}
        # 4「{sentence_list[3]}」->{bunsetu_list[3]}
        """
    return examples

def get_fewshot_examples(sentence_list:list, pattern_list:list) -> str:
    fewshot_examples = f"""
        【text->パターンへの変換例】
        ### GOOD_EXAMPLE
        #1「{sentence_list[0]}」->{pattern_list[0]}
        #2「{sentence_list[1]}」->{pattern_list[1]}
        #3「{sentence_list[2]}」->{pattern_list[2]}
        #4「{sentence_list[3]}」->{pattern_list[3]}
        """
    return fewshot_examples

            
def prepare_prompt(ontology: dict, train_sent: str) -> str:
    csv_path = "../data/ontology_fewshot_examples.csv"
    target_ontlogy_id = ontology["id"]
    sentence_list, bunsetu_list, pattern_list = listup_by_ontlogy_id(csv_path, target_ontlogy_id)
    # print(sentence_list)
    # print(bunsetu_list)
    # print(pattern_list)
    prompt_fixed = """
    次に示す仕様・オントロジー・変換例に従い、
    Input Sentenceから知識グラフの述語・目的語を抽出するための文型パターンに変換してください。
    - 知識グラフの主語の抽出は行いません\n
    - 知識グラフの述語・目的語は【パターン記述法】に従って、変数にすること\n
    - 可能な限りパターンを最小単位で作成すること\n
    - 複数パターンが必要な場合は","で分割すること\n
    - 文中にない表現をパターンに含めないこと\n
    - 出力は複数行可、順不同可。余計な説明や空行を入れないこと。\n
    - 出力はJSONのみで返してください\n"
    --- 以下に仕様・オントロジー・例示を示す ---\n
    """

    pattern_manual = """
    【パターン記述法】
    前提条件：パターン記述記号は文節単位で記述する
    [Xi]：目的語要素
    [Yi]：述語要素
    [Mi]：修飾語
        ◻︎　それぞれ（1 ≦ i）で１からi番目の要素
        ◻︎　リテラル以外の要素は[]で必ず囲む
    ・＊n：連体修飾がn文節分されたもの
    ・#n：連用修飾n文節分されたもの
    *もしくは#の後には数値（修飾回数）もしくは、構造を示す()が必ず出現する
    ・[Xi] & [Xi+1]&...&[Xi+n]：並列要素[Xi],[Xi+1],...[Xi+n]が並列構造をなす
        ◻︎　& ∈ {と｜、｜や｜および｜または}
        ◻︎　i ：並列要素数(1 ≦ i )
    ・-{品詞指定}
        ◻︎　ex. [X-サ変]
        ◻︎ 使える品詞辞書(動詞,名詞,サ変,形容詞)
    ・[*([〇〇1]&[△△2])]：並列要素〇〇1,△△2に対する修飾
    ・[*([M1]&[M2])〇〇]：要素〇〇に対する並列修飾
    ・[...]を, [〇〇-サ変]する：リテラルとしての文字列（を/する/などの助詞）
    """
    
    prompt = prompt_fixed
    prompt += 'CONTEXT:\n\n'
    prompt += 'Ontology Concepts: '
    ont_concepts = get_ontology_concepts(ontology)
    prompt += ont_concepts
    prompt += '\nOntology Relations: '
    prompt += get_ontology_relations(ontology)
    prompt += pattern_manual
    prompt += get_clause_defintion(sentence_list, bunsetu_list)
    prompt += get_fewshot_examples(sentence_list, pattern_list)
    prompt += get_sent_prompt(train_sent)

    return prompt


def load_json(src_file):
    with open(src_file, 'r', encoding='utf-8') as f:
        source = json.load(f)
        return source


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
    "target_ontlogy": {"type": "STRING"},
    "pattern": {"type": "STRING"}
}

def ask_gemini(prompt):
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


def ask_model(prompt):
    if MODEL_NAME.startswith("gpt"):
        return ask_gpt(prompt)
    elif MODEL_NAME.startswith("gemini"):
        return ask_gemini(prompt)
    else:
        raise ValueError("未対応モデル名")
    

def find_ont_id_from_path(path):
    # パス全体を _ で分割して順に確認
    parts = []
    parts.extend(os.path.basename(path).split('_'))
    for p in path.split('/'):
        parts.extend(p.split('_'))
    for i in range(len(parts) - 2):
        if parts[i] == "ont" and parts[i+1].isdigit():
            # 例: ont_1_movie → 1_movie_ontology
            return f"{parts[i+1]}_{parts[i+2]}_ontology"
    # サブ文字列でont_n_xxにマッチする場合
    for s in path.split('/'):
        if s.startswith("ont") and len(s.split('_')) >= 3 and s.split('_')[1].isdigit():
            base = s.split('_')
            return f"{base[1]}_{base[2]}_ontology"
    return None


def load_ontology_by_id(ontology_dir, ont_id):
    # オントロジーJSONファイルをIDで検索して読み込み
    for fname in os.listdir(ontology_dir):
        if ont_id in fname and fname.endswith('.json'):
            with open(os.path.join(ontology_dir, fname), 'r', encoding='utf-8') as f:
                return json.load(f)
    raise FileNotFoundError(f"{ont_id}に該当するオントロジーjsonがありません")


def sanitize_filename(filename):
    # ファイル名から拡張子を除く
    return os.path.splitext(os.path.basename(filename))[0]

if __name__ == "__main__":
    ontology_list = [
        "ont_1_movie",
        "ont_2_music",
        "ont_3_sport",
        "ont_4_book",
        # "ont_5_military",
        "ont_6_computer"
    ]
    
    target_file_kind = "parallel_subject-object_ja_sent_po_pair.csv"
    # target_file_kind = "parallel_subject-predicate_ja_sent_po_pair.csv"
    input_csv_list = []
    for i in range(len(ontology_list)):
        path = f"../data/sent_po_pair_data/{ontology_list[i]}/{target_file_kind}"
        input_csv_list.append(path)
        
    ontology_dir = "../data/translate_ontlogy_en_ja"

    for csv_path in input_csv_list:
        print(f"処理中: {csv_path}")
        save_dir = f"../data/{sanitize_filename(target_file_kind)}"
        os.makedirs(save_dir, exist_ok=True)
        print(save_dir)
        df = pd.read_csv(csv_path, sep=",")
        ont_id = find_ont_id_from_path(csv_path)
        if ont_id is None:
            print(f"ont_id特定不可: {csv_path}")
            continue
        ontology = load_ontology_by_id(ontology_dir, ont_id)

        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="進捗"):
            sent_ja = row['sent_ja']
            po_ja = row['po_ja']
            _id = row['id']
            prompt = prepare_prompt(ontology, sent_ja)
            pattern = ""
            try:
                response = ask_model(prompt)
                pattern = response.get('pattern', '')
            except Exception as e:
                print(f"API/解析エラー: {e}")
                pattern = "error"
            results.append({
                "id": _id,
                "ont_id": ont_id,
                "sent_ja": sent_ja,
                "po_ja": po_ja,
                "pattern": pattern
            })
            # print(results)
            # exit()

        # 保存ファイル名例：result_元ファイル名.csv
        in_filename = sanitize_filename(csv_path)
        out_filename = f"{ont_id}_{in_filename}.csv"
        out_path = os.path.join(save_dir, out_filename)
        df_result = pd.DataFrame(results)
        df_result.to_csv(out_path, index=False, encoding="utf-8")
        print(f"→ 保存: {out_path}")
