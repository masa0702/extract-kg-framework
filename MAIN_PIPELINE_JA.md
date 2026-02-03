# main.py 実行フロー詳細（現状仕様 + 検証仕様の確定版）

本ドキュメントは `src/main.py` の**現状の挙動**を日本語で詳細にまとめたものです。
各処理の入出力を示し、全体フローと合わせて説明します。
また、E2E運用で問題になりやすい「オントロジー整合検証（LLMプロンプト判定）」については、
**仕様を確定した形**で整理し、現状実装との差分（修正方針）も併記します。

## 1. 全体像（ざっくり）
1. パターン定義の読み込み → AST化 → コンパイル → 高速利用
2. 入力JSONLの読み込み
3. 依存解析キャッシュの更新
4. CKY表生成/読込（最適化キャッシュ）
5. 文ごとのCPU/GPU並列処理（GPU: CKY解析、CPU: フィルタ→マッチング→抽出）
6. 追加のオントロジー整合検証（relation + prompt 対応表）
7. 逐次CSV出力（candidate/verified・可視化ログ）

---

## 2. 主要な入出力（現状実装）

### 2.1 入力
- パターン定義
  - `data/patterns/patterns.index.json`（対象パターンの `pattern_id` を列挙）
  - `data/patterns/patterns.jsonl`（`pattern_id` と `pattern` を持つJSONL）
- 入力文 JSONL
  - `data/T2KGB_JA/target_data/*.jsonl`
  - 必須列: `id`, `sent_ja`（無い場合は `sent` を利用）
  - 任意列: `ontology_id` / `ontology` / `ontology_category` / `category`
- 依存解析キャッシュ（文単位gzip）
  - `../results/.../<prefix>/cache/dep/*.json.gz`
- CKY表キャッシュ（文単位gzip）
  - `../results/.../<prefix>/cache/cky/*.json.gz`
- プロンプト管理
  - `prompts/prompts.json`
  - `prompts/relation_prompt_map.json`
- オントロジー
  - `ontology/*.json`

### 2.2 出力
- 候補（candidate）と検証済み（verified）の出力は**同一の列名**で揃える
  - 例（同一スキーマ）: `id`, `sentence`, `ontology_id`, `relation_ja`, `pid`, `prompt_id`, `prompt_name`,
    `domain_arg`, `range_arg`, `domain_concept_ja`, `range_concept_ja`, `verdict`, `ast_uid`, `stage`
  - `stage` は `candidate` / `verified` を明示（差分を明確化）
  - candidate 側は `prompt_id` / `verdict` などが空の場合がある
- 候補（candidate）については、閲覧用に最小列へ絞った `candidate_slim` も併せて出力する（既定ON）
  - `../results/.../<prefix>_triples_candidate_slim.csv`
  - 列: `id,sentence,ontology_id,relation_ja,domain_arg,range_arg,ast_uid,stage`
- AST可視化ログ
  - `../results/.../<prefix>_ast_visualization.csv`
  - 変数マッピングやリテラル情報、`pattern_id`/`pattern` を含む
  - `EXPORT_AST_REPR=1` のときは `ast_repr` を追加
- 検証用プロンプトログ
  - `../results/.../<prefix>_prompt_log.jsonl`
- 進捗/診断ログ
  - `../results/.../logs/*.csv`

---

## 3. ステップ別の詳細（入出力つき）

### 3.1 パターン定義のロードとコンパイル
**入力**
- `data/patterns/patterns.index.json`
- `data/patterns/patterns.jsonl`

**処理**
- `patterns.index.json` から対象 `pattern_id` を確定
- `patterns.jsonl` から該当パターンを読み込み、`PatternParser` でAST化
- パターンごとに以下のメタ情報を付与:
  - `literal_list`: ASTから抽出したリテラル
  - `parallel_var_count`: 並列変数数
  - `ast_uid`: ASTのハッシュID
- `var_count` ごとに `ast_dict` を構築

**出力**
- `ast_dict`（`var_count` → パターンASTエントリ配列）

**利用コード/モデル**
- `src/main.py`
- `src/modules_core/pattern_compiler.py`
- `src/pattern/pattern_parser.py`
- `src/pattern/pattern_nodes.py`

---

### 3.2 入力文JSONLの読み込み
**入力**
- `data/T2KGB_JA/target_data/*.jsonl`

**処理**
- `sent_ja`（無い場合は `sent`）をユニーク化して対象文一覧を作成
- `ontology_id` 系列が無ければ `DEFAULT_ONTOLOGY_ID` を使用
- さらに `ont_*.jsonl` のファイル名から `ontology_id` を補完
  - 補完規則: `ont_1_movie_*.jsonl` → `ont_1_movie`（先頭3トークン）
  - 注意: 補完された `ontology_id` が `prompts/relation_prompt_map.json` や `ontology/*_ontology_trans_ja.json` に存在しない場合、
    relation/prompt の解決に失敗しやすく、結果として抽出できる relation が極端に偏ることがある。
    その場合は JSONL 側に正しい `ontology_id` を入れるか、`DEFAULT_ONTOLOGY_ID` を指定する。

**出力**
- 文一覧 `sentences`
- 文ごとのメタ情報 `rows` の作成準備

**利用コード/モデル**
- `src/main.py`（JSONL読込）

---

### 3.3 依存解析キャッシュ更新（文単位gzip）
**入力**
- 最適化済み依存キャッシュ
- 新規文リスト

**処理**
- 既存キャッシュに無い文のみ依存解析を実行（増分更新）
- キャッシュは「文単位の分割」「圧縮」「必要フィールドのみ保持」などを前提

**出力**
- `dep_data`（文→依存解析結果）

**利用コード/モデル**
- `src/main.py`
- `src/modules_core/cache_store.py`
- `src/modules_core/bunsetu.py`（`DependencyAnalysis`）
  モデル: spaCy GiNZA `ja_ginza_bert_large`（失敗時 `ja_ginza` へフォールバック）

---

### 3.4 CKY表の生成/読込（文単位gzip）
**入力**
- 最適化済み依存キャッシュ

**処理**
- CKY表キャッシュも文単位で生成・保存
- 参照時は必要な文のみを読み込み

**出力**
- `cky_json_data`（文→CKY表＋文節情報）

**利用コード/モデル**
- `src/main.py`
- `src/modules_core/cache_store.py`
- `src/modules_core/cky_table.py`（`CkyTable`）

---

### 3.5 GPUステージ（CKY解析）
**入力**
- 文単位の `cky_table`, `clauses`

**処理**
- `CKYAnalyzer` をGPUプロセスで実行し `cky_dep` を生成
- タイムアウト超過でプロセスkill

**出力**
- `cky_dep`
- 診断ログ: `gpu_timing.csv`, `gpu_done.csv`, `gpu_timeout.csv`

**利用コード/モデル**
- `src/main.py`
- `src/modules_bert/bert_modules.py`（`CKYAnalyzer`）
  モデル: `tohoku-nlp/bert-base-japanese-v3`（Mask）, ローカル依存修正モデル
  `./models/output_bert_dependency_bunsetsu_ver3.0/depbert_bunsetsu_20260117_072956/final_model`

---

### 3.6 CPUステージ（フィルタ→マッチング）
**入力**
- `cky_dep`
- `clauses`
- `ast_dict`

**処理（主な流れ）**
1) **候補ASTの列挙**
   - 文節数Bに基づき `var_count` が 2..B のパターンを収集

2) **リテラル/並列フィルタ（粗）**
   - リテラルの順序一致判定
   - 並列キー出現数チェック

3) **候補span生成**
   - リテラル位置から (i,j) span を推定
   - spanを拡張してCKYMatcher用の候補セルを作成

4) **CKYMatcherによるマッチング**
   - `match_table` の結果から変数マッピング取得
   - 変数マッピングの重複を除去

5) **並列判定（LLM）**
   - 並列変数がある場合は `ParallelJudgeLLMJP` で妥当性確認

6) **X/Y変数の抽出**
   - `X*` を引数候補、`Y*` を relation 候補として抽出

7) **抽出ペアの生成（candidate）**
   - `X×Y` の組み合わせで candidate 出力に追記（列名は verified と同一）
   - `relation_ja` は、`Y` の表層（例: `監督し`）をそのまま出すのではなく、
     可能な限りオントロジー定義の正規形（canonical label）へ寄せて出力する
     - 例: `監督し` → `監督`（`label_wiki_ja` 優先）
   - candidate 段階では domain/range/prompt が未確定のため、`pid/prompt_id/verdict` 等は空のままにする

8) **可視化ログの記録**
   - AST UID, リテラル, 変数マッピングを `ast_visualization.csv` に追記

**出力**
- `candidates`（候補出力）
- `vis_rows`（可視化ログ）

**利用コード/モデル**
- `src/main.py`
- `src/modules_core/matcher.py`（`CKYMatcher`）
- `src/pattern/pattern_nodes.py`
- `src/llm/parallel_judge.py`（`ParallelJudgeLLMJP`）
  モデル: `llmjp-13b`（vLLM OpenAI互換）

---

### 3.7 オントロジー整合検証
**入力**
- relation（`Y*`）
- argument（`X*`）
- `relation_prompt_map.json`
- `prompts.json`
- `ontology/*.json`

**処理**
0) **前提**
   - 実行時に Wikidata API は呼ばない（外部ネットワーク依存を避ける）。
   - relation/prompt/ontology はローカルの
     `prompts/relation_prompt_map.json`, `prompts/prompts.json`, `ontology/*_ontology_trans_ja.json`
     から解決する。

1) **relation に対応するマッピング行（row）を解決**
   - 入力: `relation`（`Y*` の文字列）, `ontology_id`
   - 解決順（安全側の優先順位）:
     1. `relation` が `P57` のような PID 形式なら PID として解決（`pid`）
     2. `(ontology_id, predicate_ja)` の完全一致で解決（`relation_prompt_map.json`）
     3. ontology の `relations.label_ja / label_wiki_ja / label` を alias として許容し、完全一致で解決
     4. それでもダメなら、ontology スコープ内でファジー一致で解決
        - 既定 `RELATION_FUZZY_THRESHOLD=0.78`
        - ベストと2番手が僅差のときは曖昧として棄却（誤マッチ抑止）
   - `Y` 由来で解決できない場合は、フォールバックとして文中の部分文字列一致で relation 候補を拾って再解決する
     - `predicate_ja` だけでなく ontology の `label_ja/label_wiki_ja` も探索対象

2) **prompt の選択**
   - 対応表（row）から `prompt_id` を取得し、`prompts.json` からテンプレートを引く

3) **domain/range 概念の解決**
   - row の domain/range が QID のみの場合、ontology ラベル辞書から日本語ラベルへ展開する

4) **プロンプト種別の判定（prompts.json）**
   - **2値（0/1）・side別判定タイプ**: `01,15,17,21`
     - `side=domain` / `side=range` を分けて判定する前提
   - **3値（0/1/2）・方向判定タイプ**: `04`
     - `arg1/arg2` を同時に与え、`verdict=1` で (arg1=domain,arg2=range)、`verdict=2` で逆、`verdict=0` で棄却
   - **2値（0/1）・同義語生成ゲート**: `10`
     - プロンプト本文に「domain/range の割当は確定しなくてよい」と明記されているため、
       **verified triple の採否判定には使わない**（ゲートとしてのログ用途に限定）

5) **X引数（argument候補）の扱い**
   - `x_values` は「そのパターンマッチで得られた X 変数の値集合」（文全体からの全候補ではない）
   - `x_values` から (arg1,arg2) のペアを作り、プロンプト検証に渡す

   5.1) Xが2つ（基本ケース）: `x_values=[A,B]`
   - 2値（side別）プロンプトの場合（01/15/17/21）:
     - **domain/range のペアが成立したときのみ** verified を生成する（単独side成立では採用しない）
     - 判定は「Aをdomain→Bをrange / Bをdomain→Aをrange」の順で試行する
   - 3値（方向判定）プロンプトの場合（04）:
     - `arg1=A, arg2=B` を1回投げる
       - verdict=1 → verified: (domain=A, range=B)
       - verdict=2 → verified: (domain=B, range=A)
       - verdict=0 → 棄却

   5.2) Xが3つ以上（例: `x_values=[A,B,C,...]`）
   - 並列判定（`ParallelJudgeLLMJP`）が通った要素は「並列グループ」として扱い、
     同一グループ内のペア（例: AとBが並列）は domain/range の候補ペアとして禁止する
   - それ以外の組合せ（例: (A,C), (B,C)）は検証対象

6) **relation の canonical 化（表層を ontology に吸収）**
   - verified 出力の `relation_ja` は、可能な限り ontology に定義された関係ラベルへ正規化する
   - canonical は ontology の `relations` を参照し、`label_wiki_ja` → `label_ja` → `label` の優先順で選ぶ
   - 例: `監督し` / `監督した` → `監督`
   - デバッグ用として `*_prompt_log.jsonl` には `relation_raw`（元の表層）も併記する

7) **重複排除（dedup）とログ**
   - LLM 検証は高コストなので、同一入力（同一 prompt/arg の組）を文単位で重複して投げない（dedup）
   - `LOG_DUPLICATE_SKIPS=1` のとき、スキップした重複呼び出しも `prompt_log.jsonl` に `skipped_duplicate=true` で残す
   - 注意: 現状は **検証ペア数の上限制御は入っていない**ため、`X` が多いパターンでは組合せ爆発し得る

8) **起動時チェック（preflight）と fail-fast**
   - `ONTOLOGY_VERIFY_PREFLIGHT=1`（既定）: onto vLLM の `/models` を確認し、到達不可/モデル不一致を開始直後に検出する
   - `ONTOLOGY_VERIFY_STRICT=1`（既定）: LLM呼び出し失敗を握りつぶさず例外で停止（結果が空で終わるのを防ぐ）
   - `prompt_log.jsonl` には `model_used/base_url/request_url` や `error` が記録され、原因追跡ができる

**出力**
- verified 出力に追記（列名は candidate と同一）
- 検証用プロンプトを JSONL で保存
  - `prompt_log.jsonl` には **verdict を必ず記録**する（どの判定が 0/1/2 だったかを後から追えることが重要）

**利用コード/モデル**
- `src/main.py`
- `src/modules_core/ontology_verify.py`（relation→prompt解決、LLM判定）
- `src/tools/relation_prompt_map.py`（対応表生成ロジック）
  モデル: `llmjp-13b`（vLLM OpenAI互換）

---

### 3.8 結果の逐次書き込み
**入力**
- CPUステージの結果

**処理**
- 結果CSVへ追記
- ログCSVへ追記

**出力**
- candidate / verified（同一スキーマで別出力）
  - `../results/.../<prefix>_triples_candidate.csv`
  - `../results/.../<prefix>_triples_verified.csv`
- `ast_visualization.csv`
- `prompt_log.jsonl`

---

## 4. パターン入力構成（現状実装）
- `patterns.index.json` の `patterns[*].pattern_id` を基準として処理対象を確定
- `patterns.jsonl` から該当 `pattern_id` の `pattern` を取得し、実行時にAST化・コンパイル

---

## 5. 注意点（現状実装）
- 入力JSONLは `data/T2KGB_JA/target_data/` 配下に配置する
- 主要パスは環境変数で上書き可能（`PATTERN_INDEX_JSON`, `PATTERN_JSONL`, `INPUT_JSONL_DIR`, `RESULTS_ROOT`）
- GPU/CPUプロセスはタイムアウト時に強制終了される
- LLM検証（並列判定/オントロジー判定）は vLLM(OpenAI互換) の `chat_json` を利用
- candidate / verified は**同一列名**の出力として分離される
- 推奨: 研究運用での再現性のため、以下を固定/明示する
  - Hugging Face キャッシュ（例: `HF_HOME=/workspace/.cache/huggingface`）
  - オフライン実行（必要に応じて）: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`
  - CKY候補の組合せ爆発を防ぐ上限（デフォルトで有効）:
    - `CKY_MAX_CHILD_CANDIDATES`（セル結合時に参照する左右候補数の上限）
      - **デフォルト: 6**
      - 小さくすると高速化する一方、候補を捨てるため再現率が下がり得る
    - `CKY_MAX_CELL_CANDIDATES_TOTAL`（各セルの候補総数上限）
      - **デフォルト: 64**
      - 小さくすると高速化する一方、セル内候補が減りパターンマッチ材料が減る
    - 目的: 1文でも起き得る候補の組合せ爆発（左候補×右候補の直積の連鎖）を抑制し、処理停止/極端な遅延を防ぐ
    - 運用: 速度/再現率トレードオフのため、研究/本番で値を固定して比較することを推奨

---

以上が `src/main.py` の現状挙動と、パターン入力仕様の確定方針です。
