# main.py 実行フロー詳細（現状仕様）

本ドキュメントは `src/main.py` の**現状の挙動**を日本語で詳細にまとめたものです。
各処理の入出力を示し、全体フローと合わせて説明します。

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
1) relation に対応するマッピング行を解決
   - relationが `Pxxx` なら pidで解決
   - 日本語述語なら `predicate_ja` で解決
   - `ontology_id` があれば優先

2) 対応表から prompt_id を取得しプロンプトを選択

3) domain/range概念を決定
   - conceptがQIDのみの場合はオントロジーから日本語ラベルを引く

4) プロンプトごとの検証
   - arg1/arg2タイプ（prompt_id=04,10）は、2引数を組み合わせて判定
   - side固定タイプ（01,15,17,21）は domain/range それぞれで判定

5) verdict!=0 のもののみ Triple として採用

**出力**
- verified 出力に追記（列名は candidate と同一）
- 検証用プロンプトを JSONL で保存

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

---

以上が `src/main.py` の現状挙動と、パターン入力仕様の確定方針です。
