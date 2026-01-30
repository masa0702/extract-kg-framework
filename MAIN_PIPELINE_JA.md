# main.py 実行フロー詳細（現状仕様）

本ドキュメントは `src/main.py` の**現状の挙動**を日本語で詳細にまとめたものです。
各処理の入出力を示し、全体フローと合わせて説明します。

## 1. 全体像（ざっくり）
1. パターンAST(JSON)の読み込み → コンパイル → 高速利用
2. 入力CSVの読み込み
3. 依存解析キャッシュの更新
4. CKY表生成/読込（最適化キャッシュ）
5. 文ごとのCPU/GPU並列処理（GPU: CKY解析、CPU: フィルタ→マッチング→抽出）
6. 追加のオントロジー整合検証（relation + prompt 対応表）
7. 逐次CSV出力（抽出結果・可視化ログ・検証済みTriple）

---

## 2. 主要な入出力（仕様反映）

### 2.1 入力
- パターンAST（JSON）
  - `data/patterns/patterns_ast.json`（仕様名。実体は JSON / JSONL を想定）
  - 形式: list[dict] もしくは JSONL、各要素に `pattern_id`, `pattern`, `ast` などを含む
  - 受け取ったASTは「コンパイル済み表現」に変換し、以降は高速に参照できる形式で利用する
- 入力文 CSV
  - `data/T2KGB_JA/target_data/*.csv`
  - 必須列: `id`, `sent`
  - 任意列: `ontology_id` / `ontology` / `ontology_category` / `category`
- 依存解析キャッシュ（最適化版）
  - 仕様上は「増分更新」「圧縮」「部分読込」を前提とした形式に移行
- CKY表キャッシュ（最適化版）
  - 仕様上は「文ごとの分割保存」「必要最小限フィールドの保持」「圧縮」を前提
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
  - 変数マッピングやリテラル情報など
- 進捗/診断ログ
  - `../results/.../logs/*.csv`

---

## 3. ステップ別の詳細（入出力つき）

### 3.1 パターンASTのロード（JSON→コンパイル）
**入力**
- `data/patterns/patterns_ast.json`（仕様名。JSON / JSONL どちらでも可）

**処理**
- ASTをJSONから読み込み、パターンごとに以下のメタ情報を付与:
  - `literal_list`: ASTから抽出したリテラル
  - `parallel_var_count`: 並列変数数
  - `ast_uid`: ASTのハッシュID
- ASTを「コンパイル済み表現」に変換し、繰り返し利用に耐える形へ変換
  （例: ノード参照の正規化、頻出リテラル索引の事前構築など）
- `var_count` ごとに `ast_dict` を構築

**出力**
- `ast_dict`（`var_count` → パターンASTエントリ配列）

**利用コード/モデル**
- `src/main.py`
- `src/pattern/pattern_nodes.py`（`extract_literal_strings`, `count_parallel_variables`）

---

### 3.2 入力文CSVの読み込み
**入力**
- `data/T2KGB_JA/target_data/*.csv`

**処理**
- `sent` 列をユニーク化して対象文一覧を作成
- `ontology_id` 列が存在すれば文ごとに保持（なければ空文字）
- 環境変数 `DEFAULT_ONTOLOGY_ID` を補完として使用可能

**出力**
- 文一覧 `sentences`
- 文ごとのメタ情報 `rows` の作成準備

**利用コード/モデル**
- `src/main.py`（`pandas`でCSV読込）

---

### 3.3 依存解析キャッシュ更新（最適化仕様）
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
- `src/modules_core/bunsetu.py`（`DependencyAnalysis`）
  モデル: spaCy GiNZA `ja_ginza_bert_large`（失敗時 `ja_ginza` へフォールバック）

---

### 3.4 CKY表の生成/読込（最適化仕様）
**入力**
- 最適化済み依存キャッシュ

**処理**
- CKY表キャッシュも文単位で生成・保存
- 参照時は必要な文のみを読み込み

**出力**
- `cky_json_data`（文→CKY表＋文節情報）

**利用コード/モデル**
- `src/main.py`
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
- `recs`（抽出ペア）
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
- `ast_visualization.csv`

---

## 4. パターン入力仕様（確定事項）

### 4.1 仕様方針
今後のマッチングは **`data/patterns/patterns.index.json` を唯一のパターンカタログ** として扱う。
`src/main.py` は **pattern.index.json を参照してマッチング対象パターンを決定する** 仕様に固定する。
パターンASTは **JSON入力→コンパイル済み表現** を標準とし、実行時はコンパイル済み表現を優先利用する。

### 4.2 パターン解決のルール（仕様）
- `patterns.index.json` の `patterns[*].pattern_id` を基準として処理対象を確定
- pattern本体は `data/patterns/patterns.jsonl` から `pattern_id` で取得する
- ASTは **JSONとして入力**し、読み込み時にコンパイルして高速利用する
  - 例: `data/patterns/patterns_ast.json`（pattern_id とASTを持つJSON/JSONL）
  - `ast_sig` による紐付け・キャッシュの保持は許容（ただし入力はJSONを起点とする）

### 4.3 旧方式の扱い
- 現状 `patterns_ast.pkl.gz` を直接ロードしているが、
  **今後は `patterns.index.json` を起点とし、ASTはJSON入力→コンパイルで供給する**。
- 実装変更は後続タスクで行う。

---

## 5. 注意点（仕様反映後）
- 入力CSVは `data/T2KGB_JA/target_data/` 配下に配置する
- GPU/CPUプロセスはタイムアウト時に強制終了される
- LLM検証（並列判定/オントロジー判定）は vLLM(OpenAI互換) の `chat_json` を利用
- candidate / verified は**同一列名**の出力として分離される

---

以上が `src/main.py` の現状挙動と、パターン入力仕様の確定方針です。
