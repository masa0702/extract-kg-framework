# SELECT_MODE_MAIN_PIPELINE.md（仕様確定：select_mode_main）

本ドキュメントは `src/select_mode_main.py` の仕様です。  
`MAIN_PIPELINE_JA.md`（現行 `src/main.py` の仕様）をベースに、**モード切替（アブレーション）**と**matchキャッシュ**を追加した「確定版」仕様をまとめます。

---

## 1. 目的

- `src/main.py`（既存実験コード）は安定版として維持しつつ、次の3モードを自由に切替できる実行入口を追加する。
  - `default`
  - `no_verification`（オントロジー検証OFF）
  - `no_parallel_verification`（並列構造検証OFF）
- **CKY解析＋パターンマッチング＋candidate列挙（検証前）までを共通化**し、その中間結果（match cache）を保存して、モード変更時に重い再計算を避ける。

---

## 2. 実行スクリプトとCLI

実行入口: `src/select_mode_main.py`

### 2.1 CLI（全引数に default あり）

- `--mode`（default: `default`）: `default|no_verification|no_parallel_verification`
- `--pattern_index_json`（default: 環境変数 `PATTERN_INDEX_JSON` or `data/patterns/patterns.index.json`）
- `--pattern_jsonl`（default: 環境変数 `PATTERN_JSONL` or `data/patterns/patterns.jsonl`）
- `--input_jsonl_dir`（default: 環境変数 `INPUT_JSONL_DIR` or `data/T2KGB_JA/(test_)target_data`）
- `--results_root`（default: 環境変数 `RESULTS_ROOT` or `results/extract_pred_arg_pair`）
- `--run_tag`（default: `auto`）
  - `auto` の場合: `YYYYMMDD_HHMMSS__mode-<mode>` を採番（UTC）
- `--cache_mode`（default: `reuse`）: `reuse|refresh`
  - `reuse`: matchキャッシュが有効なら再利用
  - `refresh`: matchキャッシュを再生成（該当文のみ上書き）
- `--export_candidate_slim`（default: `1`）: `0/1`
- `--export_ast_repr`（default: 環境変数 `EXPORT_AST_REPR` と同等）: `0/1`

### 2.2 既存の環境変数互換（main.py 相当）

- `GPU_WORKERS`（default: `1`）
- `GPU_TIMEOUT_SEC`（default: `1800`）

※その他、LLM/judge/endpoint などは既存の `LLMJP_*` / `LLMJP_ONTO_*` の設定に従う。

---

## 3. モード仕様（確定）

### 3.1 共通（全モード）

- dep/cky キャッシュは現行と同様に利用（文単位gzip）。
- matchキャッシュ（後述）を **文単位**で生成/再利用し、そこから candidate を生成する。
- `candidate` 出力は **全モードで同一意味**（検証前の候補。parallel/ontologyのON/OFFで変えない）。

### 3.2 `default`

- candidate生成: 実行
- 並列構造検証（ParallelJudgeLLMJP）: 実行
- オントロジー検証（prompt LLM）: 実行
- verified生成: 実行（並列検証に通った match のみを対象）

### 3.3 `no_verification`

- candidate生成: 実行
- 並列構造検証: 実行（ログ目的）
- オントロジー検証: スキップ
- verified生成: 0行（ヘッダのみ）
- extracted_triples:
  - パターンで得た `X_values` と `Y_values` から triple を構築する（検証前の疑似抽出）
  - relation は ontology の relation label/alias に「部分一致」で吸収できたもののみ採用（吸収できない relation は triple を作らない）
  - Xが2つ以上の場合、各 unordered ペア (arg1,arg2) について両方向を出す:
    - (sub, rel, obj)=(arg1, relation, arg2) と (arg2, relation, arg1)

### 3.4 `no_parallel_verification`

- candidate生成: 実行
- 並列構造検証: スキップ
- オントロジー検証: 実行
- verified生成: 実行（並列検証に基づく除外を行わない）

### 3.5 preflight（起動時チェック）

- `default` / `no_parallel_verification` のときのみ、onto vLLM への到達性（`/models`）をチェックする。
- `no_verification` では preflight を行わない。

---

## 4. キャッシュ仕様

### 4.1 dep/cky キャッシュ（既存）

- `results_root/<dir>/<prefix>/cache/dep/`
- `results_root/<dir>/<prefix>/cache/cky/`

### 4.2 matchキャッシュ（追加）

#### 保存場所

- `results_root/<dir>/<prefix>/cache/match/`

#### 文キー

- sentence の sha1 をファイル名キーにする（既存 `SentenceCacheStore` と同様）。

#### payload（gzip JSON / 1文）

- `schema_version`: `1`
- `patterns_fingerprint`: パターン集合のfingerprint（後述）
- `sentence`: 原文
- `matches`: list
  - 1要素（1 match）あたり:
    - `ast_uid`, `pattern_id`, `pattern`, `var_count`, `parallel_var_count`, `literal_list`
    - `parallel_var_groups`: `[[X1,X2], ...]`（ParallelNode単位の変数名グループ）
    - `varmap_raw`, `varmap_clean`
    - `X_values`, `Y_values`（空・重複除去済み）

#### patterns_fingerprint

- `ast_dict` から全 `ast_uid` を集めてソートし、`"|"` で連結して sha1。
- 読み込み時に fingerprint 不一致なら cache miss 扱いで再生成する。

---

## 5. 出力（run単位）

### 5.1 出力ディレクトリ

- `results_root/<dir>/<prefix>/select_mode/<run_tag>/`
  - `logs/`（runログ）

※キャッシュ（dep/cky/match）は `select_mode` 配下ではなく、親の `cache/` を共用する。

### 5.2 出力ファイル

- `*_triples_candidate.csv`
- `*_triples_candidate_slim.csv`（`--export_candidate_slim=1` のとき）
- `<mode>_*_triples_verified.csv`
- `<mode>_*_ast_visualization.csv`
- `<mode>_*_prompt_log.jsonl`
- `<mode>_*_extracted_triples.jsonl`

### 5.3 並列検証ログ

- `select_mode/<run_tag>/logs/<mode>_<prefix>_parallel_verify.jsonl`

---

## 6. 実装メモ（重要な意図）

- matchキャッシュ生成時は **LLM（parallel/ontology）を呼ばない**。
  - これにより、`default` と `no_parallel_verification` の差分が再現できる。
- candidate は常に「検証前」なので、モード差分の比較は verified / prompt log / parallel log を見る。
