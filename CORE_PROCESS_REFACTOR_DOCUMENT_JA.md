# CORE_PROCESS_REFACTOR_DOCUMENT_JA

本ドキュメントは `MAIN_PIPELINE_JA.md` の仕様に合わせて、
**実験可能な最小変更**に絞って「どのコードを改良・修正・追加・削除するか」を明確化した改訂版です。

---

## 1. 目的（MAIN_PIPELINE_JA.md 準拠）

- パターンASTは **JSON入力 → コンパイル済み表現** に統一
- 入力CSVは **`data/T2KGB_JA/target_data/` 配下**を使用
- 依存解析・CKY表キャッシュは **増分・圧縮・部分読込**へ最適化
- candidate / verified の出力は **同一列名**で比較可能にする

---

## 2. 実験可能にするための最小変更（一覧）

### 2.1 追加（新規ファイル）
- **`src/modules_core/pattern_compiler.py`**
  - JSON/JSONL の AST を読み込み、実行向けにコンパイルするユーティリティ
  - 入力: `patterns_ast.json` / JSONL
  - 出力: コンパイル済み AST エントリ（`literal_list`, `parallel_var_count`, `ast_uid`, `var_count` 付き）
- **`src/modules_core/cache_store.py`**
  - 依存解析キャッシュ / CKYキャッシュの分割保存・部分読込を行うストレージ層

### 2.2 修正（既存ファイル）
- **`src/main.py`**
  - パターンロード元を `patterns.index.json` + JSON AST に切替
  - 入力CSVの探索先を `data/T2KGB_JA/target_data/` に変更
  - 依存解析・CKYキャッシュの読み書きを `cache_store.py` に委譲
  - candidate/verified 出力の列名を統一し、`stage` 列で区別
- **`src/modules_core/cky_table.py`**
  - CKY表生成を「文単位」で出力できるように調整
- **`src/modules_core/bunsetu.py`**
  - 依存解析の結果を文単位で返却し、キャッシュ層へ保存しやすい形式に調整

### 2.3 削除/無効化（最小限）
- **`patterns_ast.pkl.gz` 直接読み込みロジック**
  - JSON AST を起点にするため、現行のPkl直読みは実験対象から除外

---

## 3. ファイル別の詳細変更（最小限で実験可能にする）

### 3.1 `src/main.py`
**改良・修正**
- **パターン入力**
  - `patterns.index.json` を最初に読み込んで **対象 pattern_id を確定**
  - `patterns_ast.json` / JSONL から該当パターンを読み込み → `pattern_compiler.py` でコンパイル
- **入力CSV**
  - `data/T2KGB_JA/target_data/` のCSVを列挙し、順次処理できるようにする
- **キャッシュ**
  - `cache_store.py` を経由して依存解析・CKY表を **文単位で保存/読み込み**
  - 既存の巨大JSON一括ロードは実験対象から外す
- **出力**
  - candidate/verified ともに同一列名スキーマ
  - 例: `id`, `sentence`, `ontology_id`, `relation_ja`, `pid`, `prompt_id`, `prompt_name`,
    `domain_arg`, `range_arg`, `domain_concept_ja`, `range_concept_ja`, `verdict`, `ast_uid`, `stage`
  - candidate には `stage=candidate`, verified には `stage=verified`

**削除/置換**
- `AST_PICKLE` を参照する箇所を削除
- `INPUT_SENT_CSV` を固定パス参照する箇所を削除

---

### 3.2 `src/modules_core/pattern_compiler.py`（新規）
**追加理由**
- JSON AST を読み込んだだけでは実行が遅くなるため、
  **事前にコンパイル済み表現**（literal索引など）を作成して高速利用する必要がある

**最低限の責務**
- JSON/JSONL 読込
- pattern_id による抽出
- `literal_list`, `parallel_var_count`, `ast_uid`, `var_count` を付与
- `var_count` → AST のインデックスを作成

---

### 3.3 `src/modules_core/cache_store.py`（新規）
**追加理由**
- 現行 `dependency_analysis.json` / `dependency_analysis_with_cky.json` は
  **巨大な一括JSON読み込み**になり、実験時のボトルネックになる

**最低限の責務**
- 依存解析キャッシュ: 文単位で保存・読み込み
- CKY表キャッシュ: 文単位で保存・読み込み
- 圧縮（gzipなど）を前提
- 未解析文のみ更新できるAPI

---

### 3.4 `src/modules_core/cky_table.py`
**修正点**
- 文単位で CKY表を生成し、キャッシュ層へ渡せる形式に整形
- 表示や巨大出力はデフォルト無効化

---

### 3.5 `src/modules_core/bunsetu.py`
**修正点**
- 依存解析結果を **文ごとに独立した構造**で返却
- `cache_store.py` へ保存しやすい返却形式

---

## 4. 実験のための最小差分（導入順）

1. **`pattern_compiler.py` の追加**
   - JSON AST を読み込めるようにする
2. **`cache_store.py` の追加**
   - 依存解析・CKY表キャッシュを文単位にする
3. **`main.py` から AST/CSV/キャッシュ読込を差し替え**
   - `patterns.index.json` → JSON AST → コンパイル
   - `data/T2KGB_JA/target_data/` のCSVを読む
   - candidate/verified の同一列名出力
4. **`bunsetu.py` / `cky_table.py` の最小調整**

---

## 5. 最小限の削除・整理対象

- `patterns_ast.pkl.gz` 直読み
- `INPUT_SENT_CSV` 固定パス利用

---

## 6. 期待される実験効果

- JSON AST を起点にした **再現性の高いパターン運用**
- 巨大JSONの一括読み込みを避け、**メモリ負荷を削減**
- candidate / verified の比較を **同一スキーマで明確化**

---

## 注意事項（実験を安定して回すための最低限）

本リファクタでは「実験が止まらず、結果が再現できる」ことを最優先とする。以下は、実装時に落とすと高確率で事故るポイントである。

### 1. キャッシュキー設計（再現性の地雷）
文単位キャッシュは高速化に有効だが、同一キーの定義が曖昧だと結果が混ざる。

- キャッシュキーには最低限以下を含めること
  - `sentence_id`（またはデータの一意ID）
  - 解析器/モデルの識別子（例：`ginza_version`, `bert_model_name`, `llm_model_name` など）
  - 実験設定の主要パラメータ（例：閾値、優先度設定、プロンプトIDなど）
- 推奨：`cache_key = hash(sentence_id + model_versions + core_config)` のようにハッシュ化して衝突を避ける
- 目的：古いキャッシュや別設定のキャッシュが混入する事故を防止する

### 2. 部分書き込みの破損対策（落ちたら終わり問題）
gzipなどの圧縮ファイルを文単位で保存する場合、途中中断でファイル破損が発生しやすい。

- 書き込みは必ず「一時ファイル → rename（原子的置換）」で行う
  - 例：`xxx.jsonl.gz.tmp` に書き出し完了後 `xxx.jsonl.gz` へrename
- 並列実行時は特に重要（同名ファイルへの競合を避ける）
- 破損ファイル検出のため、読み込み時に例外発生したら「破損扱いで作り直し」できる設計にする

### 3. patterns.index.json と ASTファイル群の整合
`patterns.index.json`（pattern_idの正）と、`patterns_ast.json / patterns_ast.jsonl`（AST実体）が不整合になると静かに欠損する。

- 不整合パターンの基本方針を明示する（推奨：欠損はログしてスキップ）
  - indexにあるのにASTにない
  - ASTにあるのにindexにない
- 不整合件数は集計してレポートに残す（後で比較不能になるのを防ぐ）
- 「必須」扱いのpattern_idが欠けた場合はエラー停止にしてもよい（ただし方針を固定すること）

### 4. ast_uid の安定性（順序依存を禁止）
`ast_uid` を付与する場合、読み込み順や処理順に依存した採番は再現性を壊す。

- `ast_uid` は「内容に基づく安定ID」を推奨
  - ASTを正規化した表現（例：キー順固定、不要情報削除）を作る
  - その文字列に対してハッシュ（SHA-1/256等）を取り `ast_uid` とする
- 目的：同一ASTが常に同一IDになり、比較・重複除去・追跡が可能になる

### 5. 出力スキーマの必須列固定（後段が死ぬのを防ぐ）
列名の統一（candidate/verifiedの揺れ解消）だけでなく、後段の評価・集計のため「必ず出す列」を固定する。

- 空でも必ず出す列（例）
  - `sentence_id`
  - `sentence`
  - `stage`（e.g., `candidate` / `verified`）
  - `pattern_id`
  - `ast_uid`
  - `verdict`（採用/不採用/保留など）
  - `prompt_id` / `prompt_name`（LLMや判定ロジックに紐づくもの）
- 目的：列欠損による集計スクリプトの破綻を防ぐ

---