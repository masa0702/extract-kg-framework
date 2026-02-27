# 提案手法の実験設定（ver11.0 / extract_pred_arg_pair）

本ドキュメントは、`ver11.0` 系の **extract_pred_arg_pair** 実験の設定を、
再現性の観点で簡潔にまとめたものです。実際の評価は `src/eval_wiki_id.py` を用いた
バッチ評価の出力に基づきます。

## 1. 実験対象（タスク）
- 日本語テキストから **述語-項（predicate-argument）ペア**を抽出し、
  オントロジーに整合する triple を出力・評価する。
- 抽出対象は Text2KGBench（和訳）由来の JSONL 入力。

## 2. データ構成
- 入力（ターゲット文）
  - `data/T2KGB_JA/target_data/*.jsonl`
  - 必須列: `id`, `sent_ja`（無い場合は `sent` を利用）
  - 任意列: `ontology_id` / `ontology` / `ontology_category` / `category`
- オントロジー
  - `ontology/*.json`
- プロンプト
  - `prompts/prompts.json`
  - `prompts/relation_prompt_map.json`
- Gold（評価用）
  - 推奨: `data/T2KGB_JA/gold_data/*.jsonl`
  - 代替: `data/T2KGB_JA/all_wikidata_tekgen_ground_truth/*.jsonl`

## 3. 抽出パイプラインの要点
- パターン
  - `data/patterns/patterns.index.json`
  - `data/patterns/patterns.jsonl`
- 依存解析
  - spaCy GiNZA `ja_ginza_bert_large`（失敗時 `ja_ginza`）
- CKY解析（GPU）
  - `tohoku-nlp/bert-base-japanese-v3`（Mask）
  - ローカル依存修正モデル
    - `./models/output_bert_dependency_bunsetsu_ver3.0/depbert_bunsetsu_20260117_072956/final_model`
- 候補抽出
  - CKY表 + パターンASTにより候補spanを生成
  - `X*` を引数、`Y*` を関係候補とし、`X×Y` で candidate を作る
- オントロジー正規化
  - `relation_ja` は ontology の関係ラベルへ正規化（`label_wiki_ja` 優先）

## 4. モード（評価対象の違い）
- `default`
  - 通常の抽出 + オントロジー整合検証 + verified 出力。
- `no_verification`
  - 検証工程を省略し、match 由来の抽出結果をそのまま出力。

## 5. 評価手順（eval_wiki_id）
`src/eval_wiki_id.py` の一括評価で以下を実施。

1) **ID付与（Wikidata link）**
   - pred の triple に QID/PID を付与し、`eval/<tag>/with_ids/*.jsonl` を出力。
   - ネットワーク不可なら `--offline` でキャッシュのみ使用。

2) **全体評価（all）**
   - gold と pred の比較で Precision / Recall / F1（文字列ベース）。
   - `triple_ids` がある場合は **IDベース評価** も算出。

3) **カバレッジ評価（covered）**
   - pred 側で triple が空のレコードを母集団から除外して評価。

4) **no_verification を default の covered ID に制限した評価**
   - `no_verification_with_default_eval_ids` を追加生成。
   - default の covered ID と同一 ID集合で no_verification を評価。

## 6. 実験出力の配置（例）
- 抽出結果
  - `results/ver11.0/extract_pred_arg_pair/extract_target_data/<ont_*>/select_mode/<run_tag>__mode-*/` 
- 評価結果
  - `results/ver11.0/extract_pred_arg_pair/eval/<eval_tag>/...`
- 例（今回参照している集計）
  - `results/ver11.0/extract_pred_arg_pair/eval/ver11_default_codex/no_verification_with_default_eval_ids/summary.tsv`

## 7. summary.tsv の列の意味（no_verification_with_default_eval_ids）
- `default_covered_ids`: default の covered 評価で対象になった ID 数
- `wrote_pred_records`: no_verification 側で評価対象として書き出された件数
- `eval_records`, `gold_triples`: 評価対象レコード数 / gold triple 数
- `tp`, `fp`, `fn`, `precision`, `recall`, `f1`: 文字列ベース評価
- `id_tp`, `id_fp`, `id_fn`, `id_precision`, `id_recall`, `id_f1`: IDベース評価
- `pred_path`, `pred_subset_path`, `pred_with_ids_path`, `pred_with_ids_subset_path`, `gold_path`: 対応する入出力の参照パス

## 8. 再現のための実行イメージ
- 抽出（select mode）: `src/select_mode_main.py` を利用
- 評価: `src/eval_wiki_id.py`

詳細な実行フローは `MAIN_PIPELINE_JA.md` を参照してください。

## 9. モデルのパラメータ設定
### 9.1 依存解析・CKY（ローカルモデル）
- 依存解析（spaCy GiNZA）: `ja_ginza_bert_large`（失敗時 `ja_ginza` へフォールバック）
- CKY解析（BERT）: `tohoku-nlp/bert-base-japanese-v3`
- 依存修正モデル（ローカル）:
  `./models/output_bert_dependency_bunsetsu_ver3.0/depbert_bunsetsu_20260117_072956/final_model`

### 9.2 並列判定 LLM（ParallelJudge）
- モデル名: `llmjp-13b`（vLLM served model）
- 温度: `0.0`
- `max_tokens`: `32`
- キャッシュ: `LRU 4096`

### 9.3 オントロジー整合 LLM（OntologyJudge）
- モデル名: `LLMJP_ONTO_MODEL` → `LLMJP_MODEL` → 既定 `llmjp-13b`
- 温度: `0.0`（既定）
- `max_tokens`: `64`（既定）
- キャッシュ: `LRU 4096`
- 例外的な温度/長さ:
  - prompt_id `21` は `PROMPT21_TEMPERATURE`（既定 `0.15`）と
    `PROMPT21_MAX_TOKENS`（未指定なら既定値）で上書きされる
  - prompt_id `21` の判定が `0` の場合、fallback として prompt_id `22` を使用し、温度 `0.0`
