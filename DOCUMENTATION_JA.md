# リポジトリ解説ドキュメント

## 1. 概要
- 本リポジトリは、日本語文をCKYベースの句構造に分解し、パターンASTとマッチングして述語・項の対応を抽出する試験的フレームワークです【F:README.md†L1-L22】。
- `src/main.py` では、ASTのロード、依存解析、フィルタリングを並列実行しつつCKYマッチングを回すパイプライン全体をまとめています【F:src/main.py†L1-L120】。

## 2. ディレクトリ構成
- `src/`: コア実装。AST定義・パーサ、CKY表生成、依存解析、マッチャー、BERTユーティリティ、可視化などが含まれます。
- `data/`: 依存解析のサンプルJSONや固定句設定などの補助データ。
- `ontology/`: オントロジー(JSON形式)のサンプル。
- `tests/`: フィルタロジックを確認するスモールテスト群。
- `README.md`: 実行例とアップデート概要【F:README.md†L7-L22】。

## 3. コアロジックとモジュールのつながり
### 3-1. パターン定義とAST化
- `src/pattern_parser.py` が `grammar.lark` を読み込み、`PatternTransformer` によりパターン文字列を `SequenceNode`/`VariableNode`/`ParallelNode` などのASTへ変換します【F:src/pattern_parser.py†L3-L69】。
- ASTノードの実体は `src/pattern_nodes.py` で定義され、依存ラベル要求やリテラル抽出などマッチング時に使うAPIを提供します【F:src/pattern_nodes.py†L1-L74】。
- 可視化用に `ast_visualizer.py` を読み込む仕組みもあり、ASTを外部出力できます【F:src/pattern_parser.py†L15-L21】。

### 3-2. CKY表生成と依存解析
- `src/cky_table.py` は文節リストからCKY表を初期化し、セル文字列取得や並列接続詞のカウントといったユーティリティをまとめています【F:src/cky_table.py†L6-L110】。
- 依存解析は `src/bert_modules.py` の `CKYAnalyzer`（BERTモデル利用想定）や `src/clause_analysis.py` の `DependencyAnalysis` と連携し、`main.py` から呼び出されます【F:src/main.py†L43-L49】。

### 3-3. マッチングとフィルタ
- `src/matcher.py` の `CKYMatcher` がパターンASTとCKYセル候補を照合し、変数マッピング結果を返します【F:src/matcher.py†L1-L47】。
- 並列キーやリテラル順序の簡易フィルタは `tests` 配下のスクリプトで実証されており、`CkyTable` のユーティリティを組み合わせています【F:tests/test_literal_filter.py†L12-L38】【F:tests/test_parallel_filter.py†L13-L31】。
- 文節数チェックなどの素朴な前処理ロジックもテスト化され、ASTの変数数とセル幅の整合を確認しています【F:tests/test_bunsetsu_filter.py†L12-L33】。

### 3-4. 実行パイプライン
- `src/main.py` はASTピクルのロード→CKY解析（GPU/CPU並列）→フィルタ→マッチング→CSV/可視化ログ出力までを段階的に制御します【F:src/main.py†L1-L120】。
- 依存解析結果やCKY表のキャッシュ経路、ログファイル名を定数としてまとめており、長時間バッチ処理を想定した構成です【F:src/main.py†L51-L90】。

## 4. データ・設定ファイル
- `data/dependency_analysis.json` などはCKY表生成処理の入出力例として利用できます【F:src/cky_table.py†L136-L170】。
- `src/fixed_chunks.yml` や `src/parallel_connectives.yml` は句切りの固定リストや並列接続詞設定として読み込まれる前提の補助設定です。
- `ontology/1_movie_ontology_trans_ja.json` は外部オントロジーを参照する試験データで、`text2pattern_with_ontology.py` などの補助スクリプトで利用可能です。

## 5. テスト
- 3種類の短いスクリプトで、リテラル順序フィルタ、並列キー数フィルタ、文節数フィルタの挙動を確認しています【F:tests/test_literal_filter.py†L12-L38】【F:tests/test_parallel_filter.py†L13-L31】【F:tests/test_bunsetsu_filter.py†L12-L33】。
- いずれも `PatternParser` と `CkyTable` を組み合わせて、マッチング前の前処理ロジックを手軽に検証できます。

## 6. 補助・拡張モジュール
- `src/mask_module.py` はBERTのマスク予測で係り受けを判定するクラスを提供し、`bert_modules.py` から読み込まれます【F:src/mask_module.py†L1-L10】【F:src/bert_modules.py†L1-L10】。
- `src/visual_table.py` や `src/ast_visualizer.py` はCKY表やASTの可視化に使えるユーティリティです。
- `src/text2pattern_transform_by_api.py`、`src/text2pattern_with_ontology.py`、`src/eval_text2pattern.py` などはテキストからパターンを生成・評価する実験用スクリプト群です。

## 7. 未使用/整理候補
- `src/extract_pred_arg_pair.py.save` は同名スクリプトのバックアップで、現行パイプライン（`src/main.py`）と役割が重複しています。自動テストや他ファイルからの参照は無く、整理候補です【F:src/extract_pred_arg_pair.py.save†L1-L9】。
- `data/target_datas` の参照を前提とした定数が `src/main.py` にありますが、リポジトリ内に該当データが無いため、実行時に別途配置が必要です【F:src/main.py†L54-L65】。
- `src/look_ast_pikl.py` などデバッグ向けスクリプトは現行テストからは呼ばれておらず、用途を確認した上で存続可否を判断すると良いでしょう。
