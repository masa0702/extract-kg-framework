# パターンパーサーの概要

ここでは `src/pattern_parser.py` で提供されている `PatternParser` クラスの仕組みと、パターン DSL の解析手順を解説します。パターン文字列から AST を構築し、マッチング処理に渡すまでの流れを整理します。

## 1. Lark による文法定義

パターン記法は `grammar.lark` にて Lark 形式で定義されています。句読点や修飾表現、変数などをトークンとして定義し、BNF 風のルールで AST の構造を表します。

- `[]` で囲まれた部分は変数や修飾を表す
- `&` を用いた並列表現
- `*2` のような回数指定

`PatternParser` はこの文法ファイルを読み込み、`Lark` パーサを生成します。

## 2. PatternParser クラス

`PatternParser` は二つの主要メソッドを持ちます。

- `__init__`
  - `grammar.lark` を読み込み `Lark` オブジェクトを作成
  - 併せて `PatternTransformer` を用意
- `parse(text)`
  - パターン文字列 `text` を受け取り、Lark でパース
  - 生成された `Tree` を `PatternTransformer` が AST ノードに変換し、`SequenceNode` を返す

## 3. PatternTransformer

`PatternTransformer` は Lark の `Transformer` を継承し、各種ノード (`VariableNode`, `ParallelNode` など) に変換する役割を持ちます。主要な処理は次の通りです。

1. `pattern` ルールで `SequenceNode` を生成し、要素をリスト化
2. `parallel_group` では `ParallelNode` を構築
3. `mod_chain` では修飾子の種類に応じて `Modifier*Node` を組み立て

結果として得られる AST は `pattern_nodes.py` で定義されるクラス群で表現されます。

## 4. AST の活用

生成された AST は `matcher.CKYMatcher` へ渡され、文中の CKY 表と照合する際のテンプレートとして機能します。AST 上のノードには動的にインデックスが付与され、依存ラベルや品詞制約をもとにマッチングが行われます。

以上がパターンパーサーの基本的な流れです。`grammar.lark` の変更により新しい表現を追加することも可能で、柔軟にパターンマッチングを記述できます。
