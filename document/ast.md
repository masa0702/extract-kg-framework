# パターン AST の構造

ここではパターンマッチングで利用される AST (抽象構文木) の設計について解説します。`src/pattern_nodes.py` に定義された各ノードクラスがどのような役割を持ち、どのように組み合わされてパターンを表現するかをまとめます。

## 1. 基本ノード

### `PatternNode`
すべてのノードの基底クラスで、子ノードのリスト `children` を保持します。`walk()` による前順走査やデバッグ用の `pretty()`、`debug()` メソッドを提供します。

### `SequenceNode`
複数の要素を順番に並べたノードです。パターン全体は最終的に `SequenceNode` として扱われます。

### `VariableNode`
`[X1-名詞]` のように変数名 (`symbol`) と番号 (`index`) 、必要に応じて品詞タグ (`pos_tag`) を持ちます。マッチング時には `leaf_idx` が付与され、対応する文節位置を指します。

### `LiteralNode`
固定語句を表すノードで、`text_tokens` に文字列のリストを保持します。`leaf_idx` はマッチング対象となる葉の位置を示します。

### `ParallelNode`
`[A&B]` のような並列表現をまとめるノードです。複数の選択肢 (`options`) を持ち、実際の照合は `matcher.CKYMatcher` 側で行われます。

## 2. 修飾ノード

修飾回数や並列修飾を表すために以下のノードが用意されています。

- `ModifierSingleNode` : 単一要素に対する修飾
- `ModifierRepeatNode` : `*2` など回数指定の修飾
- `ModifierParallelNode` : 並列修飾を内包
- `ModifierBlockRepeatNode` : ブロック全体への修飾

これらは `mod_chain` ルールから組み立てられ、対象となる `VariableNode` や `ParallelNode` を `head` 属性として保持します。

## 3. 依存関係の指定

一部のノードは `dependency_edges` や `dep_label` 属性を持ち、マッチング時に依存ラベル数を要求します。`PatternNode.get_dependency_label_requirements()` を通じて AST 全体で必要な依存ラベルの数を集計できます。

## 4. AST の生成と利用

`PatternParser` によりパターン文字列が解析されると、上記ノードで構成された AST が得られます。`CKYMatcher` はこの AST を走査し、各 `VariableNode` にインデックスを割り当てた後、品詞や依存関係を考慮して CKY 表とのマッチングを行います。

以上がパターン AST の構造と各ノードの役割です。ソースコードを読みながらこの概要を参照することで、マッチング処理の流れを把握しやすくなるはずです。
