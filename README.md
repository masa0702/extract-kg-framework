# Extract KG Framework

This repository contains a small prototype for parsing Japanese sentences with a
CKY based algorithm and matching them against pattern ASTs.  The code is written
for experimental usage and does not rely on external BERT models during tests.

### Updates

* CKY tables now store both **UPOS** and **XPOS** information for each clause.
  `xpos` is used for pattern matching so that fine grained tags such as
  "サ変名詞" can be handled.  For backward compatibility `pos` also points to
  `xpos`.

### Running the example

The example in `src/main.py` generates a simple CKY table from sample clauses,
performs a heuristic dependency analysis and matches it with a pattern.

```bash
pip install lark graphviz
python src/main.py
```


## 実装概要 (Japanese)

このリポジトリは、日本語文から知識グラフのトリプルを抽出するための実験的なフレームワークです。主な処理は以下のステップで構成されています。

1. **文節・依存解析** (`clause_analysis.DependencyAnalysis`)
   - spaCy GiNZA と `BunsetsuSegmenter` を用いて文を文節に分割します。
   - `analyze_sentences` では GiNZA の依存解析結果と文節情報をまとめて JSON 形式で保存します。

2. **CKY 表の生成** (`cky_table.CkyTable`)
   - 文節列から CKY 表を作成し、各セルには文節の表層形、形態素列、UPOS/XPOS などを格納します。
   - `CkyTable.process_json_to_cky_and_save` を使うと、依存解析結果から CKY 表付きの JSON を作成できます。

3. **依存関係の推定** (`bert_modules.CKYAnalyzer`)
   - CKY 表の各スパンに対し、BERT による判定または簡易ヒューリスティックで係り受けラベルを推定します。
   - 結果は `candidates` として CKY セルに保存され、後続のマッチングで利用されます。

4. **パターンの構築** (`pattern_parser.PatternParser`)
   - `grammar.lark` で定義された DSL をパースして AST を生成します。
   - AST は `pattern_nodes.py` で定義される `PatternNode` 系のクラス ( `VariableNode` , `LiteralNode` , `ParallelNode` など) で表現されます。
   - 例: `[X1-名詞]を[Y1-動詞]` のようなパターンを記述できます。

5. **マッチング処理** (`matcher.CKYMatcher`)
   - CKY 表の各候補に対して AST を照合し、変数 `Xn`, `Yn` 等と文節の対応を求めます。
   - マッチングは以下の段階的なフィルタで行われます。
     1. 依存ラベルの確認 (`_dependency_label_filter`)
     2. リテラル (固定語) の一致確認 (`_literal_filter`)
     3. 品詞条件と変数割当て (`_pos_and_variable_filter`)
   - 一致した場合は `MatchResult` として変数と文字列のマッピングが取得できます。

6. **サンプル実行** (`src/main.py`)
   - CSV から文とパターンを読み込み、上記ステップを順に実施して結果を保存するサンプルです。

### 主要クラス

- **PatternNode** : すべてのパターンノードの基底クラス。`walk()` による AST 走査やデバッグ表示メソッドを備えます。
- **VariableNode** : `[X1-名詞]` のような変数を表し、`symbol` と `index`、必要に応じて `pos_tag` を持ちます。
- **LiteralNode** : リテラル文字列を保持するノード。
- **ParallelNode** : `[A&B]` のような並列構造を表現します。
- **Modifier*Node** : 修飾回数や並列修飾を表すノード群 (`ModifierRepeatNode`, `ModifierParallelNode`, `ModifierBlockRepeatNode` など)。
- **CKYMatcher** : 上記ノードで構成された AST を用い、CKY 表から条件に合致するスパンを探索します。
- **CKYAnalyzer**, **CkyTable** : 依存判定付き CKY 表を生成するユーティリティ。

### 実行例

```bash
pip install lark graphviz
python src/main.py
```

`src/main.py` は簡単なデータセットに対して上記の解析とマッチングを実行し、一致した変数ペアを CSV に出力します。

## 付属ドキュメント

`document` ディレクトリに各処理の詳細をまとめた解説を用意しています。

- `matching_document.md` : CKYMatcher のマッチング処理
- `cky_table.md` : CKY 表生成と操作
- `parser.md` : パターンパーサーの仕組み
- `clause_segmentation.md` : 文節分割処理の概要
- `ast.md` : パターン AST の構造

必要に応じて参照してください。
