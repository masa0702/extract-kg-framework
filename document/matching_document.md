# CKYMatcher によるマッチング処理の詳細

このドキュメントでは、`src/matcher.py` に実装された `CKYMatcher` クラスの
動作を中心に、パターン AST と CKY 表を用いたマッチング処理の流れを詳細に
説明します。ここではコードを初めて読む開発者を想定し、各関数の役割や
条件分岐まで踏み込んで解説します。

## 1. 全体の流れ

1. **`match_table`** で CKY 表全体を走査し、各セルの候補に対して
   `_match_candidate` を呼び出す。
2. 候補毎に、パターン AST に対する動的インデックス付与
   (`_assign_dynamic_indices`) を行う。
3. 以下の三つのフィルタを順に適用し、条件を満たした場合に
   `MatchResult` を生成する。
   - `_dependency_label_filter`
   - `_literal_filter`
   - `_pos_and_variable_filter`

## 2. CKYMatcher の主要メソッド

### match_table

- 引数 `cky` は `CkyTable` インスタンス、または二次元リスト形式の CKY 表。
- 表を **スパン長の長い順** に走査し、各セル(`cell`)内の `candidates` を評価します。
- `candidates` は BERT やヒューリスティックにより推定された依存構造を
  持つ辞書オブジェクトです。
- 条件を満たした候補ごとに `MatchResult(i, j, mapping, cand)` をリストへ追加します。

```python
for span in range(n, 0, -1):
    for i in range(1, n - span + 2):
        j = i + span - 1
        cell = mat[i][j]
        if not isinstance(cell, dict) or "candidates" not in cell:
            continue
        for cand in cell["candidates"]:
            mapping = self._match_candidate(cand)
            if mapping is not None:
                results.append(MatchResult(i=i, j=j, variable_mapping=mapping, cell=cand))
```

### _match_candidate

- 各候補(`cand`)に対して以下の処理を順番に実施します。
  途中で失敗した場合は `None` を返し、次の候補に進みます。

1. **動的インデックス付与** : `_assign_dynamic_indices`
   - パターン AST の各ノードに `leaf_idx` を割り当てます。
2. **依存ラベルフィルタ** : `_dependency_label_filter`
   - AST が要求する依存ラベル数を `cand` から満たせるか判定します。
3. **リテラルフィルタ** : `_literal_filter`
   - `LiteralNode` に指定された語が候補のスパンに存在するか確認します。
4. **品詞・変数フィルタ** : `_pos_and_variable_filter`
   - 変数の品詞条件を確認し、変数名と表層形のマッピングを構築します。

### MatchResult

- 成功した場合は `MatchResult` オブジェクトを返します。
- `i`, `j` は CKY 表におけるスパンの開始・終了インデックス (1 始まり)。
- `variable_mapping` は `{"X1": "表層形", ...}` 形式の辞書です。

## 3. 動的インデックス付与 (_assign_dynamic_indices)

この関数はパターン AST を DFS で辿りながら、対象候補の葉ノード
(= 文節や語句) にインデックスを割り当てます。主なポイントは次の通りです。

- **VariableNode**
  - `leaf_idx` が未設定の場合、現在の `leaf_ptr` 以降から条件に合う葉を探して
    割り当てます。
  - `pos_tag` 指定がある場合は、葉の `xpos`/`upos` に該当タグが含まれるか
    をチェックします。
  - 割り当てに成功すると `leaf_ptr` を進め、最後に割り当てた位置を
    `last_var_idx` として保持します。

- **LiteralNode**
  - 直前の変数 (`last_var_idx`) にリテラル文字列が含まれていればそのインデックスを
    流用します。無ければ `leaf_ptr` の位置を確認し、一致しなければ失敗となります。

- **ParallelNode**
  - 選択肢をすべての順列で試行し、いずれか一つでも成功すればマッチ成功とみなします。
  - 接続詞を要求するため、並列要素の境界には `CONNECTIVES_REGEX` で定義された語が
    含まれるか確認します。

- **Modifier*Node**
  - 修飾回数・ブロック修飾・並列修飾に対応しており、子ノードのマッチング結果を
    利用して成功/失敗を決定します。

戻り値は `(ok, leaf_ptr, last_var_idx)` のタプルで、
`ok` が `True` ならすべてのノードにインデックスが割り当てられたことを意味します。

## 4. フィルタ処理の詳細

### _dependency_label_filter

- `PatternNode.get_dependency_label_requirements()` が返すラベル要求を集計し、
  候補が持つ依存ラベルの数で満たせるか確認します。
- すべての要求を満たした場合のみ `True` を返します。

### _literal_filter

- パターン中の `LiteralNode` と、候補から得られた葉の文字列を比較します。
- リテラルに指定された文字列がスパン内に存在しなければ `False` を返します。
- 葉インデックスのずれや結合も考慮し、`_collect_text_recursive` で
  部分文字列を取得しています。

### _pos_and_variable_filter

- `_collect_leaves` で候補木の葉を配列化し、事前に付与された `leaf_idx`
  に基づいて変数→文字列の辞書を作成します。
- `pos_tag` がある場合は葉の品詞列からタグを検索し、見つからないと `None`
  を返して失敗となります。
- 最後に接続詞や補助語の末尾を除去し、整形した文字列を返します。

## 5. 失敗時の分岐

- いずれかのフェーズで条件を満たさない場合、 `_match_candidate` は `None` を返し、
  `match_table` 側ではその候補をスキップして次の候補を評価します。
- 全フェーズを通過すると、変数と文字列の対応を保持した `MatchResult` が得られます。

## 6. 参考: 補助関数

- `_collect_dep_labels` : 依存構造のサブツリーからラベルを抽出します。
- `_collect_leaves` : 二分木の葉ノードをリストとして取得します。
- `_snapshot` / `_restore` : マッチング過程での状態保存と復元に使用します。

以上が `CKYMatcher` を中心としたマッチング処理の概要と詳細です。
このドキュメントを参照することで、各関数がどのように連携し、
どの条件で分岐するかを理解できるはずです。
