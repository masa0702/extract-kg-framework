# MATCHING_ALGORITHM_DOCS

本ドキュメントは `src/modules_core/matcher.py` の実装から読み取れるマッチング仕様を整理したものです。

## 1. 入力・出力
- 入力
  - パターンAST（`pattern_parser` が生成する `PatternNode` 系）
  - CKY 表（1-based ヘッダ付き 2D 配列）
    - セルが `{"candidates": [...]}` を持つ場合のみ評価対象
- 出力
  - `MatchResult(i, j, variable_mapping, cell)` のリスト
  - `variable_mapping` は `X1`, `Y1` などの変数→文字列の写像

## 1.5 用語・変数一覧（実装上の意味）
- `leaf`  
  - 候補木（`cand`）の葉ノード。文節（または文節相当）を表す最小単位。
- `leaves`  
  - `cand` から再帰的に収集した葉の配列。`leaves[0]` が文中で最も左の文節に相当。
- `leaf_ptr`  
  - 「次に評価する leaf の開始位置」を示すインデックス（前進のみ）。
- `last_var_idx`  
  - 直前に割り当てた変数の leaf インデックス。Literal の一致判定に利用。
- `parent_children` / `idx_in_parent`  
  - 現在評価中ノードの親 Sequence/Parallel の子配列と、その中での位置。
- `force_leaf_ptr`  
  - 現在位置の leaf のみ評価する強制フラグ。スキップ禁止の制約に使われる。
- `cand`  
  - CKY セル内の候補構造。`left/right` で二分木を構成し、葉に `candidate/text/xpos/upos` を持つ。
- `MatchResult.i / MatchResult.j`  
  - CKY 表上のセル位置（1-based span）。
- `variable_mapping`  
  - 変数名（`X1`, `Y1` など）→抽出された文字列。

## 2. CKY 表の前提
- `cky[0][i]` と `cky[i][0]` はヘッダ文字列
- `cky[i][j]` が `dict` かつ `"candidates"` を持つセルのみマッチを試行
- 候補は二分木（`left` / `right`）で、葉は以下のキーを想定
  - `candidate` / `text` / `xpos` / `upos`

## 3. セル走査順
- span 長が **大きい順**
- span 内は **左から右**（`i` 昇順）

## 4. マッチングの全体フロー
1. 候補セルごとに **全解列挙**を実施
2. 動的割当（`_iter_assignments`）で leaf への割当を探索
3. 依存ラベルフィルタ（`_dependency_label_filter`）
4. リテラルフィルタ（`_literal_filter`）
5. 品詞・変数フィルタ（`_pos_and_variable_filter`）
6. 重複写像を除外し、結果に追加

## 4.5 主要データ構造
- **Pattern AST**  
  - `SequenceNode / ParallelNode / VariableNode / LiteralNode / GapNode / Modifier*` で構成。
- **Candidate Tree (`cand`)**  
  - CKYセル内の候補木。二分木として `left/right` を持ち、葉が文節に対応。
- **Leaves 配列**  
  - `cand` の葉を左から順に並べた配列。`leaf_ptr` はこの配列のインデックスを指す。

## 5. 動的割当の仕様（全解列挙）
### 5.1 基本
- `leaf_ptr` を **前方に進めながら**探索
- 逆順や過去の leaf への対応はしない
- 変数・リテラル・修飾・並列・ギャップを順に処理

### 5.2 SequenceNode
- 子ノードを **パターンの順序通り**に評価

### 5.3 VariableNode
- `leaf_ptr` 以降から候補 leaf を探索
- `pos_tag` 指定がある場合、`xpos/pos/upos` にタグが含まれる必要がある
- 直前が `LiteralNode` の場合、**現在の leaf のみ評価**（スキップ禁止）
- 並列ブロック内の 2 要素目以降は **現在の leaf のみ評価**（連続性の強制）

### 5.4 LiteralNode
- 直前の変数 leaf を優先して含有確認
- 合わなければ現在 `leaf_ptr` の leaf を確認

### 5.5 ParallelNode
- **順序固定**（パターンに記載された順）
- 2 要素目以降は **直前要素の次の leaf のみ**評価
- 並列ブロックの開始位置は **候補内で自由**

### 5.6 ModifierSingleNode / ModifierRepeatNode / ModifierBlockRepeatNode
- 子ノードを順に評価
- `ModifierRepeatNode(kind="*")` は最大 `min(count, 5)` 回まで反復展開

### 5.7 GapNode（G{m,n}）
- 文節（leaf）単位のスキップ
- `G{m,n}` は **現在の leaf_ptr から m〜n 個スキップ**
- Gap 自体は割当を持たず、次の評価位置を進める

## 6. フィルタ仕様
### 6.1 依存ラベル
- `PatternNode.get_dependency_label_requirements()` の要求数を満たす必要がある

### 6.2 リテラル
- Literal が指定 leaf（またはセル全体 `cand["text"]`）に含まれる必要がある

### 6.3 品詞・変数
- `get_variable_constraints()` が返す `(symbol, leaf_idx, pos_tag)` を満たす
- 結果は `X1`, `Y1` などの連番で格納

## 7. 文字列後処理
- 変数の末尾にある接続詞・補助動詞などを除去
  - 例: 「であり」「および」「、」など

## 8. 仕様上の特徴
- **文節順は常に前進**
- **並列ブロックは連続要素のみ**
- **Literal直後の変数はスキップ禁止**
- **開始位置は自由**（文頭からの固定一致ではない）

## 9. 具体例
### 9.1 並列 + 直後動詞
文: `りんごとみかんを購入する太郎と花子を監視する。`
パターン: `[X0]&[X1]を[Y1]する`

期待される結果:
- `{'X1': 'りんご', 'X2': 'みかんを', 'Y1': '購入する'}`
- `{'X1': '太郎', 'X2': '花子を', 'Y1': '監視する。'}`

### 9.2 ギャップ
文: `会社の仕事を太郎と花子が担当する。`
パターン: `[X0]を[G{4,5}][Y1]する`

期待:
- `X0` に `仕事を` を割当後、4〜5 文節スキップして `Y1` を評価
- 文節数が不足する場合はマッチしない

## 10. 既知の制約
- 候補木の粒度（文節の切り方）に強く依存
- CKY 表に複数候補がない場合、同文内の複数マッチが得られない
- 依存情報がない候補では構造的な優先順位付けができない
