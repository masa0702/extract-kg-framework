# パターン記法と文法の詳細（処理連携付き）

本ドキュメントでは、`src/pattern/grammar.lark` で定義されているパターン文法の構文と派生する AST 構造、および `PatternParser`・`CKYMatcher` がどのように処理しているかをまとめます。全ての記述は日本語です。

## 1. トークンと字句レベルの規則

- 無視対象: インライン空白、改行、行コメント (`// ...`) はすべて無視されます。
- 特殊記号トークン: `[` `]` `(` `)` `&` `*` `#` `{` `}` `,` `:` の各1文字。
- 変数トークン: `VAR` は `X`/`Y`/`M` の単一大文字のみ。後続の整数と合わせて変数名を構成します（`G` はギャップ専用）。
- ギャップトークン: `G` はギャップ専用トークン（`VAR` には含まれません）。
- 品詞タグトークン: `TAG` は `]` `[` `(` `)` と空白を除く任意の連続文字列。`-TAG` の形で変数に付与されます（`&` などの記号は個別トークンとして先に切られるためタグ内に入りません）。
- リテラル: `[]&*#()` を除く任意の文字列。パターン内の自由テキストとして扱われます（`{ } , :` は個別トークンとして切られるためリテラルには含まれません）。【F:src/pattern/grammar.lark†L1-L73】

## 2. 構文規則（生成できるパターン）

- パターン全体は1つ以上の `element` の連接です。`element` は以下のいずれかになります。
  - `parallel_chain`: `[...]&[...]` 形式のトップレベル並列（外側の `[]` は不要）。
  - `bracketed`: `[expr]` 形式。内部の `expr` が実際の式になります。
  - `literal`: トークン化された自由文字列。
- `expr` のバリエーション（`[]` 内で許可されるのはこの3種のみ）:
  - `mod_chain`: 修飾子の連鎖 + 変数本体。例 `[*2X1-名詞]`。
  - `gap_expr`: ギャップ指定。例 `[G{1,3}]` や `[G{0,2}:名詞]`。
  - `block_repeat`: `*n(...)` または `#n(...)` でブロック全体に回数修飾を掛ける独立構文（※`[]` 内で使用）。
- 修飾子 (`modifier`) の種類:
  - `mod_repeat`: `*n` / `#n` の回数指定（連体/連用）。
  - `mod_parallel_count`: `*n( ... & ... )` / `#n( ... & ... )` のように回数と並列塊を組み合わせた修飾。回数なしの並列修飾は不可。
- 並列ブロック (`parallel_group_inner`): `( [..] & [..] & ... )`。要素は必ず `bracketed`（角括弧付き）です。
- 変数 (`var_atom`): `X`/`Y`/`M` + 整数 + 任意の品詞タグで構成されます。例: `[Y2-サ変+する]`。【F:src/pattern/grammar.lark†L24-L73】

## 3. AST 生成の流れ (`PatternParser` と Transformer)

- `PatternParser` は `grammar.lark` を LALR で読み込み、`PatternTransformer` により `SequenceNode` を根とする AST に変換します。【F:src/pattern/pattern_parser.py†L23-L45】
- `element` ごとに `ParallelNode`・`VariableNode`・`LiteralNode` などの `PatternNode` 派生クラスへマッピングされます。【F:src/pattern/pattern_parser.py†L48-L128】
- 修飾子連鎖 (`mod_chain`) の適用順序は「右側（変数に近い）から左側へ」。各修飾子は以下のノードに展開されます。【F:src/pattern/pattern_parser.py†L93-L162】
  - `mod_repeat` → `(kind, count)` タプルから `ModifierRepeatNode(kind, count, head)` を生成。
  - `mod_parallel_count` → `("PAR", kind, cnt, parallel_block)` 形式を `ModifierParallelNode(kind, parallel_block, head, count=cnt)` に変換。
- `block_repeat` は `ModifierBlockRepeatNode(kind, count, block)` を直接返し、後続の head を持たないブロック修飾として扱われます（`[]` 内で単独利用）。【F:src/pattern/pattern_parser.py†L79-L118】
- ギャップ指定 `gap_expr` は `GapNode(min,max,tag)` に変換されます。【F:src/pattern/pattern_parser.py†L64-L83】

## 4. 生成されるノードと意味付け

- `VariableNode(symbol, index, pos_tag)`: 変数本体。`leaf_idx` はマッチング時に付与されます。
- `LiteralNode(text_tokens)`: 文字列リテラル。`get_literal_nodes()` で葉インデックスとともに収集され、リテラルフィルタで利用されます。
- `GapNode(min_skip, max_skip, tag)`: 文節スキップ幅の指定。`tag` は任意のラベルで、現状はマッチ制約用途の想定です。
- `ParallelNode(options)`: 並列の各選択肢を保持。マッチ時には「順列」ではなく **パターンに書かれた順** で評価され、並列要素（最後以外）には接続詞検知が要求されます。
- `ModifierRepeatNode(kind, count, head)`: `*n`/`#n` の回数指定。`seq_id` の加算に `count` が反映されます（※マッチング時の反復展開は現状行われません。詳細は後述）。
- `ModifierParallelNode(kind, parallel_block, head, count)`: 並列塊を修飾として head に掛ける構造。`count` は `*n/#n` の n を保持します。
- `ModifierBlockRepeatNode(kind, count, block, head=None)`: 並列含む任意ブロックを回数付き修飾としてまとめるノード。head を持たない形が基本です。
- `DependencyEdgeNode`: 変数間の必須依存エッジを明示する専用ノード。**文法からは生成されません**が、手動で AST に追加する用途があります。`get_required_dependency_edges()` で取得可能です。【F:src/pattern/pattern_nodes.py†L12-L214】【F:src/pattern/pattern_nodes.py†L216-L244】

## 5. マッチング時の扱い（`CKYMatcher`）

1. **シーケンス ID 付与**: DFS 順で `seq_id` を設定。`VariableNode` と `ModifierSingleNode` と `ModifierParallelNode` は 1 ずつ、`ModifierRepeatNode` は `count` 分だけ加算されます。`ModifierBlockRepeatNode` と `LiteralNode` は対象外です。【F:src/modules_core/matcher.py†L128-L181】
2. **粗いリテラル事前チェック**: 割当列挙の前に、候補 `cand.text`（無い場合は再帰収集）に literal が含まれるかを検査します。【F:src/modules_core/matcher.py†L73-L125】
3. **依存ラベルフィルタ**: AST の `dep_label` 要求数と候補の依存ラベル数を比較し、必要数未満なら即座に不一致。【F:src/modules_core/matcher.py†L126-L159】
4. **動的インデックス割当**: CKY 木の葉を左から順に変数へ割り当てます。主な挙動は以下です。【F:src/modules_core/matcher.py†L182-L396】
   - `ParallelNode` は **順列探索ではなくパターン記述順** で評価されます。並列要素のうち **最後以外** は接続詞正規表現にマッチする葉のみが許容されます。
   - `VariableNode` は pos タグ不一致ならスキップされます。直前が `LiteralNode` の場合は leaf_ptr を固定し、次の literal を先読みして含まれない葉もスキップします。
   - `LiteralNode` は「直前に割り当てた葉」または「現在の leaf_ptr」への部分一致で成立します。
   - `GapNode` は `min_skip..max_skip` の範囲で葉をスキップします。
   - `ModifierRepeatNode` / `ModifierParallelNode` / `ModifierBlockRepeatNode` は **子ノードを1回だけ辿る** 実装になっています（`ModifierRepeatNode` の可変長展開用コードは後段にありますが、現状分岐到達しません）。
5. **リテラル精密チェック**: 割当後に `LiteralNode` を再度検証。`leaf_idx` が無い場合は候補全体テキストに部分一致するか、`leaf_idx` がある場合は該当葉（複数指定も対応）を再帰連結して照合します。【F:src/modules_core/matcher.py†L94-L172】
6. **品詞・変数フィルタ**: `get_variable_constraints()` で得た (記号, 葉位置, 品詞タグ) に従い CKY 葉をチェック。タグ不一致や葉範囲外なら不一致とし、最後に表層語の末尾接続詞や補助動詞を正規化してマッピングを返します。**変数の番号（X1/X2 など）は AST 内の出現順で再採番される**ため、パターン上の index 値はマッピング順序に影響しません。【F:src/modules_core/matcher.py†L173-L248】

## 6. 代表的なパターン例と例外事項

- 並列 + 修飾の複合: `[*1([M1-形容]&[M2-形容])X1-名詞]を[Y1-動詞]` → 並列ブロックを修飾として名詞に掛け、後続の動詞を要求します。
- ギャップ指定: `[G{1,3}]` は 1〜3 文節のスキップを許容、`[G{0,2}:名詞]` はタグ付きギャップ指定です（現状タグはマッチ制約用途の想定）。
- ブロック独立修飾: `[*2([X1]&[X2])]` は head 無しの `ModifierBlockRepeatNode` として AST に残りますが、現状のマッチング実装では子ブロックを1回だけ辿ります。
- 接続詞の例外: 並列の途中要素で `CONNECTIVES_REGEX` にマッチしない葉はスキップされ、並列全体が不一致扱いになります。接続詞を含む形態のみ並列要素として認められます。
- 品詞タグ未指定: `pos_tag` が無い変数は表層語だけで一致判定され、品詞フィルタを素通りします。

## 7. 開発上の注意点

- 文法拡張時は `PatternTransformer` 側で新ノードへのマッピングを追加しないと AST が欠落する可能性があります。
- 並列の探索は「順列」ではなく記述順なので、柔軟性は下がる一方で探索量は抑えられます。必要に応じて `CONNECTIVES_REGEX` を調整します。
- `ModifierRepeatNode` の可変長展開は現状の分岐構造では到達しないため、回数指定は依存ラベル要求にのみ反映されます（マッチング側の意図と実装がズレている可能性に注意）。
- `LiteralNode` に `leaf_idx` が設定されるケースでは CKY 木の葉数を超える指定を行うと即時失敗になるため、パターン生成時に要注意です。
