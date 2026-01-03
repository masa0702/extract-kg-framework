# パターン記法と文法の詳細（処理連携付き）

本ドキュメントでは、`src/grammar.lark` で定義されているパターン文法の構文と派生する AST 構造、および `PatternParser`・`CKYMatcher` がどのように処理しているかをまとめます。全ての記述は日本語です。

## 1. トークンと字句レベルの規則

- 無視対象: インライン空白、改行、行コメント (`// ...`) はすべて無視されます。
- 特殊記号トークン: `[` `]` `(` `)` `&` `*` `#` の各1文字。
- 変数トークン: `VAR` は単一大文字 (例: `X`)、後続の整数と合わせて変数名を構成します。
- 品詞タグトークン: `TAG` は `[]&*#()` と空白を除く任意の連続文字列として扱われ、`-TAG` の形で変数に付与されます。
- リテラル: 上記の特殊記号を含まない任意の文字列。パターン内の自由テキストとして扱われます。【F:src/grammar.lark†L5-L73】

## 2. 構文規則（生成できるパターン）

- パターン全体は1つ以上の `element` の連接です。`element` は以下のいずれかになります。
  - `parallel_group`: `[...]&[...]` 形式のトップレベル並列。中身は必ずブラケットで囲われます。
  - `bracketed`: `[expr]` 形式。内部の `expr` が実際の式になります。
  - `literal`: トークン化された自由文字列。
- `expr` の主なバリエーション:
  - `mod_chain`: 修飾子の連鎖 + 変数本体。例 `*2#1X1-名詞`。
  - `parallel_expr`: `atom` を `&` で連結した並列。例 `[X1]&[X2-動詞]`。
  - `block_repeat`: `*n(...)` または `#n(...)` でブロック全体に回数修飾を掛ける独立構文。
- 修飾子 (`modifier`) の種類:
  - `mod_repeat`: `*n` / `#n` の回数指定（連体/連用）。
  - `mod_parallel`: `*( ... & ... )` のように並列塊を修飾として付与。
  - `mod_parallel_count`: `*n( ... )` のように回数と並列塊を組み合わせた修飾。
- 並列式 (`parallel_expr`): `atom ( & atom )+`。`atom` には変数 (`VAR INT` に任意の `-TAG`) または `bracketed` 式が入ります。
- 変数 (`var_atom`): 大文字1文字 + 整数 + 任意の品詞タグで構成されます。例: `Y2-サ変+する`。【F:src/grammar.lark†L24-L73】

## 3. AST 生成の流れ (`PatternParser` と Transformer)

- `PatternParser` は `grammar.lark` を LALR で読み込み、`PatternTransformer` により `SequenceNode` を根とする AST に変換します。【F:src/pattern_parser.py†L23-L39】
- `element` ごとに `ParallelNode`・`VariableNode`・`LiteralNode` などの `PatternNode` 派生クラスへマッピングされます。【F:src/pattern_parser.py†L48-L99】
- 修飾子連鎖 (`mod_chain`) の適用順序は「右側（変数に近い）から左側へ」。各修飾子は以下のノードに展開されます。【F:src/pattern_parser.py†L117-L156】
  - `mod_repeat` → `(kind, count)` タプルから `ModifierRepeatNode(kind, count, head)` を生成。
  - `mod_parallel` → `(kind, ParallelNode)` タプルから `ModifierParallelNode(kind, parallel, head)`。
  - `mod_parallel_count` → `("BLOCK", kind, cnt, content)` タグ付きタプルとして扱い、`ModifierBlockRepeatNode` を生成。
  - 上記以外の `(kind, VariableNode)` は単一要素修飾として `ModifierSingleNode(kind, child, head)` を組み立て。
- `block_repeat`（スタンドアロン）は `ModifierBlockRepeatNode` を直接返し、後続の head を持たないブロック修飾として扱われます。【F:src/pattern_parser.py†L100-L156】

## 4. 生成されるノードと意味付け

- `VariableNode(symbol, index, pos_tag)`: 変数本体。`leaf_idx` はマッチング時に付与されます。
- `LiteralNode(text_tokens)`: 文字列リテラル。`get_literal_nodes()` で葉インデックスとともに収集され、リテラルフィルタで利用されます。
- `ParallelNode(options)`: 並列の各選択肢を保持。マッチ時には順列で全通り評価され、並列要素間には接続詞検知が要求されます。
- `ModifierRepeatNode(kind, count, head)`: `*n`/`#n` で head を最大 `count` 回繰り返すノード。`seq_id` の加算に `count` が反映されます。
- `ModifierSingleNode(kind, child, head)`: `*X1` のような単一要素修飾。`child` 側を修飾語、`head` 側を被修飾語として依存ラベルを `kind` から付与します。
- `ModifierParallelNode(kind, parallel_block, head)`: 並列塊を修飾として head に掛ける構造。距離は常に1として扱われます。
- `ModifierBlockRepeatNode(kind, count, block, head=None)`: 並列含む任意ブロックを回数付き修飾としてまとめるノード。head を持つ場合と持たない場合の両方を許容します。
- `DependencyEdgeNode`: 変数間の必須依存エッジを明示する専用ノード。`get_required_dependency_edges()` で CKY マッチ時に利用されます。【F:src/pattern_nodes.py†L12-L202】【F:src/pattern_nodes.py†L205-L244】

## 5. マッチング時の扱い（`CKYMatcher`）

1. **シーケンス ID 付与**: `ModifierRepeatNode` など回数指定を考慮しつつ DFS 順で `seq_id` を設定（`ModifierBlockRepeatNode` は対象外）。【F:src/matcher.py†L119-L145】【F:src/matcher.py†L115-L200】
2. **動的インデックス割当**: CKY 木の葉を左から順に変数へ割り当て。`ParallelNode` は順列全探索で接続詞正規表現を満たす並びのみを受理し、失敗時はスナップショットからロールバックします。`Modifier*` 系は子ノードを必須回数だけ展開し、回数指定 `*` では最大5回までの実行を試行します。`leaf_idx` が無い場合は前段の値を維持する処理があり、接続詞を含まない候補はスキップされます。【F:src/matcher.py†L115-L200】【F:src/matcher.py†L201-L288】
3. **依存ラベルフィルタ**: `PatternNode` に保持された `dep_label` 需要を `Counter` で比較し、必要数未満なら即座に不一致。【F:src/matcher.py†L147-L154】
4. **リテラルフィルタ**: AST 内の `LiteralNode` を順に検証。`leaf_idx` が無ければ span 全体のテキストに部分一致するかを確認し、インデックスがあれば指定葉（複数可）の再帰連結と照合します。【F:src/matcher.py†L155-L181】
5. **品詞・変数フィルタ**: `get_variable_constraints()` で得た (記号, 葉位置, 品詞タグ) に従い CKY 葉をチェック。タグ不一致や葉範囲外なら不一致とし、最後に表層語の末尾接続詞や補助動詞を正規化してマッピングを返します。【F:src/matcher.py†L182-L200】【F:src/matcher.py†L201-L248】

## 6. 代表的なパターン例と例外事項

- 並列 + 修飾の複合: `[*1([M1-形容]&[M2-形容])]を[Y1-動詞]` → ブロック修飾で並列形容詞を1回だけ許容しつつ動詞を後続に要求。
- 回数上限付き繰り返し: `[X1-名詞]*3` は最大3回までの繰り返しが `_assign_dynamic_indices` で展開される（内部上限は5回で安全側に制約）。
- ブロック独立修飾: `*2([X1]&[X2])` は head 無しの `ModifierBlockRepeatNode` として AST に残り、葉割当では子ブロック全体が `count` 回分消費されます。
- 接続詞の例外: 並列の途中要素で `CONNECTIVES_REGEX` にマッチしない葉はスキップされ、並列全体が不一致扱いになります。接続詞を含む形態のみ並列要素として認められます。
- 品詞タグ未指定: `pos_tag` が無い変数は表層語だけで一致判定され、品詞フィルタを素通りします。

## 7. 開発上の注意点

- 文法拡張時は `PatternTransformer` 側で新ノードへのマッピングを追加しないと AST が欠落する可能性があります。
- 並列や回数指定は `_assign_dynamic_indices` に実装上限やスナップショット処理があるため、爆発的な組み合わせになると探索性能に影響します。上限値や接続詞判定の正規表現を見直すことで制御できます。
- `LiteralNode` に `leaf_idx` が設定されるケースでは CKY 木の葉数を超える指定を行うと即時失敗になるため、パターン生成時に要注意です。
