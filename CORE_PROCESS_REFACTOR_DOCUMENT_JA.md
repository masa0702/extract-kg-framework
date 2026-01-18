# CORE_PROCESS_REFACTOR_DOCUMENT_JA

本ドキュメントは `src/modules_core` と `src/main.py` の現状実装を踏まえ、改善が望ましい点と不要・整理候補のスクリプトを整理したものです。

## 1. 改善した方が良い部分（全体）

- **設定値・入出力パスのハードコード**
  - `src/main.py` で `AST_PICKLE` / `INPUT_SENT_CSV` / 出力ディレクトリが固定化されているため、環境依存が強い。CLI引数や設定ファイル化を検討。
- **モデル読み込みの重複・初期化コスト**
  - `src/modules_core/clause_analysis.py` と `src/modules_core/bunsetu.py` がそれぞれ `spacy.load()` を実行しており、同一プロセス内で二重にモデルを抱える構成。
  - 遅延ロードや共有インスタンス化（依存性注入）でメモリ/起動時間を削減できる。
- **副作用のあるトップレベル実行**
  - `modules_core` 内の複数ファイルで import 時にログ設定やテスト実行が行われる。ライブラリ用途での利用時に副作用になるため `if __name__ == "__main__":` ガードへ移動。
- **デバッグ出力が本処理に混在**
  - 例: `clause_analysis.py` の `print(target_xpos)`、`cky_table.py` の表表示など。ログレベルを設けるか、デフォルトで無効化。
- **データ更新時のキャッシュ不整合**
  - `main.py` で `dep_json_path` が更新されても `cky_json_path` が既存なら再生成されない。差分更新（未解析文のみ追加）か再生成条件の見直しが必要。

## 2. 改善した方が良い部分（ファイル別）

### src/modules_core/clause_analysis.py
- **重複 import / 未使用 import**
  - `json` が2回 import、`csv` / `re` が未使用。整理対象。
- **`clause_search` の未使用・デバッグ出力**
  - 参照箇所がなく、`print(target_xpos)` が混在。必要ならユニットとして切り出し、不要なら削除。
- **文字位置の算出仕様の明確化**
  - `char_pointer` を 1 始まりで計算しているが、空白や特殊文字を含む場合の整合性が保証されない可能性。規約を明記し、関数の入出力仕様を統一。
- **`nlp` / `seg` のグローバル実体化**
  - import 時にモデルロードが発生する。`DependencyAnalysis` 内に遅延ロードする構成が望ましい。

### src/modules_core/bunsetu.py
- **`segment_sentences` の二重解析**
  - `nlp.pipe()` を使っているが、内部で `cls.segment(sent)` が再度 `nlp(sent)` を呼ぶため二重解析になる。`doc` を使って処理を流用できる設計に改善。
- **トップレベルでのモデルロード・ログ設定**
  - `logging.basicConfig()` と `nlp = load_spacy_model(...)` が import 時に実行されるため、起動が重く、副作用も大きい。必要時に初期化する方式へ。
- **`__main__` テストコードの分離**
  - テスト文が直書き。サンプルとして残すなら `examples/` に移動。

### src/modules_core/cky_table.py
- **表示関数の重複と副作用**
  - `display_multiline_cky_table` / `cky_table_to_tsv` は `visual_table.py` と重複。どちらかに統合。
- **`process_json_to_cky_and_save` の標準出力**
  - 全文で簡易表を出力しており、大量データでボトルネックになる可能性。フラグ制御に。
- **`cky_table_to_tsv` の戻り値**
  - `return print(tsv_table)` となっており文字列が返らない。呼び出し側で扱いやすい形に変更。

### src/modules_core/visual_table.py
- **import 時にサンプルコードが実行される**
  - ファイル末尾で `cky_table_to_tsv` が実行され、標準出力に副作用が出る。`__main__` ガードへ。
- **`cky_table.py` と機能が重複**
  - 重複を解消し、ユーティリティは1箇所に集約。

### src/modules_core/utils.py
- **`most_common` の重複定義**
  - 同名関数が二度定義されており、前者が上書きされる。仕様統一が必要。
- **`load_json_from_file` の `lru_cache`**
  - ファイル更新を検知できず、古い内容を返す可能性がある。キャッシュ戦略を再検討。

### src/modules_core/matcher.py
- **`_assign_dynamic_indices` の条件分岐構造**
  - `if counters is None: ... elif isinstance(node, ParallelNode):` となっており、初回呼び出し時に `ParallelNode` 分岐が実行されない。意図した処理順なら `if` を分離すべき。
- **ParallelNode の全順列探索**
  - `itertools.permutations` により爆発的に探索が増える可能性。上限や早期枝刈りの方針が必要。

### src/modules_core/semantic_judge.py
- **import 時のクライアント生成・環境依存**
  - OpenAI/Gemini クライアントがトップレベルで初期化され、プロキシ設定も固定化。必要時の生成に変更。
- **CSV バッチ処理コードの混在**
  - モジュール本体とスクリプト的コードが混在している。`judge_parallel` を中心にし、バッチ処理は別スクリプトへ。
- **未使用の設定・変数**
  - `INPUT_CSV` / `OUTPUT_CSV` / `log_data` など、現行の `main.py` では未使用。整理対象。

### src/main.py
- **GPU/CPU ワーカーの初期化コスト**
  - `gpu_child_worker` が毎回 `CKYAnalyzer()` を作成。ワーカープール方式にして再利用できると効率的。
- **タイムアウト値が極端に大きい**
  - `GPU_TIMEOUT_SEC` / `CPU_TOTAL_TIMEOUT_SEC` が実質無制限。運用上の停止条件・再試行方針を明確化。
- **ログ出力の散在**
  - CSV ログが多数生成されるが、切り替えスイッチがない。必要に応じて ON/OFF を可能に。
- **巨大 JSON を全件ロード**
  - `cky_json_path` を一括ロードして全行を保持。文数増加時にメモリを圧迫するため、ストリーミングまたはキー単位読み込みを検討。

## 3. 不要・整理候補のスクリプト

以下は現行の `main.py` とは独立しており、ライブラリ本体に残す必然性が薄いものです。用途がある場合は `examples/` や `tools/` へ移動するのが望ましいです。

- `src/modules_core/visual_table.py` のサンプル実行コード
- `src/modules_core/cky_table.py` の `if __name__ == "__main__":` ブロック
- `src/modules_core/bunsetu.py` の `if __name__ == "__main__":` ブロック
- `src/modules_core/clause_analysis.py` の `main()` とテスト文
- `src/modules_core/semantic_judge.py` 内の CSV 一括処理コード（現状はコメントアウトだが混在）

## 4. 進め方の提案（優先度順）

1. **副作用の除去と設定の外部化**（import 時のモデルロード/ログ初期化、ハードコードパスの解消）
2. **データ更新フローの整理**（依存解析とCKYキャッシュの差分更新）
3. **パフォーマンス改善**（`segment_sentences` の二重解析削減、GPU/CPU ワーカーの再利用）
4. **重複ユーティリティの統合**（`visual_table.py` と `cky_table.py`）
5. **未使用コード・スクリプトの整理**（上記セクションのものを分離）
