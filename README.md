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

#### vLLM / LLM-jp の注意

`src/main.py` は並列判定・オントロジー検証で vLLM(OpenAI互換API) にアクセスします。
docker-compose 等で `LLMJP_*` / `LLMJP_ONTO_*` が正しく設定されていない場合、起動時に preflight で停止します。

- preflight を無効化: `ONTOLOGY_VERIFY_PREFLIGHT=0`
- LLM 失敗時に継続（非推奨）: `ONTOLOGY_VERIFY_STRICT=0`
