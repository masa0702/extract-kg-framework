**対応表の構造**

| id | ontology | prompt | prompt\_ex | expected\_out | input | output | info |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |

* id：プロンプトのID  
* ontology：対応するオントロジー  
* prompt：プロンプトの構造  
* prompt\_ex：具体的なプロンプトの例  
* expected\_out：期待する出力  
* input：入力  
* output：出力  
* info：どのような内容のプロンプトかの説明

**注意（実装上の制約）**

* オントロジー検証の出力JSONは、実装側の JSON schema により原則 `{"verdict": ...}` のみ（追加フィールド禁止）です。  
  * 本ドキュメント内の補助変数（A, A', 定義文など）は、判定のために内部で想定する目的で記述しています。
* 本ドキュメントでは `predicate` / `relation` を同義として扱います（実装側の変数名は主に `{relation_ja}`）。

## **プロンプトの選択基準**

まず、relationを型で分類する  
T2KGB（Wikidata由来）の relation は、概ね次の型に分かれる。

1. **タクソノミー型：**  
   * instance of / subclass of / occupation / genre など（「AはBの一種」）

2. **役割（ロール）型：**  
   * director / author / developer / performer / commander など  
     （「AがBに対して果たす役割」）

3. **同値・同名・別名型：**  
   * alias / also known as / 呼称違い（日本語訳ゆらぎが強い）

4. **属性型：**  
   * language / country / publisher / record label など（「BがAの属性値」）

5. **構成・包含型：**  
   * part of / located in / member of など（集合・部分関係）

6. **時間型：**  
   * publication date / point in time など（時間表現）

これらの領域に対して、差別化を行う。

簡易的なまとめとしては

* 前処理（表記揺れ吸収）：No.10  
* 分類・抽象概念への写像：No.17 ＋（必要なら）No.1  
* 役割関係の妥当性（役割構造/NLI）：No.15（補助にNo.4）  
* 役割関係の個体性（通常文↔疑問文のYES/NO）：No.22  
* 最終二値判定（BERT置換先）：No.21

## **1\. 定義ベース判定**

```
あなたは「定義に基づく型整合の2値判定器」です。
入力は predicate と、指定された side における argument と concept です。
subject/object や主語/目的語という概念は使わないでください。

入力:
- predicate: {predicate_ja}
- side: {side}                # "domain" または "range"
- concept: {concept_ja}
- argument: {argument}
- context_sentence: {context_sentence}

手順:
1) predicate と side の文脈で解釈した {argument} の定義文を、日本語で1文だけ生成する（argument_def）。
2) {concept_ja} の定義文を、日本語で1文だけ生成する（concept_def）。
3) 判定は「argument_def が concept_def を含意するか（argument ⇒ concept）」で行う。
   - 含意するなら verdict=1
   - 含意しない、または判断不能なら verdict=0
4) 出力は必ず次のJSON形式のみ（verdictのみ）。思考過程や補助情報は出力しない。

出力(JSONのみ):
{"verdict": 0 or 1}

```

**入出力**

* 入力：  
  * {predicate\_ja}  
  * {side}（domain or range：外部で固定）  
  * {concept\_ja}（sideに対応する概念）  
  * {argument}  
  * {context\_sentence}（任意）  
* 出力  
  * verdict: 1 or 0（必須）  
  * ※ argument_def / concept_def / reason 等は、判定のために内部で想定してもよいが、出力には含めない（実装側のJSON schemaは verdict のみを許可）。

**定義（差別化ポイント）**

* 狙い：  
  * entity が concept に当てはまるかを、「直接質問」ではなく **定義文の同等性/含意**で検証する。  
* 出力：  
  * domain\_ok または range\_ok（片側判定）。  
* 特徴：  
  * 概念の説明能力・百科的知識を“定義”として引き出し、**同値性（≒両方向含意）**で0/1化できる。

**適用範囲（オントロジー / relation 型）**  
最適：タクソノミー型・属性型（特に genre / occupation / type 系）

オントロジー例：

* Movie：genre / film といった分類寄り  
* Music：genre / album / song の分類  
* Book：genre / book 種別  
* Politics / Culture / Nature：社会カテゴリ・自然カテゴリの分類（例：民族、宗教、地形など）

**例（考え方）**

* 例（range側）：  
  * (X, genre, ロマンス映画) で X が “映画作品” らしいかを Xの定義 と 映画作品の定義 の一致で判定。

* 例（domain側）：  
  * (X, occupation, 首相) で Xが職業/役職として成立するかを定義一致で判定。

**他モジュールではなぜ不適か**

* No.15（フレーム役割）：  
  * 分類（is-a）を扱うのは不得手（フレームは事象中心）。  
* No.4（アナロジー）：  
  * 分類の是非を類推でやると根拠が弱くなりやすい。  
* No.10/21（同義語・言い換え）：  
  * 概念自体が同義語で解決しない場合（上位概念への写像が必要）。

## **4\. アナロジー推論モジュール**

```
あなたは「オントロジー関係のアナロジー方向判定器」です。
入力は predicate と、その predicate に係る2つの項（arguments）です。
重要: arg1/arg2 は domain/range のどちらに対応するか未確定です。subject/object や主語/目的語という概念は使わないでください。

入力:
- predicate: {predicate_ja}
- relation_signature: {predicate_ja}({domain_concept_ja}, {range_concept_ja})
- arg1: {arg1}
- arg2: {arg2}
- context_sentence: {context_sentence}

手順:
1) predicate が表す関係を短い日本語テンプレにする（例:「Xの監督はY」）。Xがdomain側、Yがrange側に対応するように書く。
2) 上の関係と同型の代表例ペア（anchor）を2つ挙げる（anchor_domainはdomain概念の代表、anchor_rangeはrange概念の代表）。
3) 次の2方向が「同型の関係」として自然に成立するかをそれぞれ判定する:
   - case1: domain側=arg1, range側=arg2
   - case2: domain側=arg2, range側=arg1
4) ラベルは相互排他で決める:
   - case1のみ成立: label=1
   - case2のみ成立: label=2
   - 両方成立、両方不成立、判断不能、arg2がNULL: label=0
5) 出力は必ず次のJSON形式のみ。reasonは1文で決め手語を含める（思考過程は書かない）。

出力(JSONのみ):
{
  "label": 0 or 1 or 2,
  "assignment": {"domain_side": "arg1|arg2|none", "range_side": "arg1|arg2|none"},
  "relation_template": "...",
  "anchors": [
    {"anchor_domain": "...", "anchor_range": "..."},
    {"anchor_domain": "...", "anchor_range": "..."}
  ],
  "analogy": "(DOMAIN_ARG : RANGE_ARG) ≈ (ANCHOR_DOMAIN : ANCHOR_RANGE)",
  "reason": "..."
}

```

**入出力**

* 入力：  
  * {predicate\_ja}：述語（日本語）  
  * {domain\_concept}：ドメイン概念（日本語ラベル）  
  * {range\_concept}：レンジ概念（日本語ラベル）  
  * {arg1}  
  * {arg2}（1項しかない場合は空文字ではなく "NULL" を渡すことを推奨）  
  * {context\_sentence}：元文（可能なら。空でも可）  
* 出力  
  * verdict: 1 or 0 or 2（必須）  
    * label \= 0：不成立（none）  
    * label \= 1：domain\_side \= arg1 かつ range\_side \= arg2（方向A）  
    * label \= 2：domain\_side \= arg2 かつ range\_side \= arg1（方向B）  
  * analogy: 1行のアナロジー式（必須）  
    * 形式：(subject : object) ≈ (anchor\_domain : anchor\_range)  
  * anchors: 代表例ペア（必須、2組）  
    * anchor\_domain: 代表的なドメイン例  
      （例：映画、楽曲、書籍、チーム）  
    * anchor\_range: 代表的なレンジ例  
      （例：監督、作曲者、著者、所属リーグ）  
  * reason: 1文（必須、決め手の語を含める）

**定義（差別化ポイント）**

* 狙い：  
  * predicate が 役割型のとき、(domain\_concept → range\_concept) を 既知の代表ペアに写像し、候補 triple が「同型の役割関係」かを検証する。  
* 出力：  
  * 基本は片側（domain または range）の妥当性を補強するために使い、最終判定は No.21 と組み合わせるのが安全。

**適用範囲**  
最適：役割（ロール）型（director, author, developer, performer, commander, coach 等）  
オントロジー例：

* Movie：監督/出演/脚本  
* Music：作曲者/作詞者/演奏者  
* Book：著者  
* Computer：開発者/プログラミング言語（「ソフト←言語」は役割というより属性だが、類推の補助は可能）  
* Military：指揮官/参加者  
* Sport：チームとリーグ/選手とチーム など

**例（考え方）**  
監督(映画, 人間) を「監督 : 映画 \= 著者 : 本」のようにロール対応で検査し、object が人間カテゴリとして成立するかを補強する。

**他モジュールではなぜ不適か**

* No.17（階層分類）：  
  * ロールはis-a階層だけでは判定しづらい（“人間”で十分だが役割の適否が残る）。  
* No.1（定義）：  
  * ロール関係は「Aとは何か」より「AがBに対して何をするか」で決まる。  
* No.10（同義語生成）：  
  * 語彙の言い換えだけでは役割の同型性は保証できない。

## **10\. 同義語生成による整合チェック**

```
あなたは「同義語生成による整合性回復の2値ゲート」です。
入力は predicate と relation_signature、および2つの項（arguments）です。
重要: subject/object や主語/目的語という概念は使わないでください。

目的:
- 表現揺れ（同義語・別表記・言い換え）が主要因で整合性が崩れている可能性が高い場合のみ verdict=1 とし、
  同義語候補を生成する。
- 表現揺れでは解決しない場合は verdict=0 とする（同義語候補は空でもよい）。

入力:
- predicate: {predicate_ja}
- relation_signature: {predicate_ja}({domain_concept_ja}, {range_concept_ja})
- arg1: {arg1}
- arg2: {arg2}
- context_sentence: {context_sentence}

判断規則:
1) verdict=1 にしてよいのは、次のいずれかが強い場合のみ:
   a) predicate の訳語揺れ・同義語がありうる
   b) domain_concept / range_concept の抽象語に同義語がありうる
   c) arg1/arg2 が略称・外来語・カタカナ表記・表記揺れを含み、別表記がありうる
2) 役割構造や階層関係の誤りが主因で、同義語では救えない場合は verdict=0。
3) verdict=1 の場合のみ、対象（targets）を選び、各対象の同義語/別表記を最大3件ずつ生成する（短い語/短句のみ）。
4) 生成結果を使って、関係の種類が伝わる「正規化した関係テンプレ」normalized_relation を1つ作る（例:「Xの監督はY」）。
   domain/range の割当は確定しなくてよい。
5) 出力は必ず次のJSON形式のみ。reasonは1文で決め手語を含める（思考過程は書かない）。

出力(JSONのみ):
{
  "verdict": 0 or 1,
  "targets": ["predicate|domain_concept|range_concept|arg1|arg2", ...],
  "synonyms": {
    "predicate": ["...", "...", "..."],
    "domain_concept": ["...", "...", "..."],
    "range_concept": ["...", "...", "..."],
    "arg1": ["...", "...", "..."],
    "arg2": ["...", "...", "..."]
  },
  "normalized_relation": "...",
  "reason": "..."
}
```

**入出力**

* 入力：  
  * {predicate\_ja}  
  * {domain\_concept\_ja}  
  * {range\_concept\_ja}  
  * {arg1}  
  * {arg2}（無い場合は "NULL"）  
  * {context\_sentence}（任意だが推奨）  
* 出力：  
  * verdict: 0/1（同義語置換で改善見込みが高い=1）  
  * targets: 同義語生成対象（predicate / domain\_concept / range\_concept / arg1 / arg2 のどれを置換するか）  
  * synonyms: 対象ごとの同義語リスト（最大3、短い語を優先）  
  * normalized\_relation: 同義語置換後の「関係テンプレ」（後続モジュールに渡すため）  
  * reason: 1文（決め手の語を含める）

**定義（差別化ポイント）**

* 狙い：  
  * 日本語訳のゆらぎ・表記差を吸収するために、entity または predicate の 同義語候補集合を作り、置換しても成立するかで0/1判定する。  
* 位置づけ：  
  * **言い換え生成**側に寄ったモジュール。単独で最終判定にせず、No.21（BERT代替可能な判定器）に渡すための前処理として強い。

**適用範囲**  
最適：同値・同名・別名型、および日本語訳揺れが強い概念・役割名

オントロジー例：

* Politics：役職・機関名の表記揺れ（「首相/総理」など）  
* Culture：文化語彙の揺れ  
* Sport：大会名・役割名（「選手/競技者」等）  
* Computer：技術用語の揺れ（「OS/オペレーティングシステム」等）  
* Movie/Music/Book：職能（「監督/ディレクター」「作曲/コンポーズ」など）

**例（考え方）**  
predicateが和訳でぶれる場合、predicate候補を生成→置換→No.21の判定に掛けて安定化する。

**他モジュールではなぜ不適か**

* No.1（定義）：  
  * 定義生成は強いが計算コストが重く、表記揺れだけの問題には過剰。  
* No.15（フレーム）：  
  * 語彙揺れの吸収が目的ならフレームは遠回り。  
* No.17（階層）：  
  * 同義語は階層ではない（同一レベルの別表現）。

## **15\. フレーム役割チェック**

```
あなたは「フレーム役割（ロール）整合の厳密2値判定器」です。
入力は relation と、指定された side における argument と concept です。
重要: subject/object や主語/目的語という概念は使わないでください。
このモジュールは方向を推定せず、side固定で判定します。

判定方針（精度優先）:
- concept はオントロジー上の型（クラス）。argument はその型の「インスタンス（個体/実体/値）」として適格かを判定する。
- argument が文の断片・活用語尾つき・助詞つき・役割語そのもの等で、個体/値として成立しない場合は必ず verdict=0。
- 判断不能は必ず verdict=0。

入力:
- relation: {relation_ja}
- side: {side}                # "domain" または "range"
- concept: {concept_ja}
- argument: {argument}
- other_argument: {other_argument}   # 無ければ "NULL"
- context_sentence: {context_sentence}

手順:
1) 文脈優先: context_sentence があれば最優先で参照し、この文脈で argument が何を指すかを解釈する。
2) 個体性フィルタ（最重要）: 次のいずれかなら verdict=0 とする。
   - 文の断片/活用/助詞付き: 「〜で」「〜に」「〜を」「監督し」「アニメで」など
   - 役割語/関係語そのもの: 「監督」「出演者」「脚本」「制作会社」など（特定の固有名ではない）
   - concept と同一/ほぼ同一で、個体名（固有名・実体名）ではない
3) relation が喚起するフレーム（イベント/状況）を短く命名する（frame）。
4) side={side} に対応する必須役割名（role_name）を短く1つ定義する（concept と整合する語にする）。
5) argument と other_argument（NULLなら省略）を含む短い関係文Aを日本語1文で作る（意味は変えない）。
6) 「診断的特性」を使った言い換え: concept={concept_ja} の診断的特性（その型らしさを示し、混同しやすい非該当例を排除できる性質）を1つ選ぶ。
   A の意味を保ちつつ、argument が concept の個体として成立し、その診断的特性を満たすことが読み取れる言い換え文 A' を日本語1文で作る。
7) 判定: A が A' を含意する（A ⇒ A'）なら verdict=1。含意しない/不明なら verdict=0。
8) 出力は必ず次のJSON形式のみ（verdictのみ）。
   思考過程や補助情報は出力しない。
9) 出力(JSONのみ):
{"verdict": 0 or 1}
```

**入出力**

* 入力：  
  * {relation\_ja}  
  * {side}（domain or range：外部で固定）  
  * {concept\_ja}（sideに対応する概念）  
  * {argument}  
  * {other\_argument}（無ければ "NULL"）  
  * {context\_sentence}（任意だが推奨）  
* 出力：  
  * verdict: 0/1（整合=1、不整合/判断不能=0）


**定義（差別化ポイント）**

* 狙い：  
  * relation を「事象フレーム」と見なし、argument がそのフレームの **必須役割（role filler）**を満たすかを判定する。  
* 重要：  
  * ここでの差別化は、単に「人か？」ではなく、(i) 個体性（断片/役割語を棄却）と (ii) 診断的特性つきの言い換え（含意/NLI）で、誤通過を減らす点。

**適用範囲**  
最適：役割（ロール）型のうち、特に 事象性が強いもの

オントロジー例：

* Movie：出演（作品←俳優）、監督（作品←人物）  
* Music：作曲（楽曲←人物）、歌唱/演奏（楽曲←人物）  
* Sport：参加（大会←選手/チーム）、所属（選手←チーム/クラブ）  
* Military：指揮（戦闘/部隊←指揮官）、参加（戦争←国家/部隊）  
* Politics：就任/任命系があるなら強い（ただし ontology の relation が静的属性中心なら適用は限定）

**例（考え方）**  
出演(映画作品, 俳優) のとき、object が**“人”**であるだけでなく 「出演者」という役割に適合する人間カテゴリかを検証（例：場所や組織が来たら落とす）。

**他モジュールではなぜ不適か**

* No.1（定義）：フレームは定義より役割構造（誰が何をするか）が中心。  
* No.17（階層）：階層だけだと「人間」で通ってしまい、ロールの誤り（例：人物だが出演者ではない）を弾きにくい。  
* No.10（同義語生成）：語彙置換で役割構造の正しさは保証できない。

## **17\. 階層分類チェック（タクソノミーの整合）**

```
あなたは「タクソノミー（is-a階層）整合の2値判定器」です。
入力は predicate と、指定された side における argument と concept です。
重要: subject/object や主語/目的語という概念は使わないでください。
このモジュールは方向を推定せず、side固定で判定します。

入力:
- predicate: {predicate_ja}
- side: {side}                # "domain" または "range"
- concept: {concept_ja}
- argument: {argument}
- context_sentence: {context_sentence}

手順:
1) context_sentence がある場合は文脈に沿って、argument の上位概念（ハイパーニム）を最大3段まで列挙する。
   形式: ["argument","hyper1","hyper2","hyper3"]（名詞の短語のみ）。
2) concept={concept_ja}（または明確な同義）が、この上位概念列に含まれるかを判定する。
   - 含まれる/明確に含意されるなら verdict=1
   - 含まれないなら verdict=0
   - 判断不能なら verdict=0
3) 出力は必ず次のJSON形式のみ。reasonは1文で決め手語を含める（思考過程は書かない）。

出力(JSONのみ):
{"verdict": 0 or 1, "chain": ["...","...","...","..."], "match": "yes|no|unknown", "reason": "..."}
```

**入出力**

* 入力：  
  * {predicate\_ja}  
  * {domain\_concept\_ja}  
  * {range\_concept\_ja}  
  * {arg1}  
  * {arg2}（無い場合は "NULL"）  
  * {context\_sentence}（任意だが推奨：固有名の種別推定に効く）  
* 出力：  
  * label: 0/1  
    * label=0：どちらも整合しない  
    * label=1：成立  
  * assignment: domain側/range側に割当てた arg  
  * taxonomy\_evidence: 各 arg の上位概念列（最大3段）  
  * match: domain\_concept/range\_concept への一致状況  
  * reason: 1文（決め手語を含める、思考過程禁止）


**定義（差別化ポイント）**

* 狙い：  
  * entity → 上位概念 の上位下位関係（is-a）を仮想的にたどらせ、domain/range concept に 上位包含されるかで判定する。  
* 意義：  
  * 日本語訳では具体名が多く、concept は抽象名なので、**“上位概念への写像”**が必要になる場面に強い。

**適用範囲**  
最適：タクソノミー型、属性型（特に抽象カテゴリがrangeに来るもの）

オントロジー例：

* Nature：生物/地形/自然物カテゴリ（上位概念が強く効く）  
* Culture：宗教/言語/文化圏など抽象カテゴリ  
* Politics：政党/役職/国家機関など  
* Movie/Music/Book：ジャンル、作品タイプ（映画/楽曲/書籍）  
* Space：天体種別（惑星/衛星/恒星などがあれば）

**他モジュールではなぜ不適か**

* No.15（フレーム）：分類関係を事象フレームで扱うのは不自然。  
* No.4（アナロジー）：階層整合は類推より包含関係で判定した方が説明可能。  
* No.10（同義語生成）：同義語は上位概念写像と異なる。

## **21\. 同義語・言い換えベース判定モジュール**

**現状観測（ver10.0 / ont_1_movie / 20260206_180709）**

* 実行結果ディレクトリ: `results/ver10.0/extract_pred_arg_pair/extract_target_data/ont_1_movie_extract_target/select_mode/20260206_180709__mode-default/`
* `mode=default` の抽出件数:
  * `default_ont_1_movie_extract_target_extracted_triples.jsonl` は 376文中 合計131 triple
* `mode=no_verification` の抽出件数（比較対象）:
  * `results/ver10.0/extract_pred_arg_pair/extract_target_data/ont_1_movie_extract_target/select_mode/20260206_153622__mode-no_verification/no_verification_ont_1_movie_extract_target_extracted_triples.jsonl` は 376文中 合計1276 triple
* `mode=default` の prompt 呼び出し内訳（`default_ont_1_movie_extract_target_prompt_log.jsonl` 2321行、accept=verdictが1/2）:
  * `prompt_id=22`: 1431回、accept 445回、accept率 0.311
  * `prompt_id=17`: 161回、accept 79回、accept率 0.491
  * `prompt_id=21`: 727回、accept 2回、accept率 0.003（致命的）

**なぜ recall が下がるか（因果）**

* `mode=default` は「候補抽出」後に「LLM検証で verified のみ残す」ため、検証の false negative がそのまま recall 低下になる。
* `prompt_id=21` が 727回中 2回しか accept していないため、`prompt_id=21` に割り当てられている relation 群の候補がほぼ全滅する。
  * 実測では `prompt_id=21` が主に（日本語表層として）「物語の舞台」「本国」「登場人物」「受賞・受章」「派生元」「ノミネート」「撮影地」などを扱っているが、ほぼ通らない。
* 旧 No.21 は A→A' の「含意（A ⇒ A'）」を要求し、かつ「判断不能は0」を強制するため、モデルが保守的に 0 を返しやすい。temperature=0 の決定的デコードとも相性が悪い。
* まれに通る `verdict=1` が誤り（方向/概念取り違え等）になり得るため、precision にも悪影響が出る。

```
あなたは「言い換え＋文脈解釈による型整合の2値判定器（No.21 改訂版）」です。
入力は relation と、指定された side における argument と concept です。
重要: subject/object や主語/目的語という概念は使わないでください。
このモジュールは方向を推定せず、side固定で判定します。

入力:
- relation: {relation_ja}
- side: {side}                # "domain" または "range"
- concept: {concept_ja}
- argument: {argument}
- other_argument: {other_argument}   # 無ければ "NULL"
- context_sentence: {context_sentence}

判定方針（バランス重視）:
- context_sentence があれば最優先で参照し、この文脈で argument が何を指すかを解釈する。
- other_argument は補助情報として扱い、argument の役割（作品名/人物名/場所/賞/国/日付など）推定に使う。
- 次の「個体性NG」に該当する場合は必ず verdict=0:
  - 文の断片/活用語尾つき/助詞つき: 「〜で」「〜に」「〜を」「監督し」「アニメで」「〜という」など
  - 役割語/関係語そのもの: 「監督」「出演者」「脚本」「制作会社」「登場人物」「本国」など（固有の個体名ではない）
  - 記号だけ/空文字/曖昧すぎる一般語（「一部」「続く」など）
- 迷ったら verdict=0（ただし上の否定例で先に落とす）。

手順:
1) relation と side の文脈で、argument が「concept の個体（実体/値）」として成立するかを判定する。
2) YES の場合のみ verdict=1。NO/UNKNOWN（判断不能）は verdict=0。

補助（判断アンカー）:
- 例: relation が「物語の舞台」「撮影地」「本国」の場合、range は通常「都市/国/場所の固有名」になりやすい。
- 例: relation が「登場人物」の場合、range は通常「人物名/キャラクター名」になりやすい。
- 例: relation が「受賞」「ノミネート」の場合、range は通常「賞（固有名または賞カテゴリ）」になりやすい。

出力(JSONのみ):
{"verdict": 0 or 1}
```

**入出力**

* 入力：  
  * {relation\_ja}：述語（日本語）  
  * {side}：domain または range（外部から固定で渡す。プロンプト内で判断しない）  
  * {concept\_ja}：side に対応する concept 名（日本語）  
  * {argument}：判定対象の項（arg）  
  * {other\_argument}：もう片方の項（存在する場合のみ。無ければ "NULL"）  
  * {context\_sentence}：元文（任意だが推奨）  
* 出力：  
  * verdict: 0/1（整合=1、不整合/判断不能=0）
  * ※ reason 等は内部で想定してもよいが、出力には含めない（実装側のJSON schemaは verdict のみを許可）。


**定義（差別化ポイント）**  
狙い：

* 旧 No.21 の「A⇒A'（含意）」は保守的に 0 へ寄りやすく、default の recall を大きく下げる。
* 改訂 No.21 は「言い換え＋文脈解釈」で relation を理解しつつ、最終判定は No.22 に近い「個体（値）としての型整合」を直接 0/1 化する。
* ただし No.22 よりも「relation を読んでタイプを推定する」色を強め、場所/人物/賞/国などの relation で落とし過ぎを減らすことを狙う。

**適用範囲**  
最適：役割・場所・人物・賞・国など「range が値（個体）になる」relation の型整合  
補助：taxonomy（genre/typeなど）は No.17 の方が安定しやすい

**推奨生成パラメータ（実装対応案）**

* 改訂 No.21 は「解釈・言い換え」が絡むため、`temperature=0.1〜0.2` を選べるようにする（他promptは原則0で良い）。

**他モジュールではなぜ不適か**  
No.1/17/15/4は“専門モジュール”。  
No.21は「relationの解釈を織り込んだ型整合」を狙うが、taxonomy は No.17 の方が説明しやすい。  

## **22\. QA型（通常文↔疑問文）型整合チェック**

```
あなたは「通常文↔疑問文(QA)による型整合の2値判定器」です。
入力は relation と、指定された side における argument と concept です。
重要: subject/object や主語/目的語という概念は使わないでください。
このモジュールは方向を推定せず、side固定で判定します。

判定方針（精度優先）:
- まず通常文Sを作り、次に同内容の疑問文Qに変換し、QにYES/NOで答える（頭の中で）。
- YESのときのみ verdict=1。NO/UNKNOWN（判断不能）は verdict=0。

入力:
- relation: {relation_ja}
- side: {side}                # "domain" または "range"
- concept: {concept_ja}
- argument: {argument}
- other_argument: {other_argument}   # 無ければ "NULL"
- context_sentence: {context_sentence}

手順:
1) 通常文S（日本語1文）を作る。
   Sは「この文脈で argument は concept の“個体（実体/値）”を指す」という意味を必ず含める。
2) Sを、意味を変えずに疑問文Q（はい/いいえで答えられる形、日本語1文）に変換する。
3) context_sentence と一般常識に基づき、Qへの答えを YES/NO/UNKNOWN で決める（頭の中で）。
   - 注意: 「監督」「制作会社」などの役割語・一般名詞は、通常は特定の個体名ではないため YES にしない。
   - 「〜で」「〜し」などの断片も YES にしない。
4) answer=YES のときのみ verdict=1。それ以外は verdict=0。
5) 出力は必ず次のJSON形式のみ（verdictのみ）。
出力(JSONのみ):
{"verdict": 0 or 1}
```

**入出力**

* 入力：  
  * {relation\_ja}  
  * {side}（domain or range：外部で固定）  
  * {concept\_ja}（sideに対応する概念）  
  * {argument}  
  * {other\_argument}（無ければ "NULL"）  
  * {context\_sentence}（任意だが推奨）  
* 出力：  
  * verdict: 0/1（整合=1、不整合/判断不能=0）

**定義（差別化ポイント）**

* 狙い：  
  * 「通常文→疑問文」への言い換えと YES/NO で、concept への“個体（インスタンス）”としての適格性を強く二値化する。  
* 特徴：  
  * 役割語そのもの（例：「監督」）や文の断片（例：「〜で」「監督し」）を YES にしないことで、誤通過を減らす。

**No.21 との使い分け**

* No.22 は「個体性フィルタが強い」ため precision を上げやすいが、保守的に 0 に寄って recall を落とす場合がある。
* No.21（改訂版）は「relation を読んでタイプ推定」を強め、場所/人物/賞/国などの relation で落とし過ぎを減らす用途を想定する。
* taxonomy（genre/type等）は原則 No.17 を優先し、No.21/22 は補助に回す。

---

## **プロンプト検証の改良点（実装対応案）**

* `prompt_id=21` の accept率監視を導入する（prompt_log から prompt_id 別 accept率を算出し、異常（例: < 1%）なら警告）。
* Fallback設計（バランス重視）:
  * `prompt_id=21` が 0 の場合に限り、`prompt_id=22` を追加で当てて 1 なら救済する案（ただし個体性NGなら救済しない）。
  * `prompt_id=17` は taxonomy（genre/type等）専用に寄せ、役割・場所・人物の判定には使わない。
* 概念欠損の扱い:
  * `relation_prompt_map` の domain/range concept が空の pid は default で丸ごと落ちやすい。
  * concept を埋めるか、concept 不要の汎用promptへルーティングする方針を明記し、運用で避ける。

## **検証手順（回帰評価の最低ライン）**

```
# 1) default 実行（例）
python src/select_mode_main.py --mode default --input_jsonl_dir data/T2KGB_JA/extract_target_data --cache_mode refresh

# 2) prompt_log の集計（prompt_id別 accept率）
python - <<'PY'
import json
from collections import Counter
path='results/ver10.0/extract_pred_arg_pair/extract_target_data/ont_1_movie_extract_target/select_mode/20260206_180709__mode-default/default_ont_1_movie_extract_target_prompt_log.jsonl'
c=Counter(); a=Counter()
for line in open(path,'r',encoding='utf-8'):
    if not line.strip(): continue
    r=json.loads(line)
    pid=str(r.get('prompt_id',''))
    v=r.get('verdict',0)
    c[pid]+=1
    if v in (1,2): a[pid]+=1
for pid,tot in c.most_common():
    print(pid, tot, a[pid], f'{a[pid]/tot:.3f}')
PY

# 3) extracted_triples の合計数（簡易）
python - <<'PY'
import json
path='results/ver10.0/extract_pred_arg_pair/extract_target_data/ont_1_movie_extract_target/select_mode/20260206_180709__mode-default/default_ont_1_movie_extract_target_extracted_triples.jsonl'
n_sent=0; n_tr=0
for line in open(path,'r',encoding='utf-8'):
    if not line.strip(): continue
    r=json.loads(line)
    n_sent += 1
    n_tr += len(r.get('extracted_triples') or [])
print('sentences',n_sent,'triples',n_tr)
PY
```

**Acceptance Criteria（暫定）**

* `prompt_id=21` の accept率が極端に低くない（目標: 10%以上）。
* `ont_1_movie` の `mode=default` の triple 数が、旧 default（131）から有意に増える（暫定目標: 数百）。
* 既知の誤通過（役割語そのもの、断片）が増えていない（No.22/15 の個体性NGで抑制）。

## **実装対応チェックリスト**

* [ ] `prompts/prompts.json` の `id=21` を本ドキュメントの改訂版に差し替える
* [ ] `prompt_id=21` の温度を個別設定できるようにする（例: `OntologyJudgeConfig` か prompt_id別パラメータ）
* [ ] `prompts/relation_prompt_map.json` の prompt 割当を見直す（taxonomy→17、個体性→22、解釈込み型整合→改訂21）
* [ ] prompt_id別 accept率の自動集計と警告を追加する
* [ ] `ont_1_movie` / `ont_2_music` で回帰評価し、precision/recall の差分を記録する
