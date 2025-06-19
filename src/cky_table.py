import json
import pprint

class CkyTable:
    @staticmethod
    def create_initializing_cky_table(clauses_list: list) -> list:
        """
        文の文節データを入力として、初期CKY表を作成

        Args:
            clauses_list: 文の文節データ
                [
                    [clause_str, span, [tokens], [conf_token_pos], [config_token_span]],
                    [],
                    ...,
                    []
                ]

        Returns:
            list: ２次元リストで表現したCKY表
        """
        clause_num = len(clauses_list)
        cky_table_matrix = [[0] * (clause_num + 1) for _ in range(clause_num + 1)]

        # CKY表のヘッダーを設定
        for i in range(1, clause_num + 1):
            cky_table_matrix[0][i] = str(i)  # 列ヘッダー
            cky_table_matrix[i][0] = str(i)  # 行ヘッダー

        # 対角線に dict形式の文節データを配置
        for index, clause_element in enumerate(clauses_list):
            """
            clause_element は以下いずれかの形式を想定する:

            [clause_str, span, tokens, upos_list, xpos_list, token_span]
            [clause_str, span, tokens, pos_list, token_span]

            既存のデータとの互換性を保つため、項目数によって判断する。
            """

            surface = clause_element[0]
            span = clause_element[1]
            tokens = clause_element[2]
            if len(clause_element) >= 6:
                upos_list = clause_element[3]
                xpos_list = clause_element[4]
                token_span = clause_element[5]
            else:
                # 旧形式: 第4要素に品詞(XPOS 相当) が格納されている
                upos_list = clause_element[3]
                xpos_list = clause_element[3]
                token_span = clause_element[4]

            clause_dict = {
                "id": index + 1,
                "candidate": surface,
                "span": span,
                "tokens": tokens,
                "upos": upos_list,
                "xpos": xpos_list,
                # 互換性のため 'pos' も XPOS を入れておく
                "pos": xpos_list,
                "token_span": token_span,
            }
            cky_table_matrix[index + 1][index + 1] = clause_dict

        return cky_table_matrix

    @staticmethod
    def display_simple_cky_table(cky_table: list):
        """
        CKY表を番号と文節文字列だけで簡潔に表示する。

        Args:
            cky_table: CKY表を表す２次元リスト
        """
        print("簡易CKY表:")
        for row_index, row in enumerate(cky_table):
            simplified_row = []
            for col_index, cell in enumerate(row):
                # ヘッダー行・列
                if row_index == 0 or col_index == 0:
                    simplified_row.append(cell)
                else:
                    # セルがdict形式(= 対角線の文節など)の場合
                    if isinstance(cell, dict):
                        # "candidate" を表示
                        simplified_row.append(cell.get("candidate", ""))
                    # それ以外(まだ0のままなど)の場合
                    else:
                        simplified_row.append("0")
            print("\t".join(map(str, simplified_row)))


    @staticmethod
    def process_json_to_cky_and_save(input_json_file: str, output_json_file: str):
        """
        JSONファイルを読み込み、各文に対してCKY表を生成し、結果をJSONに保存する。

        Args:
            input_json_file: 入力JSONファイルのパス
            output_json_file: 出力JSONファイルのパス
        """
        try:
            with open(input_json_file, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        except FileNotFoundError:
            print(f"ファイル {input_json_file} が見つかりません。")
            return
        except json.JSONDecodeError:
            print(f"ファイル {input_json_file} の読み込み中にエラーが発生しました。")
            return

        for sentence, data in json_data.items():
            print(f"\n文: {sentence}")
            clauses_list = data.get("clauses", [])
            cky_table = CkyTable.create_initializing_cky_table(clauses_list)

            # CKY表の簡易表示（必要に応じて削除可）
            CkyTable.display_simple_cky_table(cky_table)

            # CKY表を JSON データの dependency_table に格納
            json_data[sentence]["dependency_table"] = cky_table

        # 更新されたJSONを保存
        with open(output_json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

        print(f"CKY表を {output_json_file} に保存しました。")


    @staticmethod
    def display_multiline_cky_table(cky_table: list, width=60):
        """
        CKY表を複数行に対応した形で綺麗に表示する。
        辞書はpprintで整形し、列ごとに幅を合わせつつ表示する。

        Args:
            cky_table: CKY表を表す二次元リスト
            width: pprintで整形する際の幅（必要に応じて調整）
        """
        # 1. セル内容を複数行に分割して持つリストに変換
        #    table_lines[row][col] = ["line1", "line2", ...]
        table_lines = []
        for row in cky_table:
            row_data = []
            for cell in row:
                if isinstance(cell, dict):
                    # pprintで整形 → splitlines() で複数行に分割
                    cell_str = pprint.pformat(cell, width=width, compact=False)
                    lines = cell_str.splitlines()
                else:
                    # 辞書以外は単純にstr化（必要であれば改行で分割してもOK）
                    lines = [str(cell)]
                row_data.append(lines)
            table_lines.append(row_data)

        # 2. 列ごとの最大幅を算出
        #    まず、列数の最大値を把握
        max_col_count = max(len(row_data) for row_data in table_lines)
        col_widths = [0] * max_col_count

        # 各セルの複数行の中で最長の行長を列の幅とする
        for row_data in table_lines:
            for col_idx, lines in enumerate(row_data):
                max_line_length = max(len(line) for line in lines) if lines else 0
                if max_line_length > col_widths[col_idx]:
                    col_widths[col_idx] = max_line_length

        # 3. 行の出力
        #    各row_dataについて、セルが複数行ある場合は最大行数に揃えて出力
        print("マルチライン対応CKY表:\n")
        for row_data in table_lines:
            # この行での最大行数
            max_lines_in_row = max(len(lines) for lines in row_data)

            # サブ行（同じCKY表の行に対して、辞書などの複数行を縦に展開）のループ
            for sub_line_index in range(max_lines_in_row):
                sub_line_cells = []
                for col_idx, lines in enumerate(row_data):
                    # 現在のセルに sub_line_index 行目があれば取得、なければ空文字
                    if sub_line_index < len(lines):
                        cell_line = lines[sub_line_index]
                    else:
                        cell_line = ""
                    # 幅(col_widths[col_idx])に合わせて左寄せ
                    sub_line_cells.append(cell_line.ljust(col_widths[col_idx]))
                # 列間の区切り文字を定義（必要に応じて調整）
                print(" | ".join(sub_line_cells))
            # 行間の空白行を入れるなど、見やすくしたい場合はこちらに追記
            # print()  # 例: 行と行の間に1行空ける


    @staticmethod
    def cky_table_to_tsv(cky_table):
        """
        CKY表（2次元リスト）をTSV文字列に変換する。

        各セルがdict, listの場合はjson.dumpsで1行str化して埋める。
        0や数字はそのままstrで埋める。
        """
        tsv_lines = []
        for row in cky_table:
            row_strs = []
            for cell in row:
                if isinstance(cell, (dict, list)):
                    # 1行で収まるようにcompactなjson
                    cell_str = json.dumps(cell, ensure_ascii=False)
                else:
                    cell_str = str(cell)
                row_strs.append(cell_str)
            # タブ区切りで結合
            tsv_lines.append("\t".join(row_strs))
        # 改行区切りで結合
        tsv_table="\n".join(tsv_lines)
        
        return print(tsv_table)


if __name__ == "__main__":
    # 入力と出力のJSONファイル名
    CkyTable = CkyTable()
    input_json = "../data/dependency_analysis.json"  # 先ほど作成したJSONファイル
    output_json = "../data/dependency_analysis_with_cky.json"

    # # CKY表を生成して保存
    CkyTable.process_json_to_cky_and_save(input_json, output_json)
    
    data_path = "../data/dependency_analysis_with_cky.json"
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except:
        pass
    for sentence, data in json_data.items():
        cky_table = json_data[sentence]["dependency_table"]
        # CkyTable.display_simple_cky_table(cky_table)
        CkyTable.display_multiline_cky_table(cky_table)
        # print(cky_table)