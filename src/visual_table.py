import pprint

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


import json

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



# 使用例
cky_table_example = [
    ["",   1,     2,     3],
    [1,   {"candidate":"猫", "score":5},  0,  {"candidate":"動いた", "score":2}],
    [2,    0,    {"candidate":"は", "score":3},  0],
    [3,    0,     0,    {"candidate":"走る", "score":10}],
]

tsv_str = cky_table_to_tsv(cky_table_example)
print(tsv_str)


# # CKY表の例
# cky_table_example = [
#     ["",   1,     2,     3],
#     [1,   {"candidate":"猫", "score":5},  0,  {"candidate":"動いた", "score":2}],
#     [2,    0,    {"candidate":"は", "score":3},  0],
#     [3,    0,     0,    {"candidate":"走る", "score":10}],
# ]

# display_multiline_cky_table(cky_table_example, width=40)
