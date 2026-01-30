import json
from typing import Any, Iterable, List, Sequence

from config.filter_settings import PARALLEL_KEYS


class CkyTable:
    @staticmethod
    def create_initializing_cky_table(clauses_list: Sequence[Sequence[Any]]) -> List[List[Any]]:
        """文節リストを CKY 表の対角線に配置した 2D 配列を返す。"""
        clause_num = len(clauses_list)
        cky_table_matrix: List[List[Any]] = [[0] * (clause_num + 1) for _ in range(clause_num + 1)]

        for i in range(1, clause_num + 1):
            idx_str = str(i)
            cky_table_matrix[0][i] = idx_str  # 列ヘッダー
            cky_table_matrix[i][0] = idx_str  # 行ヘッダー

        for index, clause_element in enumerate(clauses_list):
            clause = CkyTable._normalize_clause_entry(clause_element, index)
            cky_table_matrix[index + 1][index + 1] = clause

        return cky_table_matrix

    # -------------------------------------------------------------
    #  Utility methods for cell access
    # -------------------------------------------------------------
    @staticmethod
    def get_cell_span_text(clauses: list, i: int, j: int) -> str:
        """Return concatenated surface text for clauses[i-1:j]."""

        if i > j:
            return ""
        surfaces = [cl[0] for cl in clauses[i - 1 : j]]
        return "".join(surfaces)

    @staticmethod
    def count_parallel_keys(clauses: list, i: int, j: int, keys: list | None = None) -> int:
        """Count occurrences of parallel connective keys within a clause span."""

        text = CkyTable.get_cell_span_text(clauses, i, j)
        search_keys = PARALLEL_KEYS if keys is None else keys
        return sum(text.count(k) for k in search_keys)

    @staticmethod
    def display_simple_cky_table(cky_table: list):
        """Print clause indices and surface strings for debugging."""
        print("簡易CKY表:")
        for row_index, row in enumerate(cky_table):
            simplified_row = []
            for col_index, cell in enumerate(row):
                if row_index == 0 or col_index == 0:
                    simplified_row.append(cell)
                else:
                    if isinstance(cell, dict):
                        simplified_row.append(cell.get("candidate", ""))
                    else:
                        simplified_row.append("0")
            print("\t".join(map(str, simplified_row)))


    @staticmethod
    def process_json_to_cky_and_save(
        input_json_file: str,
        output_json_file: str,
        *,
        verbose: bool = False,
    ):
        """Load dependency JSON, attach CKY tables, and write out."""
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
            clauses_list = data.get("clauses", [])
            cky_table = CkyTable.create_initializing_cky_table(clauses_list)

            if verbose:
                CkyTable.display_simple_cky_table(cky_table)

            json_data[sentence]["dependency_table"] = cky_table

        with open(output_json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

        if verbose:
            print(f"CKY表を {output_json_file} に保存しました。")

    @staticmethod
    def build_entry_from_clauses(clauses_list: Sequence[Sequence[Any]]) -> Dict[str, Any]:
        """Create a minimal CKY entry dict from clause list."""
        cky_table = CkyTable.create_initializing_cky_table(clauses_list)
        return {
            "clauses": list(clauses_list),
            "dependency_table": cky_table,
        }

    @staticmethod
    def display_multiline_cky_table(cky_table: list, width: int = 60) -> str:
        """Return a formatted multiline table string (no implicit print)."""
        table_lines: list[list[list[str]]] = []
        for row in cky_table:
            row_data = []
            for cell in row:
                if isinstance(cell, dict):
                    cell_str = json.dumps(cell, ensure_ascii=False, indent=2)
                    lines = cell_str.splitlines()
                else:
                    lines = [str(cell)]
                row_data.append(lines)
            table_lines.append(row_data)

        max_col_count = max(len(r) for r in table_lines)
        col_widths = [0] * max_col_count
        for row_data in table_lines:
            for col_idx, lines in enumerate(row_data):
                max_line_length = max(len(line) for line in lines) if lines else 0
                col_widths[col_idx] = max(col_widths[col_idx], max_line_length)

        out_lines: list[str] = ["マルチライン対応CKY表:"]
        for row_data in table_lines:
            max_lines_in_row = max(len(lines) for lines in row_data)
            for sub_line_index in range(max_lines_in_row):
                sub_line_cells = []
                for col_idx, lines in enumerate(row_data):
                    cell_line = lines[sub_line_index] if sub_line_index < len(lines) else ""
                    sub_line_cells.append(cell_line.ljust(col_widths[col_idx]))
                out_lines.append(" | ".join(sub_line_cells))
        return "\n".join(out_lines)


    @staticmethod
    def cky_table_to_tsv(cky_table):
        """Convert a CKY table (2D list) to TSV string."""
        tsv_lines = []
        for row in cky_table:
            row_strs = []
            for cell in row:
                if isinstance(cell, (dict, list)):
                    cell_str = json.dumps(cell, ensure_ascii=False)
                else:
                    cell_str = str(cell)
                row_strs.append(cell_str)
            tsv_lines.append("\t".join(row_strs))
        return "\n".join(tsv_lines)

    @staticmethod
    def _normalize_clause_entry(clause_element: Sequence[Any], index: int) -> dict:
        """
        Normalize clause entry into a dict for CKY cell.
        Accepts both newer format:
          [surface, span, tokens, upos_list, xpos_list, token_span]
        and legacy:
          [surface, span, tokens, pos_list, token_span]
        """
        if len(clause_element) < 5:
            raise ValueError(f"clause[{index}] is missing fields: {clause_element}")

        surface = clause_element[0]
        span = clause_element[1]
        tokens = clause_element[2]

        if len(clause_element) >= 6:
            upos_list = clause_element[3]
            xpos_list = clause_element[4]
            token_span = clause_element[5]
        else:
            upos_list = clause_element[3]
            xpos_list = clause_element[3]
            token_span = clause_element[4]

        return {
            "id": index + 1,
            "candidate": surface,
            "span": span,
            "tokens": tokens,
            "upos": upos_list,
            "xpos": xpos_list,
            "token_span": token_span,
        }
