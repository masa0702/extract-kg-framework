import csv
import json
import os
from functools import lru_cache

class MyUtility:
    @staticmethod
    @lru_cache(maxsize=128)
    def load_json_from_file(file_path):
        """
        file_pathで指定したファイルからjsonデータを読み込む

        Args:
            file_path (str): ファイルパス
        
        Returns:
            dict: jsonデータ
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    
    @staticmethod
    @lru_cache(maxsize=128)
    def check_comma(text):
        """
        文字列中に「、」があるかどうかを確認する関数。

        Parameters:
        text (str): 確認する文字列。

        Returns:
        bool: 文字列中に「、」があればTrue、なければFalse。
        """
        if text is None:
            return False
        return "、" in text

    @staticmethod
    @lru_cache(maxsize=128)
    def check_period(text):
        """
        文字列中に「。」があるかどうかを確認する関数。

        Parameters:
        text (str): 確認する文字列。

        Returns:
        bool: 文字列中に「。」があればTrue、なければFalse。
        """
        if text is None:
            return False
        return "。" in text
    
    
    @staticmethod
    def print_limited_data(data, limit=5):
        """
        データの数を制限してprintする関数

        Parameters:
            data (list or dict): printするデータ。
            limit (int, optional): printするデータの最大数。デフォルトは5。

        Returns:
            None
        """
        if isinstance(data, list):
            data = data[:limit]
        elif isinstance(data, dict):
            data = dict(list(data.items())[:limit])
        print(data)


    @staticmethod
    def save_to_csv(data, file_name, field_names=None):
        """
        CSVファイルにデータを保存する関数

        Parameters:
            data (list of dict): 保存するデータ。各要素が辞書であり、各辞書が1行のデータを表す。
            filename (str): 保存するファイルのパス。
            fieldnames (list of str, optional): CSVファイルのヘッダーに使用するフィールド名のリスト。

        Returns:
            None
        """
        if not field_names:
            field_names = data[0].keys() if data else []

        with open(file_name, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(data)


    @staticmethod
    def graph_convert_tsv(graph):
        """
        グラフをtsv変換する関数(変換後にspread sheetへコピペ)

        Args:
            graph (dict): グラフ構造の辞書データ
            
        Returns:
            None
        """
        # TSV形式に変換して表示
        if "graph_matrix" in graph[0]:
            for row in graph[0]["graph_matrix"]:
                # リスト内の各要素を文字列に変換し、タブ区切りで結合
                row_str = '\t'.join('' if cell in [0, 1] else str(cell) for cell in row)
                print(row_str)
        else:
            for row in graph:
                # リスト内の各要素を文字列に変換し、タブ区切りで結合
                row_str = '\t'.join('' if cell in [0, 1] else str(cell) for cell in row)
                print(row_str)
                
        return None
    
    @staticmethod
    def display_specific_graph(clause_graphs, target_id):
        """
        指定されたIDに基づいて文節グラフを表示する関数。

        Args:
            clause_graphs (list): 複数の文節グラフを含むリスト。各グラフは辞書形式で、「graph_id」および「graph_matrix」キーを持つ。
            target_id (int): 表示したい文節グラフのID。

        Returns:
            None
        """
        for graph_data in clause_graphs:
            if graph_data["graph_id"] == target_id:
                graph_id = graph_data["graph_id"]
                clause_graph = graph_data["graph_matrix"]
                print(f"Graph ID: {graph_id}")
                # display_graph_as_tsv
                MyUtility.graph_convert_tsv(clause_graph)
                break

    
    @staticmethod
    def has_string(lst):
        """
        listの中に文字列があるかどうかを判定する関

        Args:
            lst (list): 判定するリスト

        Returns:
            bool: listの中に文字列があればTrue、なければFalse
        """
        for item in lst:
            if isinstance(item, str):
                return True
        return False
    
    @staticmethod
    def ensure_directory_exists(directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory created: {directory_path}")
        else:
            print(f"Directory already exists: {directory_path}")
            
    
    @staticmethod
    def delete_space(text):
        """
        文字列の前後にある空白を削除する関数

        Args:
            text (str): 空白を削除する文字列

        Returns:
            str: 空白を削除した文字列
        """
        return text.strip()
    
    
    @staticmethod
    def save_json_from_file(data, output_filename):
        """
        JSONに変換可能なPythonオブジェクトを、指定したファイル名にJSON形式で保存します。
        
        Parameters:
            data (dict/list): 保存するデータ（JSONシリアライズ可能なオブジェクト）
            output_filename (str): 出力先のファイル名
        """
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    
    # @staticmethod
    # def load_csv(file_name):
    # """
    # 指定されたCSVファイルを読み込んでDataFrameとして返す関数。
    
    # Parameters:
    # - file_name (str): 読み込むCSVファイルの名前

    # Returns:
    # - pandas.DataFrame: 読み込んだデータを含むDataFrame
    # """
    #     try:
    #         data = pd.read_csv(file_name)
    #         return data
    #     except Exception as e:
    #         print(f"Error reading {file_name}: {e}")
    #         return None
