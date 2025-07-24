import pandas as pd

# 入力ファイル名
input_csv = '../results/extract_pred_arg_pair/target_datas/test_target_data/test_target_data_extract_po_pair.csv'  # 必要に応じて変更

# データ読み込み
df = pd.read_csv(input_csv)

id2rel_arg = {}
for id_, group in df.groupby('id'):
    rel_set = set(group['rel_ja'])
    arg_set = set(group['arg_ja'])
    id2rel_arg[id_] = {
        'rel_ja': list(rel_set),
        'arg_ja': list(arg_set)
    }

for k, v in id2rel_arg.items():
    print(f"id: {k}")
    print("rel_ja:", v['rel_ja'])
    print("arg_ja:", v['arg_ja'])
    print()