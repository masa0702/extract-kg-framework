import gzip, pickle
from collections import defaultdict
from tqdm.auto import tqdm

def show_ast_detail_force(ast, max_depth=2, indent=2, cur_depth=0):
    """dir()も含めて属性を表示する（__dict__非対応型もカバー）"""
    sp = " " * (indent * cur_depth)
    cls_name = ast.__class__.__name__
    print(f"{sp}[{cls_name}]")
    for attr in dir(ast):
        if attr.startswith("__") or attr in ("__weakref__",):
            continue
        try:
            v = getattr(ast, attr)
        except Exception as e:
            print(f"{sp}  .{attr} = <Error: {e}>")
            continue
        if callable(v):
            continue
        if isinstance(v, list):
            print(f"{sp}  .{attr} = list (len={len(v)})")
            if v and cur_depth < max_depth:
                for vi in v[:2]:
                    show_ast_detail_force(vi, max_depth, indent, cur_depth+1)
                if len(v) > 2:
                    print(f"{sp}    ... (他 {len(v)-2} 要素)")
        elif hasattr(v, "__class__") and not isinstance(v, (str, int, float, bool, dict)):
            print(f"{sp}  .{attr}:")
            if cur_depth < max_depth:
                show_ast_detail_force(v, max_depth, indent, cur_depth+1)
        else:
            print(f"{sp}  .{attr} = {repr(v)}")

AST_PICKLE = "../data/patterns/patterns_ast.pkl.gz"

with gzip.open(AST_PICKLE, "rb") as fp:
    patterns_ast = pickle.load(fp)

ast_dict = defaultdict(list)
for entry in patterns_ast:
    ast_dict[entry["var_count"]].append(entry["ast"])

for var_count in range(2):
    asts = ast_dict[var_count]
    print(f"\n--- var_count={var_count} : AST数={len(asts)} ---")
    for i, ast in enumerate(asts[:3]):
        print(f"  [AST {i+1}] 構造:")
        show_ast_detail_force(ast, max_depth=2)
        print("  ---")
    if len(asts) > 3:
        print(f"  ... (他 {len(asts)-3} 個)")

print("\n全var_count: ", sorted(ast_dict.keys()))
