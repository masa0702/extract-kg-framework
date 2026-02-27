import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, List


ONTOLOGY_FILE_RE = re.compile(r"^(\d+)_([a-zA-Z0-9]+)_ontology_trans_ja\.json$")
CSV_COLUMNS = [
    "ontology",
    "relation",
    "domain",
    "range",
    "relation_id",
    "domain_id",
    "range_id",
]
CONCEPT_COLUMNS = ["concept", "concept_id"]


def pick_label(item: Dict[str, Any]) -> str:
    return str(
        item.get("label_wiki_ja") or item.get("label_ja") or item.get("label") or ""
    ).strip()


def file_sort_key(path: Path) -> tuple:
    m = ONTOLOGY_FILE_RE.match(path.name)
    if not m:
        return (10**9, path.name)
    return (int(m.group(1)), path.name)


def build_rows(ontology_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    files = sorted(ontology_dir.glob("*_ontology_trans_ja.json"), key=file_sort_key)

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        ontology_id = str(data.get("id") or file_path.stem).strip()
        concepts = data.get("concepts", [])
        relations = data.get("relations", [])

        concept_label_by_qid: Dict[str, str] = {}
        for c in concepts:
            qid = str(c.get("qid") or "").strip()
            if qid:
                concept_label_by_qid[qid] = pick_label(c)

        for r in relations:
            relation_id = str(r.get("pid") or "").strip()
            relation_label = pick_label(r)

            domain_id = str(r.get("domain") or "").strip()
            range_id = str(r.get("range") or "").strip()

            domain_label = concept_label_by_qid.get(domain_id, "")
            range_label = concept_label_by_qid.get(range_id, "")

            rows.append(
                {
                    "ontology": ontology_id,
                    "relation": relation_label,
                    "domain": domain_label,
                    "range": range_label,
                    "relation_id": relation_id,
                    "domain_id": domain_id,
                    "range_id": range_id,
                }
            )

    return rows


def write_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_unique_concepts(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    unique = set()
    for row in rows:
        for concept_col, concept_id_col in [("domain", "domain_id"), ("range", "range_id")]:
            concept = str(row.get(concept_col) or "").strip()
            concept_id = str(row.get(concept_id_col) or "").strip()
            if not concept and not concept_id:
                continue
            if not concept:
                concept = concept_id
            unique.add((concept, concept_id))

    out = [{"concept": c, "concept_id": cid} for c, cid in sorted(unique, key=lambda x: (x[1], x[0]))]
    return out


def write_concept_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CONCEPT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology_dir", default=str(repo_root / "ontology"))
    ap.add_argument(
        "--out_csv",
        default=str(repo_root / "data" / "T2KGB_JA" / "all_ontology_relation.csv"),
    )
    ap.add_argument(
        "--out_concepts_csv",
        default=str(repo_root / "data" / "T2KGB_JA" / "all_ontology_domain_range_concept.csv"),
    )
    args = ap.parse_args()

    ontology_dir = Path(args.ontology_dir)
    out_csv = Path(args.out_csv)
    out_concepts_csv = Path(args.out_concepts_csv)

    if not ontology_dir.is_dir():
        raise FileNotFoundError(f"ontology_dir not found: {ontology_dir}")

    rows = build_rows(ontology_dir)
    write_csv(rows, out_csv)
    concept_rows = build_unique_concepts(rows)
    write_concept_csv(concept_rows, out_concepts_csv)

    print(f"[OK] wrote {len(rows)} rows to {out_csv}")
    print(f"[OK] wrote {len(concept_rows)} rows to {out_concepts_csv}")


if __name__ == "__main__":
    main()
