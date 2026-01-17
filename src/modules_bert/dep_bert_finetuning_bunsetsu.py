import argparse
import dataclasses
import hashlib
import json
import os
import platform
import random
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)

# =========================
# 便利関数（JSON保存など）
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def dump_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def dump_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# =========================
# CoNLL-U パーサ（最小）
# =========================

@dataclasses.dataclass
class ConlluToken:
    tid: int
    form: str
    head: int
    deprel: str
    misc: Dict[str, str]

@dataclasses.dataclass
class ConlluSentence:
    sent_id: str
    text: str
    tokens: List[ConlluToken]

def parse_misc(misc_str: str) -> Dict[str, str]:
    if not misc_str or misc_str == "_":
        return {}
    items = misc_str.split("|")
    d: Dict[str, str] = {}
    for it in items:
        if "=" in it:
            k, v = it.split("=", 1)
            d[k] = v
        else:
            d[it] = "true"
    return d

def read_conllu(path: str) -> List[ConlluSentence]:
    sentences: List[ConlluSentence] = []
    sent_id = ""
    text = ""
    tokens: List[ConlluToken] = []

    def flush():
        nonlocal sent_id, text, tokens
        if tokens:
            sid = sent_id or f"no_id_{len(sentences)+1}"
            sentences.append(ConlluSentence(sent_id=sid, text=text, tokens=tokens))
        sent_id = ""
        text = ""
        tokens = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                flush()
                continue
            if line.startswith("#"):
                if line.startswith("# sent_id"):
                    sent_id = line.split("=", 1)[1].strip()
                elif line.startswith("# text"):
                    text = line.split("=", 1)[1].strip()
                continue

            cols = line.split("\t")
            if len(cols) != 10:
                continue

            tid_raw = cols[0]
            if "-" in tid_raw or "." in tid_raw:
                continue

            try:
                tid = int(tid_raw)
            except ValueError:
                continue

            form = cols[1]
            head_raw = cols[6]
            try:
                head = int(head_raw)
            except ValueError:
                head = 0

            deprel = cols[7]
            misc = parse_misc(cols[9])

            tokens.append(ConlluToken(tid=tid, form=form, head=head, deprel=deprel, misc=misc))

    flush()
    return sentences

# =========================
# 文節（Bunsetu）復元
# =========================

@dataclasses.dataclass
class Bunsetsu:
    bid: int
    token_ids: List[int]
    head_token_id: int
    text: str

def build_bunsetsu(tokens: List[ConlluToken]) -> Tuple[List[Bunsetsu], Dict[int, int]]:
    if not tokens:
        return [], {}

    bunsetsu_list: List[Bunsetsu] = []
    tok2bun: Dict[int, int] = {}

    cur_token_ids: List[int] = []
    cur_forms: List[str] = []
    cur_spaceafter_no: List[bool] = []
    cur_pos_type: Dict[int, str] = {}

    def flush(bid: int):
        nonlocal cur_token_ids, cur_forms, cur_spaceafter_no, cur_pos_type
        if not cur_token_ids:
            return

        s = ""
        for i, form in enumerate(cur_forms):
            s += form
            if i < len(cur_forms) - 1:
                if not cur_spaceafter_no[i]:
                    s += " "
        s = s.replace("  ", " ").strip()

        head_tid = -1
        for tid in cur_token_ids:
            if cur_pos_type.get(tid) == "SEM_HEAD":
                head_tid = tid
                break
        if head_tid == -1:
            for tid in cur_token_ids:
                if "HEAD" in cur_pos_type.get(tid, ""):
                    head_tid = tid
                    break
        if head_tid == -1:
            head_tid = cur_token_ids[-1]

        bunsetsu_list.append(Bunsetsu(bid=bid, token_ids=list(cur_token_ids), head_token_id=head_tid, text=s))
        for tid in cur_token_ids:
            tok2bun[tid] = bid

        cur_token_ids = []
        cur_forms = []
        cur_spaceafter_no = []
        cur_pos_type = {}

    bid = -1
    for idx, t in enumerate(tokens):
        bi = t.misc.get("BunsetuBILabel", None)
        pos_type = t.misc.get("BunsetuPositionType", "")
        space_after = t.misc.get("SpaceAfter", "")
        spaceafter_no = (space_after == "No")

        start_new = (idx == 0) or (bi == "B") or (bi is None)
        if start_new:
            bid += 1
            flush(bid - 1)

        cur_token_ids.append(t.tid)
        cur_forms.append(t.form)
        cur_spaceafter_no.append(spaceafter_no)
        if pos_type:
            cur_pos_type[t.tid] = pos_type

    flush(bid)
    return bunsetsu_list, tok2bun

# =========================
# 文節間依存の正例生成
# =========================

def extract_positive_pairs(
    tokens: List[ConlluToken],
    bunsetsu_list: List[Bunsetsu],
    tok2bun: Dict[int, int],
    dep_mode: str = "head",
) -> List[Tuple[int, int]]:
    tid2tok = {t.tid: t for t in tokens}
    edges = set()

    if dep_mode not in ("head", "any"):
        raise ValueError(f"dep_mode must be 'head' or 'any', got: {dep_mode}")

    if dep_mode == "head":
        for b in bunsetsu_list:
            tok = tid2tok.get(b.head_token_id)
            if tok is None or tok.head == 0:
                continue
            if tok.head not in tok2bun:
                continue
            src = b.bid
            dst = tok2bun[tok.head]
            if src != dst:
                a, c = (src, dst) if src < dst else (dst, src)
                edges.add((a, c))
    else:
        for t in tokens:
            if t.head == 0:
                continue
            if t.tid not in tok2bun or t.head not in tok2bun:
                continue
            src = tok2bun[t.tid]
            dst = tok2bun[t.head]
            if src != dst:
                a, c = (src, dst) if src < dst else (dst, src)
                edges.add((a, c))

    return sorted(list(edges))

def make_examples_for_sentence(
    sent: ConlluSentence,
    seed: int,
    dep_mode: str = "head",
    min_neg_when_no_pos: int = 0,
) -> List[Dict[str, Any]]:
    bunsetsu_list, tok2bun = build_bunsetsu(sent.tokens)
    nb = len(bunsetsu_list)
    if nb < 2:
        return []

    pos_undirected = extract_positive_pairs(sent.tokens, bunsetsu_list, tok2bun, dep_mode=dep_mode)

    # 無向 → 双方向（あなたの指定）
    pos_directed = set()
    for a, b in pos_undirected:
        if a == b:
            continue
        pos_directed.add((a, b))
        pos_directed.add((b, a))

    num_pos = len(pos_directed)
    if num_pos == 0 and min_neg_when_no_pos <= 0:
        return []

    all_pairs = [(i, j) for i in range(nb) for j in range(nb) if i != j]
    neg_candidates = [p for p in all_pairs if p not in pos_directed]

    rng = random.Random(seed + hash(sent.sent_id) % (10**9))
    if num_pos == 0:
        num_neg = min(min_neg_when_no_pos, len(neg_candidates))
    else:
        num_neg = min(num_pos, len(neg_candidates))

    neg_pairs = rng.sample(neg_candidates, k=num_neg) if num_neg > 0 else []

    bid2text = {b.bid: b.text for b in bunsetsu_list}

    examples: List[Dict[str, Any]] = []
    for a, b in sorted(pos_directed):
        examples.append({
            "sent_id": sent.sent_id,
            "text": sent.text,
            "bun_a_id": a,
            "bun_b_id": b,
            "text_a": bid2text[a],
            "text_b": bid2text[b],
            "label": 1,
        })
    for a, b in neg_pairs:
        examples.append({
            "sent_id": sent.sent_id,
            "text": sent.text,
            "bun_a_id": a,
            "bun_b_id": b,
            "text_a": bid2text[a],
            "text_b": bid2text[b],
            "label": 0,
        })

    return examples

def build_dataset_from_conllu(
    conllu_path: str,
    seed: int,
    dep_mode: str = "head",
    min_neg_when_no_pos: int = 0,
) -> Tuple[Dataset, Dict[str, Any], List[Dict[str, Any]]]:
    sents = read_conllu(conllu_path)

    all_examples: List[Dict[str, Any]] = []
    sent_kept = 0
    sent_skipped = 0
    bun_counts = []
    pos_counts = []
    label_counter = Counter()

    for s in tqdm(sents, desc=f"Build examples: {os.path.basename(conllu_path)}"):
        bunsetsu_list, _ = build_bunsetsu(s.tokens)
        bun_counts.append(len(bunsetsu_list))

        exs = make_examples_for_sentence(
            s, seed=seed, dep_mode=dep_mode, min_neg_when_no_pos=min_neg_when_no_pos
        )
        if not exs:
            sent_skipped += 1
            continue

        sent_kept += 1
        all_examples.extend(exs)

        pc = sum(1 for e in exs if e["label"] == 1)
        pos_counts.append(pc)
        for e in exs:
            label_counter[e["label"]] += 1

    stats = {
        "conllu_path": conllu_path,
        "conllu_sha256": sha256_file(conllu_path),
        "num_sentences_total": len(sents),
        "num_sentences_used": sent_kept,
        "num_sentences_skipped": sent_skipped,
        "bunsetsu_count": {
            "min": int(min(bun_counts)) if bun_counts else 0,
            "max": int(max(bun_counts)) if bun_counts else 0,
            "avg": float(np.mean(bun_counts)) if bun_counts else 0.0,
        },
        "pos_directed_per_sentence": {
            "min": int(min(pos_counts)) if pos_counts else 0,
            "max": int(max(pos_counts)) if pos_counts else 0,
            "avg": float(np.mean(pos_counts)) if pos_counts else 0.0,
        },
        "label_counts": {str(k): int(v) for k, v in label_counter.items()},
        "dep_mode": dep_mode,
        "min_neg_when_no_pos": min_neg_when_no_pos,
    }

    ds = Dataset.from_list(all_examples) if all_examples else Dataset.from_list([])
    return ds, stats, all_examples

# =========================
# メトリクス
# =========================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# =========================
# TrainingArguments 互換（eval_strategy/evaluation_strategy）
# =========================

def make_training_args_compat(**kwargs) -> TrainingArguments:
    try:
        return TrainingArguments(**kwargs)
    except TypeError as e:
        if "eval_strategy" in str(e) and "eval_strategy" in kwargs:
            kwargs2 = dict(kwargs)
            kwargs2["evaluation_strategy"] = kwargs2.pop("eval_strategy")
            return TrainingArguments(**kwargs2)
        raise

# =========================
# 再現性設定・環境ログ
# =========================

def set_reproducibility(seed: int, deterministic: bool = True) -> None:
    """
    transformers.set_seed のシグネチャは版により異なる。
    古い版(例: v4.22.0)では set_seed(seed) のみなので後方互換にする。:contentReference[oaicite:2]{index=2}

    また、CUDA>=10.2で決定論を強める場合、PyTorchは CUBLAS_WORKSPACE_CONFIG を要求する。:contentReference[oaicite:3]{index=3}
    """
    # cuBLAS の再現性向上（CUDA>=10.2環境で deterministic を使うなら推奨）
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # transformers の set_seed は古い版だと deterministic 引数が無い
    try:
        set_seed(seed, deterministic=deterministic)
    except TypeError:
        set_seed(seed)

    # 念のため明示（set_seed が内部でやるが、環境差対策として）
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # torch.use_deterministic_algorithms の有無/引数差を吸収
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        except AttributeError:
            pass

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def collect_env_info() -> Dict[str, Any]:
    info = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", None),
    }
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import datasets
        info["datasets"] = datasets.__version__
    except Exception:
        pass
    return info

# =========================
# メイン
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="./learn_data/ja_gsd-ud-train.conllu")
    parser.add_argument("--dev_file", type=str, default="./learn_data/ja_gsd-ud-dev.conllu")
    parser.add_argument("--test_file", type=str, default="./learn_data/ja_gsd-ud-test.conllu")

    parser.add_argument("--model_name", type=str, default="tohoku-nlp/bert-base-japanese-v3")
    parser.add_argument("--model_revision", type=str, default=None)

    parser.add_argument("--save_dir", type=str, default="./models/output_bert_dependency_bunsetsu_ver3.0")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dep_mode", type=str, default="head", choices=["head", "any"])
    parser.add_argument("--min_neg_when_no_pos", type=int, default=0)

    parser.add_argument("--max_length", type=int, default=128)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_bs", type=int, default=32)
    parser.add_argument("--eval_bs", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    args = parser.parse_args()

    run_id = f"depbert_bunsetsu_{now_str()}"
    out_dir = os.path.join(args.save_dir, run_id)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "artifacts"))
    ensure_dir(os.path.join(out_dir, "plots"))

    # 再現性
    set_reproducibility(args.seed, deterministic=True)

    # 環境ログ
    env_info = collect_env_info()
    dump_json(os.path.join(out_dir, "env.json"), env_info)
    dump_json(os.path.join(out_dir, "args.json"), vars(args))

    for p in [args.train_file, args.dev_file, args.test_file]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    print("=== Build datasets from CoNLL-U (bunsetsu-based) ===")
    train_ds_raw, train_stats, train_examples = build_dataset_from_conllu(
        args.train_file, seed=args.seed, dep_mode=args.dep_mode, min_neg_when_no_pos=args.min_neg_when_no_pos
    )
    dev_ds_raw, dev_stats, dev_examples = build_dataset_from_conllu(
        args.dev_file, seed=args.seed, dep_mode=args.dep_mode, min_neg_when_no_pos=args.min_neg_when_no_pos
    )
    test_ds_raw, test_stats, test_examples = build_dataset_from_conllu(
        args.test_file, seed=args.seed, dep_mode=args.dep_mode, min_neg_when_no_pos=args.min_neg_when_no_pos
    )

    dump_json(os.path.join(out_dir, "dataset_stats_train.json"), train_stats)
    dump_json(os.path.join(out_dir, "dataset_stats_dev.json"), dev_stats)
    dump_json(os.path.join(out_dir, "dataset_stats_test.json"), test_stats)

    sample_n = 2000
    dump_jsonl(os.path.join(out_dir, "train_examples_sample.jsonl"), train_examples[:sample_n])
    dump_jsonl(os.path.join(out_dir, "dev_examples_sample.jsonl"), dev_examples[:sample_n])
    dump_jsonl(os.path.join(out_dir, "test_examples_sample.jsonl"), test_examples[:sample_n])

    print(f"Train examples: {len(train_ds_raw)}, Dev: {len(dev_ds_raw)}, Test: {len(test_ds_raw)}")

    print("=== Load model/tokenizer ===")
    tok_kwargs = {}
    model_kwargs = {}
    if args.model_revision:
        tok_kwargs["revision"] = args.model_revision
        model_kwargs["revision"] = args.model_revision

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tok_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2, **model_kwargs)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text_a"],
            batch["text_b"],
            truncation=True,
            max_length=args.max_length,
        )

    remove_cols = [c for c in train_ds_raw.column_names if c != "label"]
    train_ds = train_ds_raw.map(tokenize_batch, batched=True, remove_columns=remove_cols)
    dev_ds = dev_ds_raw.map(tokenize_batch, batched=True, remove_columns=remove_cols)
    test_ds = test_ds_raw.map(tokenize_batch, batched=True, remove_columns=remove_cols)

    cols = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in train_ds.column_names:
        cols.insert(2, "token_type_ids")

    train_ds.set_format(type="torch", columns=cols)
    dev_ds.set_format(type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=cols)

    data_collator = DataCollatorWithPadding(tokenizer)

    print("=== Training config ===")
    training_args = make_training_args_compat(
        output_dir=os.path.join(out_dir, "checkpoints"),
        run_name=run_id,

        eval_strategy="epoch",
        save_strategy="epoch",

        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,

        logging_dir=os.path.join(out_dir, "logs"),
        logging_strategy="steps",
        logging_steps=50,

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        gradient_accumulation_steps=args.grad_accum,

        fp16=args.fp16,
        bf16=args.bf16,

        seed=args.seed,
        data_seed=args.seed,

        save_total_limit=2,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("=== Train ===")
    train_result = trainer.train()

    final_dir = os.path.join(out_dir, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    dump_json(os.path.join(out_dir, "train_result.json"), train_result.metrics)
    dump_json(os.path.join(out_dir, "trainer_log_history.json"), trainer.state.log_history)

    log_history = trainer.state.log_history
    train_loss = [(e["step"], e["loss"]) for e in log_history if "loss" in e and "eval_loss" not in e]
    if train_loss:
        steps, losses = zip(*train_loss)
        plt.figure()
        plt.plot(steps, losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "plots", "training_loss_curve.png"))
        plt.close()

    dev_f1 = [(e["step"], e["eval_f1"]) for e in log_history if "eval_f1" in e]
    if dev_f1:
        steps, f1s = zip(*dev_f1)
        plt.figure()
        plt.plot(steps, f1s)
        plt.xlabel("Step")
        plt.ylabel("F1")
        plt.title("Validation F1 Curve")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "plots", "validation_f1_curve.png"))
        plt.close()

    print("=== Evaluate on test ===")
    test_metrics = trainer.evaluate(test_ds)
    dump_json(os.path.join(out_dir, "test_metrics.json"), test_metrics)
    print(test_metrics)

    print("=== Confusion matrix & error analysis ===")
    preds_out = trainer.predict(test_ds)
    logits = preds_out.predictions
    y_true = preds_out.label_ids
    y_pred = np.argmax(logits, axis=-1)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    dump_json(os.path.join(out_dir, "confusion_matrix.json"), {"labels": [0, 1], "matrix": cm})
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    errors = []
    if len(test_examples) == len(y_true):
        for i in range(len(y_true)):
            if int(y_true[i]) != int(y_pred[i]):
                e = test_examples[i]
                errors.append({
                    "sent_id": e["sent_id"],
                    "label_true": int(y_true[i]),
                    "label_pred": int(y_pred[i]),
                    "text_a": e["text_a"],
                    "text_b": e["text_b"],
                    "sentence": e.get("text", ""),
                    "bun_a_id": e.get("bun_a_id"),
                    "bun_b_id": e.get("bun_b_id"),
                })
    dump_jsonl(os.path.join(out_dir, "test_errors_top200.jsonl"), errors[:200])

    print(f"Saved outputs to: {out_dir}")

if __name__ == "__main__":
    main()
