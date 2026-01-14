import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

SAVE_DIR = "../models/output_bert_dependency_ver2.0"

# 乱数シードの固定（再現性のため）
set_seed(42)

# -----------------------------------------------------------------------------
# ① UD Japanese-GSDコーパスから各文のトークンと係り受け情報を元に
#     正例（依存関係あり）と負例（依存関係なし）のペアを生成する関数
# -----------------------------------------------------------------------------
def generate_examples(example):
    """
    入力例（文）のtokensとhead情報から
    ・正例：各トークンで head != 0 の場合、(子, head) のペア
    ・負例：文内の任意のペアのうち、正例にならないペアをランダムにサンプリング（正例数と同数）
    を生成する。
    """
    tokens = example["tokens"]
    heads = example["head"]  # ここでは各 head が文字列になっている可能性があります
    pos_pairs = []
    # 正例ペアの生成：各トークンについて head が "0" でなければ、(子, head) のペアとする
    for i, head in enumerate(heads):
        head_val = int(head)  # 文字列から整数に変換
        if head_val != 0 and (head_val - 1) < len(tokens):
            pos_pairs.append((tokens[i], tokens[head_val - 1]))
    
    # すべての候補ペア（順序を区別するので全組み合わせ）を生成
    all_pairs = []
    n = len(tokens)
    for i in range(n):
        for j in range(n):
            if i != j:
                all_pairs.append((tokens[i], tokens[j]))
    
    # 順序に依存しない形で正例集合を構築（headの関係はどちらの順序でも同じとみなす）
    pos_set = {tuple(sorted(pair)) for pair in pos_pairs}
    
    # 負例候補：正例集合に含まれないペア
    candidates = []
    for pair in all_pairs:
        if tuple(sorted(pair)) not in pos_set:
            candidates.append(pair)
    
    # 正例数と同じ数の負例をランダムサンプリング
    num_neg = len(pos_pairs)
    if len(candidates) > num_neg:
        neg_pairs = random.sample(candidates, num_neg)
    else:
        neg_pairs = candidates
    
    # 正例、負例ともに辞書形式にして返す
    examples = []
    for a, b in pos_pairs:
        examples.append({"text_a": a, "text_b": b, "label": 1})
    for a, b in neg_pairs:
        examples.append({"text_a": a, "text_b": b, "label": 0})
    return examples

# -----------------------------------------------------------------------------
# ② データセットのロードと前処理
#     Hugging Face datasetsライブラリを用いて UD Japanese-GSD を取得し、
#     各分割（train, validation, test）ごとに例を生成
# -----------------------------------------------------------------------------
print("UD Japanese-GSDデータセットのロード中…")
raw_datasets = load_dataset("universal_dependencies", "ja_gsd", trust_remote_code=True)

def process_split(split):
    examples = []
    for ex in split:
        ex_pairs = generate_examples(ex)
        examples.extend(ex_pairs)
    return Dataset.from_list(examples)

print("前処理中…")
train_dataset = process_split(raw_datasets["train"])
val_dataset = process_split(raw_datasets["validation"])
test_dataset = process_split(raw_datasets["test"])

print(f"Train例数: {len(train_dataset)}, Validation例数: {len(val_dataset)}, Test例数: {len(test_dataset)}")

# -----------------------------------------------------------------------------
# ③ モデル・トークナイザの初期化
#     モデルは「bert-base-japanese」（ここでは東北大版BERT: cl-tohoku/bert-base-japanese）を使用
# -----------------------------------------------------------------------------
model_name = "tohoku-nlp/bert-base-japanese-v3"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# -----------------------------------------------------------------------------
# ④ データセットのトークナイズ
#     入力は "text_a" と "text_b" の2つの文字列として渡す
# -----------------------------------------------------------------------------
def tokenize_function(example):
    return tokenizer(example["text_a"], example["text_b"], truncation=True)

print("トークナイズ中…")
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 不要なテキスト列を削除し、PyTorch形式に変換
columns = ["input_ids", "attention_mask", "label"]
train_dataset.set_format(type="torch", columns=columns)
val_dataset.set_format(type="torch", columns=columns)
test_dataset.set_format(type="torch", columns=columns)

data_collator = DataCollatorWithPadding(tokenizer)

# -----------------------------------------------------------------------------
# ⑤ 学習設定の定義（TrainingArguments）
#     評価・チェックポイント保存、ログ出力の設定を行い、TensorBoardなどで進捗を確認可能に
# -----------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,  # バッチサイズを小さくする
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="../logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    gradient_accumulation_steps=2,  # 実質バッチサイズを大きくするために利用
    fp16=True  # 半精度学習を有効にする
)

# -----------------------------------------------------------------------------
# ⑥ 評価指標の定義
#     sklearnを用いてAccuracy、Precision、Recall、F1を算出
# -----------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# -----------------------------------------------------------------------------
# ⑦ Trainerの定義
# -----------------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# -----------------------------------------------------------------------------
# ⑧ モデルのファインチューニング
# -----------------------------------------------------------------------------
print("学習開始…")
trainer.train()

# ファインチューニング完了後、最終モデルとトークナイザを保存
trainer.save_model(f"{SAVE_DIR}/final_model")
tokenizer.save_pretrained(f"{SAVE_DIR}/final_model")

# -----------------------------------------------------------------------------
# ⑨ 学習曲線の可視化
#     Trainerのログ情報（trainer.state.log_history）から学習損失の推移をプロットして保存
# -----------------------------------------------------------------------------
log_history = trainer.state.log_history
# 学習中のlossエントリのみ抽出（eval時のlossは除く）
train_loss = [(entry["step"], entry["loss"]) for entry in log_history if "loss" in entry and "eval_loss" not in entry]
if train_loss:
    steps, losses = zip(*train_loss)
    plt.figure()
    plt.plot(steps, losses, label="Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{SAVE_DIR}/training_loss_curve.png")
    plt.show()
else:
    print("学習損失ログが存在しません。")

# -----------------------------------------------------------------------------
# ⑩ テストデータでの評価
# -----------------------------------------------------------------------------
print("テストデータで評価中…")
eval_results = trainer.evaluate(test_dataset)
print("テスト評価結果:")
print(eval_results)

# CUDA_VISIBLE_DEVICES=1,2,3 python dep_bert_finetuning.py
# CUDA_VISIBLE_DEVICES=0,1 python dependency_bert_finetuning.py