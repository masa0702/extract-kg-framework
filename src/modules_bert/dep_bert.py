# dependency_modification.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification

class DependencyModificationRelationDetector:
    def __init__(self, model_path="./models/output_bert_dependency_bunsetsu_ver3.0/depbert_bunsetsu_20260117_072956/final_model"):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict_relation(self, text_a, text_b):
        """
        係り受け関係 (modification) を判定
        戻り値: (predicted_class, probabilities)
        """
        inputs = self.tokenizer(text_a, text_b, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return predicted_class, probabilities


if __name__ == "__main__":
    DepRel = DependencyModificationRelationDetector()
    # サンプル入力（適宜変更してください）
    # text_a = "メンバーおよび"
    # text_b = "コーチでした。"
    # text_a = "エンジニアと"
    # text_b = "マネージャーが"
    text_a = "エンジニアの"
    text_b = "仕事はプログラムです。"
    # 推論の実行
    pred, probs = DepRel.predict_relation(text_a, text_b)
    
    # 結果の表示
    relation_str = "係り受け関係あり" if pred == 1 else "係り受け関係なし"
    print(f"入力: '{text_a}' と '{text_b}'")
    print(f"予測: {relation_str} (ラベル: {pred})")
    print(f"クラスごとの確率: {probs}")
