# amod_acl.py
import torch
from transformers import BertTokenizer, BertForMaskedLM

class MaskRelationDetector:
    def __init__(self, model_name="tohoku-nlp/bert-base-japanese-v3", candidate_tokens=None, device=None):
        """
        モデルとトークナイザをロードし、デバイス設定を行います。

        :param model_name: 利用するBERTモデルの名前（デフォルトは "tohoku-nlp/bert-base-japanese-v3"）
        :param candidate_tokens: 項述語関係として判定する候補語リスト（デフォルトは ["を"]）
        :param device: 使用するデバイス。Noneの場合はcudaがあればGPUを使用します。
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.candidate_tokens = candidate_tokens if candidate_tokens is not None else ["な", "の"]
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()


    def predict_relation(self, text_A, text_B, top_k=5):
        """
        非トークナイズの文字列を入力として受け取り、項述語関係もしくは連体修飾があるかどうか（候補語が予測されるか）を判定します。

        :param text_A: 文字列（例："太郎は本を"）※末尾語を[MASK]に置換します。
        :param text_B: 文字列（例："読んでいる"）
        :return: 関係ラベル(項-述語, 連体修飾)
        """
        # MASK位置に応じてBERTに入力
        # ここを追加！ 末尾"および"ならスキップ
        if text_A[-3:] == "および":
            return None, None
        # top-kの助詞候補を取得
        topk_particles = self._get_mask_topk(text_A, text_B, top_k)
        # 優先順位に従って判定
        result = self._determine_relation_label(topk_particles["topk_results"][0])
        return result
        
    
    def _determine_relation_label(self, topk_particles):
        # 優先ルールに基づき判定
        priority_map = [
            ("項-述語", {"を", "に"}),
            ("連体修飾", {"な", "の"}),
        ]
        found = []
        for idx, particle in enumerate(topk_particles):
            for label, candidates in priority_map:
                if particle in candidates:
                    found.append((idx, label, particle))
        if found:
            # 最もtopが早いものを返す
            found.sort()
            return found[0][1], found[0][2]  # (label, particle)
        return None, None  # 判定できない場合
    
    
    def _get_mask_topk(self, text_A, text_B, k=5):
        """
        非トークナイズの文字列を入力として受け取り、[MASK]位置の上位 k 件の予測トークンとその確率、
        さらに入力トークン列と[MASK]の位置を返します。

        :param text_A: 文字列（例："太郎は本を"）※末尾語を[MASK]に置換します。
        :param text_B: 文字列（例："読んでいる"）
        :param k: 上位何件の予測を返すか（デフォルトは5）
        :return: 辞書型 { "input_tokens": [...], "mask_index": int, "topk_results": [(トークン, 確率), ...] }
        """
        tokens_A = self.tokenizer.tokenize(text_A)
        tokens_B = self.tokenizer.tokenize(text_B)
        if not tokens_A:
            raise ValueError("text_Aからトークナイズした結果が空です。")
        
        tokens_A_masked = tokens_A[:-1] + [self.tokenizer.mask_token]
        input_tokens = [self.tokenizer.cls_token] + tokens_A_masked + tokens_B + [self.tokenizer.sep_token]
        token_type_ids = [0] * (len(tokens_A_masked) + 1) + [1] * (len(tokens_B) + 1)
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        
        input_ids_tensor = torch.tensor([input_ids]).to(self.device)
        token_type_ids_tensor = torch.tensor([token_type_ids]).to(self.device)
        attention_mask = torch.ones_like(input_ids_tensor).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids_tensor,
                token_type_ids=token_type_ids_tensor,
                attention_mask=attention_mask
            )
            logits = outputs.logits
        
        try:
            mask_index = input_tokens.index(self.tokenizer.mask_token)
            
        except ValueError:
            raise ValueError("入力トークン列に[MASK]が見つかりません。")
        
        mask_logits = logits[0, mask_index]
        probs = torch.softmax(mask_logits, dim=0)
        topk_probs, topk_indices = torch.topk(probs, k)
        topk_tokens = self.tokenizer.convert_ids_to_tokens(topk_indices.tolist())
        
        topk_results = [(token, prob.item()) for token, prob in zip(topk_tokens, topk_probs)]
        
        return {
            "input_tokens": input_tokens,
            "mask_index": mask_index,
            "topk_results": topk_results
        }

if __name__ == '__main__':
    # モジュールの利用例
    detector = MaskRelationDetector()
    
    # サンプル入力: 非トークナイズの文字列
    text_A = "翌年の"    # 末尾語が[MASK]に置換される
    text_B = "アカデミー短編アニメ映画賞を"
    
    relation = detector.predict_relation(text_A, text_B)
    top_k = detector._get_mask_topk(text_A, text_B)
    print(relation)
    print(top_k)
    




# import torch
# from transformers import BertTokenizer, BertForSequenceClassification

# class AmodAclRelationDetector:
#     def __init__(self, model_path="../output_bert_amod_acl_ver2.0/final_model"):
#         self.tokenizer = BertTokenizer.from_pretrained(model_path)
#         self.model = BertForSequenceClassification.from_pretrained(model_path)
#         self.model.eval()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)

#     def predict_relation(self, token_a, token_b):
#         """
#         連体修飾(amod/acl)を判定
#         戻り値: (predicted_label, probabilities)
#         """
#         # 句読点、英数字などもtoken_a, token_bに入って良い
#         inputs = self.tokenizer(token_a, token_b, truncation=True, padding=True, return_tensors="pt")
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         logits = outputs.logits
#         predicted_label = torch.argmax(logits, dim=-1).item()
#         probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
#         return predicted_label, probabilities


# # 使用例
# if __name__ == "__main__":
#     AmodRel = AmodAclRelationDetector()
#     # 例として「美しい」が「景色」を修飾しているかを判定
#     # token_a = "美しい"
#     # token_b = "景色"
#     token_a = "大学の"
#     token_b = "教授を"
#     label, probs = AmodRel.predict_relation(token_a, token_b)
#     print(f"入力ペア: ({token_a}, {token_b})")
#     print(f"予測ラベル: {label}")
#     print(f"各クラスの確率: {probs}")


# # CUDA_VISIBLE_DEVICES=1,2,3 python rentai_bert.py