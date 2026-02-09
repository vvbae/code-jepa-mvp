import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import random
from tqdm import tqdm

# === 必须与 train.py 一致的配置 ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "microsoft/codebert-base"

# === 复用模型定义 (通常应该写在单独文件里 import，这里为了方便直接复制) ===
class CodeJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        # 即使是验证，Encoder 也是冻结的
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 加载训练好的 Predictor 权重
        # 注意：这里我们重新定义结构，稍后加载权重
        self.predictor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 768)
        )

    def forward(self, prev_ids, prev_mask):
        with torch.no_grad():
            outputs = self.encoder(input_ids=prev_ids, attention_mask=prev_mask)
            context_embedding = outputs.last_hidden_state[:, 0, :]
        return self.predictor(context_embedding)

    def encode(self, text, tokenizer):
        # 辅助函数：把文本转成 Embedding
        inputs = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        ids = inputs.input_ids.to(DEVICE)
        mask = inputs.attention_mask.to(DEVICE)
        with torch.no_grad():
            outputs = self.encoder(input_ids=ids, attention_mask=mask)
            return outputs.last_hidden_state[:, 0, :]

def main():
    print("Loading Model & Data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 1. 加载模型结构
    model = CodeJEPA().to(DEVICE)
    
    # 2. 加载训练好的权重 (Checkpoint)
    # 注意：我们在 train.py 里只保存了 predictor 的参数，所以只加载 predictor
    model.predictor.load_state_dict(torch.load("code_jepa_predictor.pth", map_location=DEVICE))
    model.eval() # 切换到评估模式
    
    # 3. 加载测试数据
    data = torch.load("mbpp_traces.pt")
    # 随机取 100 条做测试
    test_samples = random.sample(data, 100)
    
    correct_count = 0
    total_count = len(test_samples)
    
    print("-" * 40)
    print(f"Starting Evaluation on {total_count} samples...")
    print("(Task: Pick the real next state vs. a random state)")
    print("-" * 40)

    # 4. 开始测试
    for i, sample in enumerate(tqdm(test_samples)):
        # 准备输入
        prev_text = f"State: {sample['prev_state']}"
        true_next_text = f"State: {sample['next_state']}"
        
        # 随机找一个干扰项 (Distractor)
        # 确保干扰项不是正确答案
        while True:
            random_sample = random.choice(data)
            if random_sample['next_state'] != sample['next_state']:
                distractor_text = f"State: {random_sample['next_state']}"
                break
        
        # 获取 Embedding
        # a. 模型的预测值 (Predicted)
        inputs = tokenizer(prev_text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        pred_emb = model(inputs.input_ids.to(DEVICE), inputs.attention_mask.to(DEVICE))
        
        # b. 真实值的 Embedding (True Target)
        true_emb = model.encode(true_next_text, tokenizer)
        
        # c. 干扰项的 Embedding (Distractor)
        dist_emb = model.encode(distractor_text, tokenizer)
        
        # 5. 计算距离 (L2 Distance / Euclidean)
        dist_to_true = torch.norm(pred_emb - true_emb).item()
        dist_to_fake = torch.norm(pred_emb - dist_emb).item()
        
        # 6. 判定
        if dist_to_true < dist_to_fake:
            correct_count += 1
        
        # 打印前 3 个样本的详细信息，让你直观感受
        if i < 3:
            print(f"\n[Sample {i}]")
            print(f"  Prev State: {sample['prev_state']}")
            print(f"  True Next : {sample['next_state']}")
            print(f"  Dist(Pred, True) = {dist_to_true:.4f}")
            print(f"  Dist(Pred, Fake) = {dist_to_fake:.4f}")
            print(f"  Result: {'✅ Correct' if dist_to_true < dist_to_fake else '❌ Wrong'}")

    accuracy = correct_count / total_count * 100
    print("-" * 40)
    print(f"Final Accuracy: {accuracy:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    main()