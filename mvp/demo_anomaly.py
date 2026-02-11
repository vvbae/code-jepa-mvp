import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# === 配置 ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "microsoft/codebert-base"

# === 模型结构 (必须与训练时一致) ===
class CodeJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
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
        inputs = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.encoder(input_ids=inputs.input_ids.to(DEVICE), attention_mask=inputs.attention_mask.to(DEVICE))
            return outputs.last_hidden_state[:, 0, :]

def main():
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = CodeJEPA().to(DEVICE)
    model.predictor.load_state_dict(torch.load("code_jepa_predictor.pth", map_location=DEVICE))
    model.eval()

    print("-" * 50)
    print("🧪 实验：模型能否识别'逻辑错误'？")
    print("-" * 50)

    # === 构造一个简单的案例 ===
    # 场景：执行加法
    prev_state_str = "State: x: 1"
    code_action    = "x = x + 1"  # 我们假设隐含在 Context 里，或者通过 Encoder 知道这是加法上下文
    
    # 这里的 Trick 是：我们的模型是 (State -> State) 的演化
    # 训练数据的上下文里其实隐含了代码逻辑。
    # 我们来看看模型是否学到了 "数值通常是平滑变化" 或者 "简单加法" 的规律
    
    # 1. 模型的预测 (Prediction)
    # 输入：当前状态 x=1
    print(f"输入状态: {prev_state_str}")
    inputs = tokenizer(prev_state_str, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    pred_emb = model(inputs.input_ids.to(DEVICE), inputs.attention_mask.to(DEVICE))

    # 2. 候选结果 (Candidates)
    candidates = [
        ("x: 2",   "✅ 正确 (x=2)"),
        ("x: 100", "❌ 错误 (x=100)"),
        ("x: -5",  "❌ 错误 (x=-5)"),
        ("y: 99",  "❌ 离谱 (变量名都变了)")
    ]

    print(f"\n模型预测中...\n")
    
    for state_val, label in candidates:
        target_str = f"State: {state_val}"
        # 编码候选状态
        target_emb = model.encode(target_str, tokenizer)
        
        # 计算距离 (L2 Distance)
        dist = torch.norm(pred_emb - target_emb).item()
        
        # 归一化一下距离方便看 (可选)
        print(f"候选: {state_val.ljust(10)} | 距离: {dist:.4f} | {label}")

if __name__ == "__main__":
    main()