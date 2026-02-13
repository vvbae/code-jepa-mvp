import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

# === 配置 ===
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "microsoft/codebert-base"

# === 1. 定义数据集 ===
class CodeTraceDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):
        self.data = torch.load(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 输入：上一刻的状态 (Context)
        # 目标：下一刻的状态 (Target)
        # 注意：这里我们简化了任务，直接预测 Next State 的 Embedding
        # 实际上应该输入 (Code Line + Prev State)，但为了 MVP，先只看状态演变
        
        prev_text = f"State: {item['prev_state']}"
        next_text = f"State: {item['next_state']}"
        
        # Tokenize
        prev_enc = self.tokenizer(prev_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        next_enc = self.tokenizer(next_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        
        return {
            "prev_input_ids": prev_enc.input_ids.squeeze(0),
            "prev_attention_mask": prev_enc.attention_mask.squeeze(0),
            "next_input_ids": next_enc.input_ids.squeeze(0),
            "next_attention_mask": next_enc.attention_mask.squeeze(0)
        }

# === 2. 定义模型 ===
class CodeJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        # 冻结的教师模型 (Encoder)
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        for param in self.encoder.parameters():
            param.requires_grad = False  # 冻结！
            
        # 可训练的学生模型 (Predictor)
        # 这里用一个简单的 MLP：把 768 维 -> 768 维
        self.predictor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 768)
        )

    def forward(self, prev_ids, prev_mask):
        # 1. 获取 Context Embedding (Teacher)
        with torch.no_grad():
            outputs = self.encoder(input_ids=prev_ids, attention_mask=prev_mask)
            # 取 [CLS] token 作为句子的整体表示
            context_embedding = outputs.last_hidden_state[:, 0, :]
            
        # 2. 预测 Next State Embedding (Student)
        predicted_embedding = self.predictor(context_embedding)
        return predicted_embedding

    def get_target_embedding(self, next_ids, next_mask):
        # 获取真实的目标 Embedding (Teacher)
        with torch.no_grad():
            outputs = self.encoder(input_ids=next_ids, attention_mask=next_mask)
            return outputs.last_hidden_state[:, 0, :]

# === 3. 训练循环 ===
def main():
    print(f"Using device: {DEVICE}")
    
    # 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 加载数据
    dataset = CodeTraceDataset("mbpp_traces.pt", tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 初始化模型
    model = CodeJEPA().to(DEVICE)
    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # 简单起见，用均方误差 (L2 Loss)
    
    print("Start Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in loop:
            # 搬运数据到 GPU
            prev_ids = batch["prev_input_ids"].to(DEVICE)
            prev_mask = batch["prev_attention_mask"].to(DEVICE)
            next_ids = batch["next_input_ids"].to(DEVICE)
            next_mask = batch["next_attention_mask"].to(DEVICE)
            
            # 前向传播
            # Student 预测
            pred_emb = model(prev_ids, prev_mask)
            # Teacher 提供真实目标
            target_emb = model.get_target_embedding(next_ids, next_mask)
            
            # 计算 Loss
            loss = criterion(pred_emb, target_emb)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
        
    # 保存模型
    torch.save(model.predictor.state_dict(), "code_jepa_predictor.pth")
    print("Model saved!")

if __name__ == "__main__":
    main()