import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader # 注意这里用 PyG 的 Loader
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# === 配置 ===
BATCH_SIZE = 16  # 图训练显存占用大，Batch 小一点
LEARNING_RATE = 1e-4
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "microsoft/codebert-base"
DATA_FILE = "mbpp_graphs.pt"

# === 1. 定义 GNN Encoder ===
class GraphMemoryEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=768):
        super().__init__()
        # 简单的 3层 GCN
        # input_dim=3 对应我们在 tracer 里定义的 [Type, Size, Hash]
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 256)
        self.conv3 = GCNConv(256, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        # 1. 节点特征传递 (Message Passing)
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index) # 输出 [Num_Nodes, 768]
        
        # 2. 全局池化 (Readout): 把整张图变成一个向量
        # batch 参数告诉 PyG 哪些节点属于哪张图
        graph_emb = global_mean_pool(x, batch) # 输出 [Batch_Size, 768]
        return graph_emb

# === 2. 定义整体 JEPA 模型 ===
class NeuroSymbolicJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        # A. 代码编码器 (Transformer)
        self.code_encoder = AutoModel.from_pretrained(MODEL_NAME)
        for param in self.code_encoder.parameters():
            param.requires_grad = False
            
        # B. 内存编码器 (GNN)
        self.graph_encoder = GraphMemoryEncoder()
        
        # C. 预测器 (Predictor)
        # 输入: Graph_Emb(768) + Code_Emb(768) = 1536
        self.predictor = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, 768) # 预测 Next Graph Embedding
        )

    def forward(self, prev_graph_batch, code_ids, code_mask):
        # 1. Encode Graph (State)
        state_emb = self.graph_encoder(
            prev_graph_batch.x, 
            prev_graph_batch.edge_index, 
            prev_graph_batch.batch
        )
        
        # 2. Encode Code (Action)
        with torch.no_grad():
            code_out = self.code_encoder(input_ids=code_ids, attention_mask=code_mask)
            code_emb = code_out.last_hidden_state[:, 0, :]
            
        # 3. Predict Next State
        combined = torch.cat([state_emb, code_emb], dim=1)
        pred_next_emb = self.predictor(combined)
        
        return pred_next_emb

    def encode_target_graph(self, next_graph_batch):
        # 用于计算 Loss 的真实目标
        return self.graph_encoder(
            next_graph_batch.x, 
            next_graph_batch.edge_index, 
            next_graph_batch.batch
        )

# === 3. 数据集包装 ===
# PyG 的 Dataset 处理比较特殊，我们手动封装一下
class GraphListDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, tokenizer):
        self.data_list = data_list
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 处理文本
        code_enc = self.tokenizer(
            f"Code: {item['code']}", 
            max_length=128, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        return {
            "prev_graph": item['prev_graph'],
            "next_graph": item['next_graph'],
            "code_ids": code_enc.input_ids.squeeze(0),
            "code_mask": code_enc.attention_mask.squeeze(0)
        }

# PyG 需要自定义 collate_fn 来处理图的 batching
def collate_fn(batch):
    prev_graphs = [item['prev_graph'] for item in batch]
    next_graphs = [item['next_graph'] for item in batch]
    
    return {
        # Batch.from_data_list 会自动把小图拼成大图
        "prev_graph_batch": Batch.from_data_list(prev_graphs),
        "next_graph_batch": Batch.from_data_list(next_graphs),
        "code_ids": torch.stack([item['code_ids'] for item in batch]),
        "code_mask": torch.stack([item['code_mask'] for item in batch])
    }

# === 4. 训练主循环 ===
def main():
    print(f"Training GNN-JEPA on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 加载数据
    raw_data = torch.load(DATA_FILE)
    dataset = GraphListDataset(raw_data, tokenizer)
    # 使用普通的 DataLoader，但在 collate_fn 里做 PyG 的 Batching
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4
    )
    
    model = NeuroSymbolicJEPA().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in loop:
            # 搬运数据
            prev_batch = batch["prev_graph_batch"].to(DEVICE)
            next_batch = batch["next_graph_batch"].to(DEVICE)
            code_ids = batch["code_ids"].to(DEVICE)
            code_mask = batch["code_mask"].to(DEVICE)
            
            # Forward
            pred_emb = model(prev_batch, code_ids, code_mask)
            
            # Target (我们要预测的是 Next Graph 经过 GNN 后的向量)
            # 注意：这里我们同时训练 GNN Encoder 和 Predictor
            target_emb = model.encode_target_graph(next_batch)
            
            # Loss
            loss = criterion(pred_emb, target_emb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"Average Loss: {total_loss / len(dataloader):.6f}")
    
    torch.save(model.state_dict(), "gnn_jepa.pth")
    print("Saved GNN Model!")

if __name__ == "__main__":
    main()