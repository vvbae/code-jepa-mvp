import copy
import csv
import json
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# === 1. 配置与路径 ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRAPH_DATA_PATH = "csn_python_graphs.pt"
VOCAB_PATH = "node_type_vocab.json"
MODEL_NAME = "microsoft/codebert-base"

BATCH_SIZE = 32
LR = 1e-4
EMA_DECAY = 0.996
EPOCHS = 10
LOG_EVERY = 20  # 每 20 个 step 记录一次
OUTPUT_DIR = "runs"


# === 2. 日志与绘图工具 ===
def setup_run_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"jepa_csn_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def plot_losses(csv_path, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {"step": [], "total": [], "pred": [], "var": []}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["step"].append(int(row["step"]))
            data["total"].append(float(row["total_loss"]))
            data["pred"].append(float(row["pred_loss"]))
            data["var"].append(float(row["var_loss"]))

    plt.figure(figsize=(10, 6))
    plt.plot(data["step"], data["total"], label="Total Loss")
    plt.plot(data["step"], data["pred"], label="Pred Loss (MSE)", alpha=0.7)
    plt.plot(data["step"], data["var"], label="Var Loss (VICReg)", alpha=0.7)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("JEPA Training Progress")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()


# === 3. 模型定义 (GNN, VICReg, JEPA) ===
from torch_geometric.nn import GCNConv, global_mean_pool


class GNNEncoder(nn.Module):
    def __init__(self, num_types, out_dim=256):
        super().__init__()
        self.embed = nn.Embedding(num_types, 128)
        self.conv1 = GCNConv(128, 256)
        self.conv2 = GCNConv(256, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, data):
        x = self.embed(data.x.squeeze())
        x = F.gelu(self.conv1(x, data.edge_index))
        x = self.conv2(x, data.edge_index)
        x = global_mean_pool(x, data.batch)
        return self.bn(x)


class VICRegLoss(nn.Module):
    def forward(self, x, y):
        inv_loss = F.mse_loss(x, y)
        std_x = torch.sqrt(x.var(dim=0) + 1e-4)
        std_y = torch.sqrt(y.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))
        return inv_loss, var_loss


class CodeJEPA(nn.Module):
    def __init__(self, num_types):
        super().__init__()
        self.code_encoder = AutoModel.from_pretrained(MODEL_NAME)
        for p in self.code_encoder.parameters():
            p.requires_grad = False
        self.code_proj = nn.Linear(768, 256)

        self.target_encoder = GNNEncoder(num_types, 256)
        self.target_ema = copy.deepcopy(self.target_encoder)

        self.predictor = nn.Sequential(
            nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256)
        )

    @torch.no_grad()
    def update_ema(self):
        for online, ema in zip(
            self.target_encoder.parameters(), self.target_ema.parameters()
        ):
            ema.data = ema.data * EMA_DECAY + online.data * (1 - EMA_DECAY)


# === 4. 训练主循环 ===
def train():
    run_dir = setup_run_dir()
    csv_path = os.path.join(run_dir, "train_log.csv")
    print(f"Working Directory: {run_dir}")

    # 数据准备
    dataset = torch.load(GRAPH_DATA_PATH)
    with open(VOCAB_PATH, "r") as f:
        num_types = len(json.load(f))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def collate(batch):
        codes = [d.raw_code for d in batch]
        toks = tokenizer(
            codes, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        from torch_geometric.data import Batch

        return toks.input_ids, toks.attention_mask, Batch.from_data_list(batch)

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate
    )

    model = CodeJEPA(num_types).to(DEVICE)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    criterion = VICRegLoss()

    # CSV 初始化
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(
        ["epoch", "step", "total_loss", "pred_loss", "inv_loss", "var_loss"]
    )

    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for ids, mask, g_data in pbar:
            ids, mask, g_data = ids.to(DEVICE), mask.to(DEVICE), g_data.to(DEVICE)

            # Forward
            with torch.no_grad():
                c_out = model.code_encoder(ids, mask)
                code_cls = c_out.last_hidden_state[:, 0, :]
                target_z = model.target_ema(g_data)

            c_emb = model.code_proj(code_cls)
            pred_z = model.predictor(c_emb)
            online_z = model.target_encoder(g_data)

            # Loss 计算
            pred_loss = F.mse_loss(pred_z, target_z)
            inv_l, var_l = criterion(online_z, target_z)
            total_loss = pred_loss + inv_l + var_l

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            model.update_ema()

            global_step += 1
            if global_step % LOG_EVERY == 0:
                writer.writerow(
                    [
                        epoch,
                        global_step,
                        total_loss.item(),
                        pred_loss.item(),
                        inv_l.item(),
                        var_l.item(),
                    ]
                )
                csv_file.flush()

            pbar.set_postfix(L=f"{total_loss.item():.3f}", V=f"{var_l.item():.2f}")

    csv_file.close()
    torch.save(model.state_dict(), os.path.join(run_dir, "model_final.pth"))
    plot_losses(csv_path, os.path.join(run_dir, "loss_curve.png"))
    print("Training Complete. Plot saved.")


if __name__ == "__main__":
    train()
