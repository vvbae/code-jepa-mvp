import copy
import csv
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

# 关键：导入 PyG 专门的 Loader
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# === 1. 配置与路径 ===
CONFIG = {
    "batch_size": 32,
    "lr": 1e-4,
    "epochs": 10,
    "ema_decay": 0.996,
    "log_every": 10,
    "data_file": os.path.join(os.path.dirname(__file__), "mbpp_graphs.pt"),
    "model_name": "microsoft/codebert-base",
    "output_dir": "runs_jepa_vicreg",
}


def setup_run_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(CONFIG["output_dir"], f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# === 2. VICReg Loss (带分解监控) ===
class VICRegLoss(nn.Module):
    def __init__(self, inv_weight=25.0, var_weight=25.0, cov_weight=1.0):
        super().__init__()
        self.inv_weight = inv_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward(self, x, y):
        # Invariance: MSE
        repr_loss = F.mse_loss(x, y)

        # Variance: Std deviation should be > 1
        std_x = torch.sqrt(x.var(dim=0) + 1e-04)
        std_y = torch.sqrt(y.var(dim=0) + 1e-04)
        var_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        # Covariance: Off-diagonal should be 0
        def off_diagonal_cov(z):
            n, d = z.size()
            if n < 2:
                return torch.tensor(0.0, device=z.device)
            z = z - z.mean(dim=0)
            cov = (z.T @ z) / (n - 1)
            off_diag = cov.pow(2).sum() - cov.diag().pow(2).sum()
            return off_diag / d

        cov_loss = off_diagonal_cov(x) + off_diagonal_cov(y)

        total = self.inv_weight * repr_loss + self.var_weight * var_loss + self.cov_weight * cov_loss
        return total, repr_loss, var_loss, cov_loss


# === 3. 模型组件 (GNN & JEPA) ===
class GraphMemoryEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=768):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, 512)
        self.conv3 = GCNConv(512, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return self.norm(global_mean_pool(x, batch))


class NeuroSymbolicJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.code_encoder = AutoModel.from_pretrained(CONFIG["model_name"])
        for p in self.code_encoder.parameters():
            p.requires_grad = False

        self.online_encoder = GraphMemoryEncoder()
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(nn.Linear(768 * 2, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Linear(1024, 768))

    def forward(self, prev_graph, code_ids, code_mask):
        state_emb = self.online_encoder(prev_graph.x, prev_graph.edge_index, prev_graph.batch)
        with torch.no_grad():
            code_emb = self.code_encoder(code_ids, code_mask).last_hidden_state[:, 0, :]
        return self.predictor(torch.cat([state_emb, code_emb], dim=1))

    @torch.no_grad()
    def encode_target(self, next_graph):
        return self.target_encoder(next_graph.x, next_graph.edge_index, next_graph.batch)

    def ema_update(self):
        tau = CONFIG["ema_decay"]
        for tp, op in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            tp.data.mul_(tau).add_(op.data, alpha=1.0 - tau)


# === 4. 数据处理 ===
class JEPADataSet(torch.utils.data.Dataset):
    def __init__(self, data_list, tokenizer):
        self.data_list = data_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        item = self.data_list[i]
        c = self.tokenizer(item["code"], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        return item["prev_graph"], item["next_graph"], c.input_ids.squeeze(0), c.attention_mask.squeeze(0)


def collate_fn(batch):
    p_gs, n_gs, c_ids, c_masks = zip(*batch)
    return Batch.from_data_list(p_gs), Batch.from_data_list(n_gs), torch.stack(c_ids), torch.stack(c_masks)


# === 5. 绘图辅助 ===
def save_plots(csv_path, out_path):
    import matplotlib.pyplot as plt
    import pandas as pd

    try:
        df = pd.read_csv(csv_path)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].plot(df["step"], df["loss"], label="Total Loss")
        axes[0].plot(df["step"], df["inv_loss"], label="Invariance (MSE)")
        axes[0].set_title("Loss Curves")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(df["step"], df["var_loss"], label="Variance Loss", color="orange")
        axes[1].plot(df["step"], df["norm"], label="Pred Norm", color="green")
        axes[1].set_title("Collapse Monitoring")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")


# === 6. 主训练循环 ===
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = setup_run_dir()
    csv_path = os.path.join(run_dir, "train_log.csv")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    raw_data = torch.load(CONFIG["data_file"], weights_only=False)
    loader = DataLoader(JEPADataSet(raw_data, tokenizer), batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)

    model = NeuroSymbolicJEPA().to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["lr"])
    criterion = VICRegLoss().to(device)

    f = open(csv_path, "w", newline="")
    try:
        writer = csv.writer(f)
        writer.writerow(["epoch", "step", "loss", "inv_loss", "var_loss", "cov_loss", "norm"])

        global_step = 0
        for epoch in range(CONFIG["epochs"]):
            model.train()
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
            for p_g, n_g, c_ids, c_mask in pbar:
                p_g, n_g, c_ids, c_mask = [x.to(device) for x in [p_g, n_g, c_ids, c_mask]]

                pred_z = model(p_g, c_ids, c_mask)
                target_z = model.encode_target(n_g)

                loss, inv_l, var_l, cov_l = criterion(pred_z, target_z)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                model.ema_update()

                global_step += 1
                if global_step % CONFIG["log_every"] == 0:
                    norm = pred_z.norm(dim=1).mean().item()
                    writer.writerow(
                        [
                            epoch + 1,
                            global_step,
                            f"{loss.item():.6f}",
                            f"{inv_l.item():.6f}",
                            f"{var_l.item():.6f}",
                            f"{cov_l.item():.6f}",
                            f"{norm:.4f}",
                        ]
                    )
                    f.flush()
                    pbar.set_postfix(loss=f"{loss.item():.3f}", var=f"{var_l.item():.3f}", norm=f"{norm:.2f}")

            save_plots(csv_path, os.path.join(run_dir, "monitor.png"))
            torch.save(model.state_dict(), os.path.join(run_dir, "last_model.pth"))
    finally:
        f.close()

    print(f"Training Complete. Results in {run_dir}")


if __name__ == "__main__":
    main()
