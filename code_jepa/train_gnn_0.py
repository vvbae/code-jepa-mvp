import csv
import os
import copy
from datetime import datetime

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# === Config ===
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "microsoft/codebert-base"
DATA_FILE = "mbpp_graphs.pt"
LOG_EVERY = 10  # log to CSV every N steps
EMA_DECAY = 0.996  # EMA decay for target encoder
GRAD_CLIP = 1.0  # max gradient norm
OUTPUT_DIR = "runs"


# === 1. GNN Encoder ===
class GraphMemoryEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=768):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 256)
        self.conv3 = GCNConv(256, hidden_dim)
        self.relu = nn.ReLU()
        # LayerNorm on output to stabilize embeddings
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, batch):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        graph_emb = global_mean_pool(x, batch)
        graph_emb = self.norm(graph_emb)
        return graph_emb


# === 2. JEPA Model (with EMA target encoder) ===
class NeuroSymbolicJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        # A. Code encoder (frozen Transformer)
        self.code_encoder = AutoModel.from_pretrained(MODEL_NAME)
        for param in self.code_encoder.parameters():
            param.requires_grad = False

        # B. Online graph encoder (trained via gradient)
        self.graph_encoder = GraphMemoryEncoder()

        # C. Target graph encoder (updated via EMA, no gradients)
        self.target_encoder = copy.deepcopy(self.graph_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # D. Predictor
        self.predictor = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, 768),
        )

    @torch.no_grad()
    def update_target_encoder(self, decay=EMA_DECAY):
        """EMA update: target_params = decay * target_params + (1 - decay) * online_params"""
        for tp, op in zip(self.target_encoder.parameters(), self.graph_encoder.parameters()):
            tp.data.mul_(decay).add_(op.data, alpha=1.0 - decay)

    def forward(self, prev_graph_batch, code_ids, code_mask):
        # 1. Encode prev graph (online encoder)
        state_emb = self.graph_encoder(
            prev_graph_batch.x,
            prev_graph_batch.edge_index,
            prev_graph_batch.batch,
        )

        # 2. Encode code (frozen)
        with torch.no_grad():
            code_out = self.code_encoder(input_ids=code_ids, attention_mask=code_mask)
            code_emb = code_out.last_hidden_state[:, 0, :]

        # 3. Predict next state
        combined = torch.cat([state_emb, code_emb], dim=1)
        pred_next_emb = self.predictor(combined)
        return pred_next_emb

    @torch.no_grad()
    def encode_target_graph(self, next_graph_batch):
        """Encode target with EMA encoder — fully detached, no gradients."""
        return self.target_encoder(
            next_graph_batch.x,
            next_graph_batch.edge_index,
            next_graph_batch.batch,
        )


# === 3. Dataset ===
class GraphListDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, tokenizer):
        self.data_list = data_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        code_enc = self.tokenizer(
            f"Code: {item['code']}",
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "prev_graph": item["prev_graph"],
            "next_graph": item["next_graph"],
            "code_ids": code_enc.input_ids.squeeze(0),
            "code_mask": code_enc.attention_mask.squeeze(0),
        }


def collate_fn(batch):
    prev_graphs = [item["prev_graph"] for item in batch]
    next_graphs = [item["next_graph"] for item in batch]
    return {
        "prev_graph_batch": Batch.from_data_list(prev_graphs),
        "next_graph_batch": Batch.from_data_list(next_graphs),
        "code_ids": torch.stack([item["code_ids"] for item in batch]),
        "code_mask": torch.stack([item["code_mask"] for item in batch]),
    }


# === 4. Logging helpers ===
def setup_run_dir():
    """Create a timestamped output directory for this run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def plot_loss_curve(csv_path, out_path):
    """Read the CSV log and save a loss curve plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, losses = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["global_step"]))
            losses.append(float(row["loss"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, linewidth=0.8)
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("GNN-JEPA Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved to {out_path}")


# === 5. Training loop ===
def main():
    print(f"Training GNN-JEPA on {DEVICE}...")
    run_dir = setup_run_dir()
    csv_path = os.path.join(run_dir, "train_log.csv")
    plot_path = os.path.join(run_dir, "loss_curve.png")
    ckpt_path = os.path.join(run_dir, "gnn_jepa.pth")
    print(f"Outputs: {run_dir}/")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load data
    raw_data = torch.load(DATA_FILE, weights_only=False)
    print(f"Loaded {len(raw_data)} samples from {DATA_FILE}")
    dataset = GraphListDataset(raw_data, tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

    model = NeuroSymbolicJEPA().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # CSV logger
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "global_step", "loss", "pred_norm", "target_norm"])

    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in loop:
            prev_batch = batch["prev_graph_batch"].to(DEVICE)
            next_batch = batch["next_graph_batch"].to(DEVICE)
            code_ids = batch["code_ids"].to(DEVICE)
            code_mask = batch["code_mask"].to(DEVICE)

            # Forward: predictor output
            pred_emb = model(prev_batch, code_ids, code_mask)

            # Target: EMA encoder (no gradients)
            target_emb = model.encode_target_graph(next_batch)

            # Loss
            loss = criterion(pred_emb, target_emb)

            # NaN guard
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARN] NaN/Inf loss at step {global_step}, skipping batch")
                optimizer.zero_grad()
                global_step += 1
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            # EMA update target encoder
            model.update_target_encoder()

            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_steps += 1
            global_step += 1

            loop.set_postfix(loss=f"{loss_val:.4f}")

            # Log every N steps
            if global_step % LOG_EVERY == 0:
                pred_norm = pred_emb.detach().norm(dim=1).mean().item()
                target_norm = target_emb.detach().norm(dim=1).mean().item()
                csv_writer.writerow(
                    [epoch + 1, global_step, f"{loss_val:.6f}", f"{pred_norm:.4f}", f"{target_norm:.4f}"]
                )
                csv_file.flush()

        avg = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch+1} — Average Loss: {avg:.6f}")

    csv_file.close()

    # Save model
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")

    # Plot loss curve
    plot_loss_curve(csv_path, plot_path)


if __name__ == "__main__":
    main()
