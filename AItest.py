import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import time
import os


# ============================================================
# 1. DATASET
# ============================================================

class PatchDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        # Replace inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Drop any rows containing NaN
        df = df.dropna()

        # 9 input features = 3×3 patch
        feature_cols = [
            "v00", "v01", "v02",
            "v10", "v11", "v12",
            "v20", "v21", "v22"
        ]
        X = df[feature_cols].values.astype(np.float32)

        # ---- Choose your target here ----
        # Predict normal vector (2 outputs)
        y = df[["normalx", "normaly"]].values.astype(np.float32)

        self.X = torch.tensor(X).view(-1, 1, 3, 3)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# 2. CNN MODEL — multiple outputs
# ============================================================

class PatchCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),

            nn.Linear(64, 2)   # TWO outputs: normalx, normaly
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 3. TRAINING LOOP
# ============================================================

def train_model(csv_path, epochs=30, batch_size=64, lr=1e-3, checkpoint_path="checkpoint.pth"):
    dataset = PatchDataset(csv_path)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    model = PatchCNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    start_epoch = 1

    # 🔹 Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            optimizer.zero_grad()
            preds = model(X)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                preds = model(X)
                test_loss += loss_fn(preds, y).item()

        train_avg = train_loss / len(train_loader)
        test_avg = test_loss / len(test_loader)

        print(f"Epoch {epoch:02d} | Train: {train_avg:.6f} | Test: {test_avg:.6f}")

        # 🔹 Save checkpoint every epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_avg,
            "test_loss": test_avg,
        }, checkpoint_path)

    return model


# ============================================================
# 4. PREDICT ON A NEW 3×3 PATCH
# ============================================================

def predict_from_patch(model, patch9):
    arr = torch.tensor(patch9, dtype=torch.float32).view(1, 1, 3, 3)
    with torch.no_grad():
        pred = model(arr).numpy().reshape(-1)
    return pred[0], pred[1]


# ============================================================
# 5. MAIN
# ============================================================

if __name__ == "__main__":
    model = train_model("data.csv", epochs=0)

    df = pd.read_csv("validation.csv")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    l2_errors = []
    linf_errors = []
    prediction_times = []

    for _, row in df.iterrows():
        patch = [
            row["v00"], row["v01"], row["v02"],
            row["v10"], row["v11"], row["v12"],
            row["v20"], row["v21"], row["v22"]
        ]



        start = time.perf_counter()
        px, py = predict_from_patch(model, patch)
        end = time.perf_counter()

        prediction_times.append(end - start)

        # Ground truth
        tx = row["normalx"]
        ty = row["normaly"]

        # Errors
        dx = px - tx
        dy = py - ty

        err_l2 = np.sqrt(dx**2 + dy**2)
        err_linf = max(abs(dx), abs(dy))

        l2_errors.append(err_l2)
        linf_errors.append(err_linf)

        if np.isnan(px) or np.isnan(py):
            print("Prediction is NaN")
            print("Patch:", patch)

        if np.isnan(tx) or np.isnan(ty):
            print("Ground truth is NaN at index")

        if np.isnan(err_l2):
            print("Error became NaN")
            print("dx:", dx, "dy:", dy)



    # Final aggregated metrics
    mean_l2 = np.mean(l2_errors)
    mean_linf = np.mean(linf_errors)
    max_linf = np.max(linf_errors)
    avg_time = np.mean(prediction_times)

    print("L2:", mean_l2)
    print("Média L∞:", mean_linf)
    print("Max L∞:", max_linf)
    print("Tempo médio por célula (s):", avg_time)