import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 1. DATASET
# ============================================================

class PatchDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

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

def train_model(csv_path, epochs=30, batch_size=64, lr=1e-3):

    dataset = PatchDataset(csv_path)

    # 80/20 split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    model = PatchCNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            optimizer.zero_grad()

            preds = model(X)
            loss = loss_fn(preds, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Eval
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                preds = model(X)
                test_loss += loss_fn(preds, y).item()

        print(f"Epoch {epoch:02d} | Train: {train_loss/len(train_loader):.6f} "
              f"| Test: {test_loss/len(test_loader):.6f}")

    return model


# ============================================================
# 4. PREDICT ON A NEW 3×3 PATCH
# ============================================================

def predict_from_patch(model, patch9):
    arr = torch.tensor(patch9, dtype=torch.float32).view(1, 1, 3, 3)
    with torch.no_grad():
        pred = model(arr).numpy().reshape(-1)
    return {
        "normalx": pred[0],
        "normaly": pred[1]
    }


# ============================================================
# 5. MAIN
# ============================================================

if __name__ == "__main__":
    model = train_model("data.csv", epochs=40)

    example_patch = [
        0.0, 0.0, 0.5,
        0.0, 0.5, 1.0,
        0.5, 1.0, 1.0
    ]

    pred = predict_from_patch(model, example_patch)
    print(pred)