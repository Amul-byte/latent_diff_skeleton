import os
import glob
import argparse
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# -------------------------
# Robust CSV loading utilities
# -------------------------
def safe_load_csv_float32(fpath: str) -> np.ndarray | None:
    try:
        arr = np.genfromtxt(
            fpath,
            delimiter=",",
            dtype=np.float32,
            invalid_raise=False,
        )
    except Exception as e:
        print(f"[SKIP] Failed reading {fpath}: {e}")
        return None

    if arr is None or not hasattr(arr, "size") or arr.size == 0:
        print(f"[SKIP] {fpath}: empty/bad read")
        return None

    if arr.ndim == 0:
        print(f"[SKIP] {fpath}: scalar read (likely empty)")
        return None

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return arr


def coerce_to_Tx3(arr: np.ndarray, fpath: str, use_last_3_cols: bool = True) -> np.ndarray | None:
    """
    Converts sensor arrays into [T, 3].
    Accepts:
      - [T, 3]
      - [T, >=3] -> take last 3 columns if use_last_3_cols=True else first 3
      - [T*3] -> reshape
    """
    if arr.ndim == 2:
        if arr.shape[1] < 3:
            print(f"[SKIP] {fpath}: need >=3 cols, got {arr.shape}")
            return None
        if arr.shape[1] == 3:
            return arr
        return arr[:, -3:] if use_last_3_cols else arr[:, :3]

    if arr.ndim == 1 and (arr.size % 3 == 0):
        T = arr.size // 3
        return arr.reshape(T, 3)

    print(f"[SKIP] {fpath}: unexpected sensor shape {arr.shape}")
    return None


def parse_label_from_filename(fname: str, num_classes: int) -> int:
    m = re.search(r"A(\d+)", fname)
    if not m:
        return 0
    a = int(m.group(1))
    y = max(0, a - 1)
    if y >= num_classes:
        y = num_classes - 1
    return y


# -------------------------
# Dataset: paired sensors -> sliding windows
# -------------------------
class SensorSlidingWindowDataset(Dataset):
    """
    Expects two folders with matching filenames (e.g., wrist + hip).
    Returns: (x1, x2, y)
      x1: [W, 3]
      x2: [W, 3]
      y : class index (long)
    """
    def __init__(
        self,
        sensor_dir1: str,
        sensor_dir2: str,
        window: int = 90,
        stride: int = 30,
        num_classes: int = 12,
        use_last_3_cols: bool = True,
        label_mode: str = "filename",  # "filename" or "zero"
    ):
        self.window = window
        self.stride = stride
        self.num_classes = num_classes
        self.use_last_3_cols = use_last_3_cols
        self.label_mode = label_mode

        files1 = sorted(glob.glob(os.path.join(sensor_dir1, "*.csv")))
        files2 = sorted(glob.glob(os.path.join(sensor_dir2, "*.csv")))
        if len(files1) == 0:
            raise FileNotFoundError(f"No CSV files in sensor_dir1: {sensor_dir1}")
        if len(files2) == 0:
            raise FileNotFoundError(f"No CSV files in sensor_dir2: {sensor_dir2}")

        # Match by basename
        map1 = {os.path.basename(p): p for p in files1}
        map2 = {os.path.basename(p): p for p in files2}
        common_names = sorted(set(map1.keys()) & set(map2.keys()))
        if len(common_names) == 0:
            raise ValueError("No matching filenames between the two sensor folders.")

        self.data_cache1: list[np.ndarray] = []
        self.data_cache2: list[np.ndarray] = []
        self.file_names: list[str] = []
        self.file_paths1: list[str] = []
        self.file_paths2: list[str] = []
        self.index: list[tuple[int, int]] = []  # (cache_fi, start)

        skipped = 0

        for fname in common_names:
            f1 = map1[fname]
            f2 = map2[fname]

            a1 = safe_load_csv_float32(f1)
            a2 = safe_load_csv_float32(f2)
            if a1 is None or a2 is None:
                skipped += 1
                continue

            a1 = coerce_to_Tx3(a1, f1, use_last_3_cols=self.use_last_3_cols)
            a2 = coerce_to_Tx3(a2, f2, use_last_3_cols=self.use_last_3_cols)
            if a1 is None or a2 is None:
                skipped += 1
                continue

            # Align lengths (take min T) so windows pair cleanly
            T = min(a1.shape[0], a2.shape[0])
            if T < self.window:
                skipped += 1
                continue
            a1 = a1[:T]
            a2 = a2[:T]

            cache_fi = len(self.data_cache1)
            self.data_cache1.append(a1)
            self.data_cache2.append(a2)
            self.file_names.append(fname)
            self.file_paths1.append(f1)
            self.file_paths2.append(f2)

            for start in range(0, T - self.window + 1, self.stride):
                self.index.append((cache_fi, start))

        if len(self.index) == 0:
            raise ValueError(
                f"No windows created. Check window/stride. "
                f"(skipped_files={skipped}, common_files={len(common_names)})"
            )

        print(
            f"[Dataset] common_files={len(common_names)} "
            f"valid_files={len(self.data_cache1)} skipped={skipped} "
            f"total_windows={len(self.index)}"
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        cache_fi, start = self.index[idx]

        x1 = self.data_cache1[cache_fi][start : start + self.window]  # [W, 3]
        x2 = self.data_cache2[cache_fi][start : start + self.window]  # [W, 3]

        if self.label_mode == "filename":
            y = parse_label_from_filename(self.file_names[cache_fi], self.num_classes)
        else:
            y = 0

        x1 = torch.from_numpy(x1).float()
        x2 = torch.from_numpy(x2).float()
        y = torch.tensor(y, dtype=torch.long)
        return x1, x2, y


# -------------------------
# Simple 2-sensor classifier
# -------------------------
class SensorLSTM(torch.nn.Module):
    """
    Encodes each sensor with an LSTM, concatenates, classifies.
    """
    def __init__(self, hidden=128, num_classes=12):
        super().__init__()
        self.lstm1 = torch.nn.LSTM(input_size=3, hidden_size=hidden, batch_first=True, num_layers=2, dropout=0.1)
        self.lstm2 = torch.nn.LSTM(input_size=3, hidden_size=hidden, batch_first=True, num_layers=2, dropout=0.1)
        self.fc = torch.nn.Linear(2 * hidden, num_classes)

    def forward(self, x1, x2):  # [B,W,3], [B,W,3]
        o1, _ = self.lstm1(x1)
        o2, _ = self.lstm2(x2)
        h1 = o1[:, -1, :]
        h2 = o2[:, -1, :]
        return self.fc(torch.cat([h1, h2], dim=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sensor_dir1", type=str, required=True, help="e.g., wrist folder")
    p.add_argument("--sensor_dir2", type=str, required=True, help="e.g., hip folder")
    p.add_argument("--window", type=int, default=90)
    p.add_argument("--stride", type=int, default=30)
    p.add_argument("--num_classes", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_last_3_cols", action="store_true", help="If sensor CSV has many cols, use the last 3 as accel")
    p.add_argument("--label_mode", type=str, default="filename", choices=["filename", "zero"])
    args = p.parse_args()

    ds = SensorSlidingWindowDataset(
        sensor_dir1=args.sensor_dir1,
        sensor_dir2=args.sensor_dir2,
        window=args.window,
        stride=args.stride,
        num_classes=args.num_classes,
        use_last_3_cols=args.use_last_3_cols,
        label_mode=args.label_mode,
    )

    # quick split
    n = len(ds)
    n_train = int(0.8 * n)
    train_ds, val_ds = torch.utils.data.random_split(
        ds,
        [n_train, n - n_train],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = SensorLSTM(num_classes=args.num_classes).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val = 1e9
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x1, x2, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            x1 = x1.to(args.device, non_blocking=True)
            x2 = x2.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x1, x2)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x1, x2, y in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                x1 = x1.to(args.device, non_blocking=True)
                x2 = x2.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)

                logits = model(x1, x2)
                loss = loss_fn(logits, y)
                val_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        acc = 100.0 * correct / max(1, total)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={acc:.2f}%")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_sensor_model.pth")
            print("Saved: best_sensor_model.pth")


if __name__ == "__main__":
    main()