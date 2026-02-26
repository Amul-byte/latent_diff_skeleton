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
    """
    Loads a CSV into a float32 numpy array robustly.
    Handles:
      - empty files
      - blank lines
      - non-numeric cells like ''
    Returns None if file is unusable.
    """
    try:
        # genfromtxt is more tolerant than loadtxt (handles missing values)
        arr = np.genfromtxt(
            fpath,
            delimiter=",",
            dtype=np.float32,
            invalid_raise=False,
        )
    except Exception as e:
        print(f"[SKIP] Failed reading {fpath}: {e}")
        return None

    # genfromtxt can return scalar nan if file empty or nonsense
    if arr is None:
        print(f"[SKIP] {fpath}: read returned None")
        return None

    # If empty -> arr.size == 0 OR all nan
    if not hasattr(arr, "size") or arr.size == 0:
        print(f"[SKIP] {fpath}: empty array")
        return None

    # If arr is 1D but represents rows, keep it and reshape later
    # If arr is scalar (0-dim), skip
    if arr.ndim == 0:
        print(f"[SKIP] {fpath}: scalar read (likely empty/bad)")
        return None

    # Replace NaNs/Infs with 0 (or you can choose mean-impute)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return arr


def coerce_to_Tx96(arr: np.ndarray, fpath: str) -> np.ndarray | None:
    """
    Convert various possible shapes into [T, 96]:
      - [T, 96]
      - [T, 97] -> drop first col (common timestamp/index col)
      - [T, 32, 3] -> reshape
      - [T*96] -> reshape
    Returns None if cannot be coerced.
    """
    # Case: [T, 97] -> drop first column
    if arr.ndim == 2 and arr.shape[1] == 97:
        arr = arr[:, 1:]

    # Case: [T, 96]
    if arr.ndim == 2 and arr.shape[1] == 96:
        return arr

    # Case: flattened [T*96]
    if arr.ndim == 1 and (arr.size % 96 == 0):
        T = arr.size // 96
        return arr.reshape(T, 96)

    # Case: [T, 32, 3]
    if arr.ndim == 3 and arr.shape[1:] == (32, 3):
        T = arr.shape[0]
        return arr.reshape(T, 96)

    print(f"[SKIP] {fpath}: unexpected shape {arr.shape} (expected Tx96 / Tx97 / (T,32,3) / flat)")
    return None


def parse_label_from_filename(fname: str, num_classes: int) -> int:
    """
    Default label parser for SmartFall-like names: S48A02T05.csv
    Returns class index in [0..num_classes-1].
    If parsing fails, returns 0 (safe default).
    """
    # Find Axx pattern
    m = re.search(r"A(\d+)", fname)
    if not m:
        return 0
    a = int(m.group(1))  # e.g. 2 for A02
    # Convert A01 -> 0, A02 -> 1, ...
    y = max(0, a - 1)
    if y >= num_classes:
        # clamp to valid range to avoid crash if num_classes mismatched
        y = num_classes - 1
    return y


# -------------------------
# Dataset: folder of CSVs -> sliding windows
# -------------------------
class SkeletonSlidingWindowDataset(Dataset):
    def __init__(
        self,
        skeleton_dir: str,
        window: int = 90,
        stride: int = 30,
        num_classes: int = 12,
        drop_first_col_if_97: bool = True,
        min_windows_per_file: int = 1,
        strict: bool = False,
        label_mode: str = "filename",  # "filename" or "zero"
    ):
        self.window = window
        self.stride = stride
        self.num_classes = num_classes
        self.drop_first_col_if_97 = drop_first_col_if_97
        self.min_windows_per_file = min_windows_per_file
        self.strict = strict
        self.label_mode = label_mode

        self.files = sorted(glob.glob(os.path.join(skeleton_dir, "*.csv")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No CSV files found in: {skeleton_dir}")

        # IMPORTANT FIX:
        # Build cache FIRST, with cache_idx that only increments for good files.
        self.data_cache: list[np.ndarray] = []
        self.file_names: list[str] = []      # aligned with data_cache
        self.file_paths: list[str] = []      # aligned with data_cache

        # index is list of (cache_fi, start_frame)
        self.index: list[tuple[int, int]] = []

        skipped = 0
        total_windows = 0

        for fpath in self.files:
            fname = os.path.basename(fpath)

            arr = safe_load_csv_float32(fpath)
            if arr is None:
                skipped += 1
                continue

            arr = coerce_to_Tx96(arr, fpath)
            if arr is None:
                skipped += 1
                continue

            T = arr.shape[0]
            if T < self.window:
                # too short
                skipped += 1
                continue

            # How many windows will this file yield?
            n_w = 1 + (T - self.window) // self.stride
            if n_w < self.min_windows_per_file:
                skipped += 1
                continue

            # Add to cache and create windows using cache index (NOT original file idx)
            cache_fi = len(self.data_cache)
            self.data_cache.append(arr)
            self.file_names.append(fname)
            self.file_paths.append(fpath)

            for start in range(0, T - self.window + 1, self.stride):
                self.index.append((cache_fi, start))
                total_windows += 1

        if len(self.index) == 0:
            raise ValueError(
                f"No windows created. "
                f"Check window/stride vs sequence lengths. "
                f"(skipped_files={skipped}, total_files={len(self.files)})"
            )

        print(
            f"[Dataset] Loaded {len(self.data_cache)} valid files "
            f"(skipped {skipped}/{len(self.files)}), "
            f"total windows={len(self.index)}"
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        cache_fi, start = self.index[idx]

        # This should never fail now, but keep assertion for sanity
        if cache_fi < 0 or cache_fi >= len(self.data_cache):
            raise IndexError(
                f"Bad cache_fi={cache_fi} (len(data_cache)={len(self.data_cache)}), "
                f"idx={idx}, len(index)={len(self.index)}"
            )

        x_np = self.data_cache[cache_fi][start : start + self.window]  # [W, 96]

        if x_np.shape != (self.window, 96):
            # safety net
            msg = f"[Bad window] file={self.file_names[cache_fi]} start={start} got={x_np.shape}"
            if self.strict:
                raise RuntimeError(msg)
            # fallback: pad/truncate to ensure shape
            x_fixed = np.zeros((self.window, 96), dtype=np.float32)
            w = min(self.window, x_np.shape[0])
            d = min(96, x_np.shape[1]) if x_np.ndim == 2 else 96
            if x_np.ndim == 2:
                x_fixed[:w, :d] = x_np[:w, :d]
            x_np = x_fixed

        if self.label_mode == "filename":
            y = parse_label_from_filename(self.file_names[cache_fi], self.num_classes)
        else:
            y = 0  # dummy

        x = torch.from_numpy(x_np).float()           # [W, 96]
        y = torch.tensor(y, dtype=torch.long)        # scalar class index

        return x, y


# -------------------------
# A very simple LSTM classifier
# -------------------------
class SimpleLSTM(torch.nn.Module):
    def __init__(self, input_size=96, hidden=256, num_classes=12):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            batch_first=True,
            num_layers=2,
            dropout=0.1,
        )
        self.fc = torch.nn.Linear(hidden, num_classes)

    def forward(self, x):  # x: [B, T, D]
        out, _ = self.lstm(x)
        last = out[:, -1, :]          # [B, hidden]
        return self.fc(last)          # [B, C]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skeleton_dir", type=str, required=True)
    p.add_argument("--window", type=int, default=90)
    p.add_argument("--stride", type=int, default=30)
    p.add_argument("--num_classes", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--label_mode", type=str, default="filename", choices=["filename", "zero"])
    args = p.parse_args()

    ds = SkeletonSlidingWindowDataset(
        skeleton_dir=args.skeleton_dir,
        window=args.window,
        stride=args.stride,
        num_classes=args.num_classes,
        label_mode=args.label_mode,
        strict=False,
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

    model = SimpleLSTM(input_size=96, num_classes=args.num_classes).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val = 1e9
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
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
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)
                logits = model(x)
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
            torch.save(model.state_dict(), "best_skeleton_model.pth")
            print("Saved: best_skeleton_model.pth")


if __name__ == "__main__":
    main()