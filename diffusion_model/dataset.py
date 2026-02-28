# ============================================================
# This file helps you load SmartFall data from CSV files
# and turn it into training samples (windows) for a model.
#
# It works with:
#   1) Skeleton CSVs (32 joints -> 32 * 3 = 96 numbers per frame)
#   2) Sensor 1 CSVs (last 3 columns are accel x,y,z)
#   3) Sensor 2 CSVs (last 3 columns are accel x,y,z)
#
# Each CSV filename looks like: S01A10T03.csv
#   S01 = subject 01
#   A10 = activity 10
#   T03 = trial 03
#
# We will create windows of length T (example: 90 frames)
# so the model always sees the same size input.
# ============================================================
#One critical behavioral detail (you should be aware of)

# Every window in a fall trial is labeled as “fall” (1) even if the fall happens only near the end.

# That means your “fall” class contains:

# pre-fall walking

# destabilization

# impact/post-fall

# If you later train a classifier or condition on y, the signal can be noisy unless your method is trial-level on purpose.

import os               # lets us look inside folders and list files
import re               # lets us find patterns like "A10" inside filenames
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd     # reads CSV files into tables (DataFrames)
import numpy as np      # fast math arrays
import torch            # tensors for PyTorch
from torch.utils.data import Dataset  # base class for PyTorch datasets

# ------------------------------------------------------------
# PART 1) READ ALL CSV FILES FROM A FOLDER
# ------------------------------------------------------------

def read_csv_files(folder):
    """
    Read every .csv file in a folder.
    Return a dictionary:
        key   = filename (example: "S01A01T01.csv")
        value = pandas DataFrame (the CSV data)
    """

    # get all file names in the folder that end with ".csv"
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]

    # this will store all CSV tables
    data = {}

    # loop through each csv filename
    for file in files:

        # build the full path like ".../folder/S01A01T01.csv"
        file_path = os.path.join(folder, file)

        try:
            # read the csv into a DataFrame (header=None means no header row)
            data[file] = pd.read_csv(file_path, header=None)

        # if file is empty or broken, skip it
        except pd.errors.EmptyDataError:
            print(f"[dataset] Skipped empty/unreadable file: {file}")

        # if any other error happens, skip it and print why
        except Exception as e:
            print(f"[dataset] Skipped file {file} due to error: {e}")

    # return the dictionary of all loaded CSVs
    return data


def common_files(*dicts):
    """
    Find filenames that exist in ALL input dictionaries.

    Example:
      skeleton_data has "S01A01T01.csv"
      sensor1_data has "S01A01T01.csv"
      sensor2_data has "S01A01T01.csv"
    Then that file is common.

    Returns a sorted list of common filenames.
    """

    # if no dictionaries were given, return empty list
    if not dicts:
        return []

    # start with keys from the first dictionary
    s = set(dicts[0].keys())

    # keep only filenames that are also in every other dictionary
    for d in dicts[1:]:
        s &= set(d.keys())

    # return as a sorted list (nice and consistent order)
    return sorted(list(s))


# ------------------------------------------------------------
# PART 2) FIX NaNs (missing values)
# ------------------------------------------------------------

def fill_nan_with_col_mean(x: np.ndarray) -> np.ndarray:
    """
    Sometimes data has NaN (means missing / not a number).
    We replace NaNs using the average (mean) of that column.

    If a whole column is ALL NaNs, we set that column to 0.
    """

    # make sure the array is float32 (common for ML) and make a copy
    x = x.astype(np.float32, copy=True)

    # find columns where every single value is NaN
    all_nan_cols = np.all(np.isnan(x), axis=0)

    # if there are any "all NaN columns", replace them with zeros
    if np.any(all_nan_cols):
        x[:, all_nan_cols] = 0.0

    # compute the mean of each column, ignoring NaNs
    col_mean = np.nanmean(x, axis=0)

    # find where NaNs are located
    nan_mask = np.isnan(x)

    # if there are NaNs, replace them with the column mean
    if np.any(nan_mask):
        # np.take(col_mean, ...) picks the right mean for each NaN position
        x[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])

    # return the cleaned array
    return x


# ------------------------------------------------------------
# PART 3) GET LABELS AND SUBJECT IDS FROM THE FILENAME
# ------------------------------------------------------------

# This looks for "A" followed by 2 digits, like A01, A10, A14
ACT_RE = re.compile(r"A(\d{2})", re.IGNORECASE)

# This looks for "S" followed by 2 digits, like S01, S12
SUB_RE = re.compile(r"S(\d{2})", re.IGNORECASE)


def label_from_filename_binary(fname: str, fall_activities=(10, 11, 12, 13, 14)) -> int:
    """
    We make a simple label:
      0 = normal activity (ADL)
      1 = fall

    By default, SmartFall falls are activities A10 to A14.

    Example:
      "S01A10T03.csv" -> activity 10 -> label 1 (fall)
      "S01A02T01.csv" -> activity 2  -> label 0 (ADL)
    """

    # find activity code in the filename
    m = ACT_RE.search(fname)

    # if it cannot find Axx, raise an error
    if not m:
        raise ValueError(f"Cannot parse activity code from filename: {fname}")

    # convert "10" into number 10
    act = int(m.group(1))

    # if activity is a fall activity, return 1 else 0
    return 1 if act in set(fall_activities) else 0


def subject_from_filename(fname: str) -> int:
    """
    Get subject number from filename.

    Example:
      "S01A01T01.csv" -> subject 01 -> returns 1
    """

    # find subject code in the filename
    m = SUB_RE.search(fname)

    # if it cannot find Sxx, raise an error
    if not m:
        raise ValueError(f"Cannot parse subject id from filename: {fname}")

    # convert "01" into number 1
    return int(m.group(1))


# ------------------------------------------------------------
# PART 4) SKELETON: TURN [T,96] into [T,32,3]
# ------------------------------------------------------------

def skeleton_window_to_T32x3(skel_win: np.ndarray) -> np.ndarray:
    """
    Skeleton CSV per frame often has:
      - 96 columns (32 joints * 3 numbers per joint)
      OR
      - 97 columns (first column is an index, then 96 skeleton values)

    Input:
      skel_win shape = [T, 96] or [T, 97]

    Output:
      [T, 32, 3]
      meaning:
        T frames,
        32 joints,
        each joint has (x,y,z)
    """

    # check it is 2D (rows and columns)
    if skel_win.ndim != 2:
        raise ValueError(f"Skeleton window must be 2D [T,C]. Got {skel_win.shape}")

    # if there is an extra first column (97 columns), drop it
    if skel_win.shape[1] == 97:
        skel_win = skel_win[:, 1:]

    # now it MUST be 96 columns
    if skel_win.shape[1] != 96:
        raise ValueError(f"Expected skeleton cols=96 (or 97 with index). Got {skel_win.shape[1]}")

    # number of frames in this window
    T = skel_win.shape[0]

    # reshape from [T, 96] to [T, 32, 3]
    # because 96 = 32 * 3
    return skel_win.reshape(T, 32, 3).astype(np.float32)


# ------------------------------------------------------------
# PART 5) IMU NORMALIZATION (Z-SCORE)
# ------------------------------------------------------------

def compute_zscore_stats(accel: torch.Tensor, eps: float = 1e-6):
    """
    accel is a tensor shaped [N, T, 3]
      N = number of windows (samples)
      T = frames per window
      3 = x,y,z

    We compute:
      mean = average value for each axis (x,y,z)
      std  = standard deviation for each axis

    We keep shape [1,1,3] so it can broadcast easily.

    eps is a tiny number so we never divide by zero.
    """

    # mean over sample dimension and time dimension
    mean = accel.mean(dim=(0, 1), keepdim=True)

    # std over sample dimension and time dimension
    std = accel.std(dim=(0, 1), keepdim=True).clamp_min(eps)

    # return both stats
    return {"mean": mean, "std": std}


def normalize_accel(
    accel: torch.Tensor,
    stats: dict,
    eps: float = 1e-6,
    clip: float | None = 6.0,
) -> torch.Tensor:
    """
    Normalize accel data using:
      (accel - mean) / std

    accel: [N,T,3]  (or [1,T,3] for one window)
    stats: {"mean": [1,1,3], "std": [1,1,3]}

    clip: if not None, clamp values into [-clip, +clip]
    """

    # make sure mean and std are tensors
    mean = torch.as_tensor(stats["mean"], dtype=accel.dtype)
    std = torch.as_tensor(stats["std"], dtype=accel.dtype).clamp_min(eps)

    # z-score normalize
    out = (accel - mean) / std

    # optional clipping (keeps extreme spikes from exploding training)
    if clip is not None:
        c = float(clip)
        out = out.clamp(-c, c)

    return out


# ------------------------------------------------------------
# PART 6) THE PYTORCH DATASET CLASS
# ------------------------------------------------------------

class SmartFallPairedSlidingWindowDataset(Dataset):
    """
    This dataset builds training samples (windows) from SmartFall CSVs.

    Each sample you get is a dictionary with:
      "X"      : skeleton window  [T, 32, 3]
      "A1"     : sensor1 accel    [T, 3]
      "A2"     : sensor2 accel    [T, 3]
      "A_pair" : stacked accel    [2, T, 3]
      "y"      : label (0 or 1)
    """

    def __init__(
        self,
        skeleton_data: dict,
        sensor1_data: dict,
        sensor2_data: dict,
        window_size: int = 90,
        stride: int = 30,
        fall_activities=(10, 11, 12, 13, 14),
        drop_misaligned: bool = True,
        align_mode: str = "strict",

        # Filter which files/subjects to use (helps subject-wise split)
        allowed_files: list[str] | None = None,
        allowed_subjects: list[int] | None = None,

        # IMU normalization settings
        imu_normalization: str = "zscore",   # "zscore" or "none"
        imu_stats: dict | None = None,       # pass train stats here for val/test
        imu_eps: float = 1e-6,
        imu_clip: float | None = 6.0,
        sensor_names: tuple[str, str] = ("right_hip", "left_wrist"),
        sensor_roots: tuple[str, str] | None = None,
        strict_sensor_identity: bool = True,
    ):
        # store the input CSV dictionaries
        self.skeleton_data = skeleton_data
        self.sensor1_data = sensor1_data
        self.sensor2_data = sensor2_data

        # store window settings
        self.window_size = int(window_size)
        self.stride = int(stride)

        # store fall activities list
        self.fall_activities = tuple(int(x) for x in fall_activities)

        # if True, skip trials where row counts mismatch
        self.drop_misaligned = bool(drop_misaligned)
        self.align_mode = str(align_mode).lower()
        if self.align_mode not in ("strict", "truncate_min"):
            raise ValueError("align_mode must be 'strict' or 'truncate_min'")

        # store IMU normalization settings
        self.imu_normalization = imu_normalization
        self.imu_eps = float(imu_eps)
        self.imu_clip = imu_clip
        self.sensor_names = sensor_names
        self.sensor_roots = sensor_roots
        self.strict_sensor_identity = bool(strict_sensor_identity)

        if self.strict_sensor_identity:
            if self.sensor_roots is None:
                raise ValueError(
                    "strict_sensor_identity=True requires sensor_roots=(right_hip_folder, left_wrist_folder)."
                )
            if len(self.sensor_roots) != 2:
                raise ValueError("sensor_roots must contain exactly two paths.")
            self._validate_sensor_identity_from_roots()

        # find filenames common across skeleton/sensor1/sensor2
        files = common_files(skeleton_data, sensor1_data, sensor2_data)

        # if no common files, dataset cannot be built
        if not files:
            raise ValueError("No common filenames across skeleton/sensor1/sensor2 inputs.")

        # -------------------------
        # OPTIONAL FILTERING
        # -------------------------

        # if allowed_files is given, keep only those filenames
        if allowed_files is not None:
            allowed_set = set(allowed_files)
            files = [f for f in files if f in allowed_set]

        # if allowed_subjects is given, keep only those subjects
        if allowed_subjects is not None:
            subj_set = set(int(s) for s in allowed_subjects)
            files = [f for f in files if subject_from_filename(f) in subj_set]

        # if filtering removed everything, error
        if not files:
            raise ValueError("After filtering, no files remain. Check allowed_files/allowed_subjects.")

        # store final file list
        self.files = files

        # ------------------------------------------------
        # BUILD AN INDEX OF ALL WINDOWS
        # ------------------------------------------------
        # self.index will store tuples: (filename, window_start_frame)
        self.index = []

        # labels for each window
        labels = []

        # loop over each file/trial
        for fname in self.files:

            # get the DataFrames for this file
            skel_df = self.skeleton_data[fname]
            s1_df = self.sensor1_data[fname]
            s2_df = self.sensor2_data[fname]

            # get number of rows (frames) in each
            n0 = len(skel_df)
            n1 = len(s1_df)
            n2 = len(s2_df)

            if self.align_mode == "strict":
                # strict mode expects equal sequence length
                if not (n0 == n1 == n2):
                    if self.drop_misaligned:
                        continue
                    raise ValueError(f"Frame mismatch {fname}: skel={n0}, s1={n1}, s2={n2}")
                n_frames = n0
            else:
                # truncate mode uses overlapping frames across modalities
                n_frames = min(n0, n1, n2)

            # window length
            T = self.window_size

            # stride step
            step = self.stride

            # if trial is shorter than one window, skip
            if n_frames < T:
                continue

            # label from filename (0 or 1)
            y = label_from_filename_binary(fname, fall_activities=self.fall_activities)

            # number of windows we can slide through
            num_windows = (n_frames - T) // step + 1

            # add each window start to the index list
            for i in range(num_windows):
                start = i * step
                self.index.append((fname, start))
                labels.append(y)

        # if no windows were created, error
        if not self.index:
            raise ValueError("No windows created. Check window_size/stride or data lengths.")

        # store labels as a torch tensor
        self.labels = torch.tensor(labels, dtype=torch.long)

        # ------------------------------------------------
        # IMU NORMALIZATION STATS
        # ------------------------------------------------
        # If zscore normalization is ON:
        #   - Use imu_stats if provided (val/test)
        #   - Else compute stats from this dataset windows (train)
        self.normalization_stats = None

        # only allow two choices
        if self.imu_normalization not in ("zscore", "none"):
            raise ValueError("imu_normalization must be 'zscore' or 'none'")

        # if we want zscore normalization
        if self.imu_normalization == "zscore":

            # CASE 1: stats were given (use them)
            if imu_stats is not None:
                s1_name, s2_name = sensor_names

                self.normalization_stats = {
                    s1_name: {
                        "mean": torch.as_tensor(imu_stats[s1_name]["mean"]),
                        "std":  torch.as_tensor(imu_stats[s1_name]["std"]),
                    },
                    s2_name: {
                        "mean": torch.as_tensor(imu_stats[s2_name]["mean"]),
                        "std":  torch.as_tensor(imu_stats[s2_name]["std"]),
                    },
                }

            # CASE 2: stats NOT given (compute from data)
            else:
                A1_all = []  # will store every sensor1 window
                A2_all = []  # will store every sensor2 window

                # loop over every window index
                for (fname, start) in self.index:
                    end = start + self.window_size

                    # read the correct rows for accel (last 3 columns)
                    s1_df = self.sensor1_data[fname]
                    s2_df = self.sensor2_data[fname]

                    s1_win = s1_df.iloc[start:end, -3:].values
                    s2_win = s2_df.iloc[start:end, -3:].values

                    # fix NaNs
                    s1_win = fill_nan_with_col_mean(s1_win)
                    s2_win = fill_nan_with_col_mean(s2_win)

                    # store
                    A1_all.append(s1_win.astype(np.float32))
                    A2_all.append(s2_win.astype(np.float32))

                # stack into big tensors:
                # A1_t shape = [N, T, 3]
                A1_t = torch.tensor(np.stack(A1_all, axis=0), dtype=torch.float32)
                A2_t = torch.tensor(np.stack(A2_all, axis=0), dtype=torch.float32)

                # compute mean/std for each sensor
                s1_name, s2_name = sensor_names
                self.normalization_stats = {
                    s1_name: compute_zscore_stats(A1_t, eps=self.imu_eps),
                    s2_name: compute_zscore_stats(A2_t, eps=self.imu_eps),
                }

    def __len__(self):
        # total number of windows
        return len(self.index)

    def get_normalization_stats(self) -> dict | None:
        """
        Return a safe copy of normalization stats.
        You use this from TRAIN dataset and pass into VAL/TEST dataset.
        """
        if self.normalization_stats is None:
            return None

        out = {}
        for k, v in self.normalization_stats.items():
            out[k] = {
                "mean": v["mean"].clone(),
                "std": v["std"].clone(),
            }
        return out

    def __getitem__(self, idx):
        """
        Get one training sample (one window).
        """

        # get which file and where the window starts
        fname, start = self.index[idx]

        # window end
        end = start + self.window_size

        # get the DataFrames for that file
        skel_df = self.skeleton_data[fname]
        s1_df = self.sensor1_data[fname]
        s2_df = self.sensor2_data[fname]

        # cut out the window rows
        skel_win = skel_df.iloc[start:end, :].values      # all skeleton cols
        s1_win = s1_df.iloc[start:end, -3:].values        # last 3 accel cols
        s2_win = s2_df.iloc[start:end, -3:].values        # last 3 accel cols

        # fix NaNs in all three windows
        skel_win = fill_nan_with_col_mean(skel_win)
        s1_win = fill_nan_with_col_mean(s1_win)
        s2_win = fill_nan_with_col_mean(s2_win)

        # reshape skeleton to [T,32,3]
        X = skeleton_window_to_T32x3(skel_win)

        # convert everything into torch tensors
        X_t = torch.tensor(X, dtype=torch.float32)           # [T,32,3]
        A1_t = torch.tensor(s1_win, dtype=torch.float32)     # [T,3]
        A2_t = torch.tensor(s2_win, dtype=torch.float32)     # [T,3]

        # strict shape contract for IMU streams
        T = self.window_size
        if tuple(A1_t.shape) != (T, 3):
            raise ValueError(f"A1 must be [T,3] with T={T}, got shape {tuple(A1_t.shape)} for file {fname}")
        if tuple(A2_t.shape) != (T, 3):
            raise ValueError(f"A2 must be [T,3] with T={T}, got shape {tuple(A2_t.shape)} for file {fname}")

        # get label for this window
        y_t = self.labels[idx]

        # if we want zscore normalization, apply it now (per sensor)
        if self.imu_normalization == "zscore":
            if self.normalization_stats is None:
                raise RuntimeError("imu_normalization='zscore' but normalization_stats is None.")

            s1_name, s2_name = self.sensor_names

            # normalize A1_t:
            # we temporarily add a batch dimension [1,T,3], normalize, then remove it
            A1_t = normalize_accel(
                A1_t.unsqueeze(0),                       # [1,T,3]
                self.normalization_stats[s1_name],        # mean/std
                eps=self.imu_eps,
                clip=self.imu_clip,
            ).squeeze(0)                                  # back to [T,3]

            # normalize A2_t similarly
            A2_t = normalize_accel(
                A2_t.unsqueeze(0),
                self.normalization_stats[s2_name],
                eps=self.imu_eps,
                clip=self.imu_clip,
            ).squeeze(0)

        # stack both sensors into one tensor [2,T,3]
        A_pair = torch.stack([A1_t, A2_t], dim=0)

        # return a dictionary (easy for training code)
        return {
            "X": X_t,                                  # skeleton [T,32,3]
            "A1": A1_t,                                # sensor1 [T,3]
            "A2": A2_t,                                # sensor2 [T,3]
            "A_pair": A_pair,                          # both sensors [2,T,3]
            "y": y_t,                                  # label scalar
            "file": fname,                             # debug info
            "start": torch.tensor(start, dtype=torch.long),  # debug info
        }

    def _validate_sensor_identity_from_roots(self) -> None:
        """
        Hard fail when folder naming does not match required sensor pair:
          - sensor1 path must indicate right hip
          - sensor2 path must indicate left wrist
        """
        assert self.sensor_roots is not None
        right_hip_root, left_wrist_root = self.sensor_roots

        p1 = os.path.basename(str(right_hip_root).strip("/\\")).lower()
        p2 = os.path.basename(str(left_wrist_root).strip("/\\")).lower()

        # expected mapping: A1=right_hip, A2=left_wrist
        # allow compact forms in folder names.
        hip_ok = ("hip" in p1) and ("right" in p1 or "rhip" in p1 or p1.endswith("hip") or "meta_hip" in p1)
        wrist_ok = ("wrist" in p2) and ("left" in p2 or "lwrist" in p2 or p2.endswith("wrist") or "meta_wrist" in p2)

        if not hip_ok or not wrist_ok:
            raise ValueError(
                "Sensor identity check failed. Expected A1=right hip folder and A2=left wrist folder. "
                f"Got sensor_roots=({right_hip_root}, {left_wrist_root})."
            )


# ------------------------------------------------------------
# PART 7) Lightweight tensor datasets used by train.py/generate.py
# ------------------------------------------------------------


@dataclass
class NormalizationConfig:
    mode: str = "zscore"
    eps: float = 1e-6
    clip: Optional[float] = 6.0

    def __post_init__(self) -> None:
        if self.mode not in ("zscore", "none"):
            raise ValueError("NormalizationConfig.mode must be 'zscore' or 'none'")


def _validate_skeleton_tensor(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        raise ValueError("Skeleton tensor is required")
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Skeleton tensor must be [N,T,J,3], got shape {tuple(x.shape)}")
    if x.shape[-1] != 3:
        raise ValueError(f"Skeleton tensor last dim must be 3, got shape {tuple(x.shape)}")
    return x


def _validate_accel_tensor(a: torch.Tensor, name: str) -> torch.Tensor:
    if a is None:
        raise ValueError(f"{name} tensor is required")
    a = torch.as_tensor(a, dtype=torch.float32)
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"{name} must be [N,T,3], got shape {tuple(a.shape)}")
    return a


def _normalize_pair(
    accel_primary: torch.Tensor,
    accel_secondary: torch.Tensor,
    normalization: NormalizationConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if normalization.mode == "none":
        return accel_primary, accel_secondary

    p_stats = compute_zscore_stats(accel_primary, eps=normalization.eps)
    s_stats = compute_zscore_stats(accel_secondary, eps=normalization.eps)
    accel_primary = normalize_accel(
        accel_primary, p_stats, eps=normalization.eps, clip=normalization.clip
    )
    accel_secondary = normalize_accel(
        accel_secondary, s_stats, eps=normalization.eps, clip=normalization.clip
    )
    return accel_primary, accel_secondary


class SkeletonDataset(Dataset):
    """Simple tensor dataset for skeleton-only Stage 1 training."""

    def __init__(self, skeleton: torch.Tensor, labels: Optional[torch.Tensor] = None) -> None:
        self.X = _validate_skeleton_tensor(skeleton)
        if labels is None:
            self.y = torch.full((self.X.shape[0],), -1, dtype=torch.long)
        else:
            labels = torch.as_tensor(labels, dtype=torch.long)
            if labels.ndim != 1 or labels.shape[0] != self.X.shape[0]:
                raise ValueError("labels must be [N] and align with skeleton batch size")
            self.y = labels

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {"X": self.X[idx], "y": self.y[idx]}


class IMUDataset(Dataset):
    """Simple tensor dataset for IMU-only generation/inference."""

    def __init__(
        self,
        accel_primary: torch.Tensor,
        accel_secondary: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        sensor_pair: Tuple[str, str] = ("sensor1", "sensor2"),
        normalization: Optional[NormalizationConfig] = None,
    ) -> None:
        self.sensor_pair = sensor_pair
        self.A1 = _validate_accel_tensor(accel_primary, "accel_primary")
        self.A2 = _validate_accel_tensor(accel_secondary, "accel_secondary")
        if self.A1.shape[:2] != self.A2.shape[:2]:
            raise ValueError("accel_primary and accel_secondary must share [N,T]")

        cfg = normalization or NormalizationConfig(mode="none")
        self.A1, self.A2 = _normalize_pair(self.A1, self.A2, cfg)

        if labels is None:
            self.y = torch.full((self.A1.shape[0],), -1, dtype=torch.long)
        else:
            labels = torch.as_tensor(labels, dtype=torch.long)
            if labels.ndim != 1 or labels.shape[0] != self.A1.shape[0]:
                raise ValueError("labels must be [N] and align with accel batch size")
            self.y = labels

    def __len__(self) -> int:
        return self.A1.shape[0]

    def __getitem__(self, idx: int) -> dict:
        a1 = self.A1[idx]
        a2 = self.A2[idx]
        return {
            "A1": a1,
            "A2": a2,
            "A_pair": torch.stack([a1, a2], dim=0),
            "y": self.y[idx],
        }


class PairedDataset(Dataset):
    """Simple tensor dataset for Stage 2/3 paired skeleton+IMU training."""

    def __init__(
        self,
        skeleton: torch.Tensor,
        accel_primary: torch.Tensor,
        accel_secondary: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        sensor_pair: Tuple[str, str] = ("sensor1", "sensor2"),
        normalization: Optional[NormalizationConfig] = None,
    ) -> None:
        self.sensor_pair = sensor_pair
        self.X = _validate_skeleton_tensor(skeleton)
        self.A1 = _validate_accel_tensor(accel_primary, "accel_primary")
        self.A2 = _validate_accel_tensor(accel_secondary, "accel_secondary")

        if self.X.shape[0] != self.A1.shape[0] or self.A1.shape != self.A2.shape:
            raise ValueError("skeleton/accel tensors must align on sample count and accel shape")
        if self.X.shape[1] != self.A1.shape[1]:
            raise ValueError("skeleton and accel sequence length T must match")

        cfg = normalization or NormalizationConfig(mode="none")
        self.A1, self.A2 = _normalize_pair(self.A1, self.A2, cfg)

        if labels is None:
            self.y = torch.full((self.X.shape[0],), -1, dtype=torch.long)
        else:
            labels = torch.as_tensor(labels, dtype=torch.long)
            if labels.ndim != 1 or labels.shape[0] != self.X.shape[0]:
                raise ValueError("labels must be [N] and align with sample count")
            self.y = labels

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> dict:
        a1 = self.A1[idx]
        a2 = self.A2[idx]
        return {
            "X": self.X[idx],
            "A1": a1,
            "A2": a2,
            "A_pair": torch.stack([a1, a2], dim=0),
            "y": self.y[idx],
        }
