"""
model_loader.py

This file helps you:
1) Load a PyTorch checkpoint safely
2) Find the correct "state_dict" inside the checkpoint
3) Load weights into your model and print what matched / did not match
4) Freeze a model (stop it from training)

Goal: make it VERY easy to read and debug.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union, Optional

import torch
import torch.nn as nn


# ------------------------------------------------------------
# 1) Safe torch.load (different PyTorch versions behave differently)
# ------------------------------------------------------------
def safe_torch_load(path: str, map_location: Union[str, torch.device] = "cpu") -> Any:
    """
    Load a checkpoint file from disk.

    Some PyTorch versions support torch.load(..., weights_only=...)
    and some don't. So we try one, and if it fails, we try the other.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # older PyTorch doesn't have weights_only argument
        return torch.load(path, map_location=map_location)


# ------------------------------------------------------------
# 2) Small helpers to compare checkpoint keys vs model keys
# ------------------------------------------------------------
def count_key_overlap(keys_a: Iterable[str], keys_b: Iterable[str]) -> int:
    """
    Count how many keys are shared between two key lists/sets.
    """
    return len(set(keys_a).intersection(set(keys_b)))


def state_dict_match_score(
    candidate: Dict[str, torch.Tensor],
    model_state: Dict[str, torch.Tensor],
) -> Tuple[int, int]:
    """
    Score how well a candidate state_dict matches the model state_dict.

    Returns:
      (shape_matches, key_overlap)

    shape_matches:
      how many keys match AND have the same tensor shape
    key_overlap:
      how many keys overlap at all
    """
    overlap = count_key_overlap(candidate.keys(), model_state.keys())

    shape_matches = 0
    for k, v in candidate.items():
        if k in model_state:
            # compare shapes if both have shape
            if hasattr(v, "shape") and hasattr(model_state[k], "shape"):
                if tuple(v.shape) == tuple(model_state[k].shape):
                    shape_matches += 1

    return shape_matches, overlap


# ------------------------------------------------------------
# 3) Find the real state_dict inside a checkpoint payload
# ------------------------------------------------------------
def resolve_state_dict(payload: Any, model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Checkpoints can be saved in many formats.
    This function tries to find the actual model weights dictionary.

    Common patterns:
      payload["state_dict"]
      payload["model"]
      payload["ema"]
      payload["something_else"]
      or sometimes payload IS already the state_dict

    We try multiple options and pick the one that best matches your model.
    """

    # model.state_dict() is what we *want* to match
    model_state = model.state_dict()

    # If payload isn't even a dict, we can't handle it
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload is not a dict, cannot find state_dict.")

    # (A) Common: payload has "state_dict"
    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]

    # (B) Sometimes payload has a key matching model class name
    # Example: model class is "Stage3Model" -> key might be "stage3model"
    model_key = model.__class__.__name__.lower()
    if model_key in payload and isinstance(payload[model_key], dict):
        return payload[model_key]

    # (C) Sometimes payload itself is already a state_dict:
    # keys are strings and values are tensors
    # We'll detect by overlap with model keys and tensor-like values
    all_string_keys = all(isinstance(k, str) for k in payload.keys())
    if all_string_keys:
        candidate = payload
        shape_score, overlap_score = state_dict_match_score(candidate, model_state)
        looks_like_tensors = all(hasattr(v, "shape") for v in candidate.values())

        # If it overlaps enough, treat it as state_dict
        if shape_score > 0 or (overlap_score > 0 and looks_like_tensors):
            return candidate

    # (D) Otherwise, search nested dicts inside payload and pick best match
    best_key: Optional[str] = None
    best_shape_score = -1
    best_overlap_score = -1

    for k, v in payload.items():
        # only consider nested dicts with string keys
        if isinstance(v, dict) and all(isinstance(inner_k, str) for inner_k in v.keys()):
            shape_score, overlap_score = state_dict_match_score(v, model_state)

            # choose best by:
            # 1) more shape matches
            # 2) if tie, more overlap
            if (shape_score > best_shape_score) or (shape_score == best_shape_score and overlap_score > best_overlap_score):
                best_key = k
                best_shape_score = shape_score
                best_overlap_score = overlap_score

    if best_key is not None and (best_shape_score > 0 or best_overlap_score > 0):
        return payload[best_key]

    # If nothing worked, fail with a clear message
    raise ValueError(
        "Could not find a usable state_dict inside checkpoint. "
        "Try printing checkpoint keys to see what's inside."
    )


# ------------------------------------------------------------
# 4) Main function: load checkpoint into model
# ------------------------------------------------------------
def load_checkpoint(
    path: str,
    model: nn.Module,
    strict: bool = True,
    map_location: Union[str, torch.device] = "cpu",
) -> Tuple[Any, list[str], list[str]]:
    """
    Load weights into a model.

    Args:
      path: checkpoint path
      model: your torch model
      strict:
        True  -> must match exactly
        False -> load what matches, ignore the rest (useful when architecture changes)
      map_location:
        usually "cpu" when loading on any machine

    Returns:
      (payload, missing_keys, unexpected_keys)
    """

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 1) load raw payload
    payload = safe_torch_load(str(checkpoint_path), map_location=map_location)

    # 2) find the state_dict inside
    state_dict = resolve_state_dict(payload, model)

    # 3) load into model
    incompatible = model.load_state_dict(state_dict, strict=strict)

    # 4) save diagnostic info
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)

    # 5) print useful debug info (so you can see what happened)
    print("\n================ CHECKPOINT LOAD ================")
    print(f"Path:   {checkpoint_path}")
    print(f"Strict: {strict}")
    print(f"Missing keys:    {len(missing_keys)}")
    for k in missing_keys[:50]:  # print first 50 so it doesn't explode
        print(f"  - missing: {k}")
    if len(missing_keys) > 50:
        print("  ... (more missing keys not shown)")

    print(f"Unexpected keys: {len(unexpected_keys)}")
    for k in unexpected_keys[:50]:
        print(f"  - unexpected: {k}")
    if len(unexpected_keys) > 50:
        print("  ... (more unexpected keys not shown)")
    print("=================================================\n")

    return payload, missing_keys, unexpected_keys


# ------------------------------------------------------------
# 5) Freeze model parameters (stop training them)
# ------------------------------------------------------------
def freeze_module(module: nn.Module) -> None:
    """
    Freeze a module so it does NOT learn / update weights.

    This is used when:
    - you want to keep Stage 1 frozen
    - and train only Stage 2 or Stage 3
    """
    for p in module.parameters():
        p.requires_grad = False

    module.eval()  # evaluation mode (turn off dropout etc.)
    verify_frozen(module)


def verify_frozen(module: nn.Module) -> bool:
    """
    Double-check that all parameters are frozen.
    If any parameter still wants gradients, raise an error.
    """
    for name, p in module.named_parameters():
        if p.requires_grad:
            raise AssertionError(f"Parameter is not frozen: {name}")
    return True


# ------------------------------------------------------------
# (Optional) Nice extra helper: print checkpoint top-level keys
# ------------------------------------------------------------
def print_checkpoint_keys(payload: Any, max_items: int = 50) -> None:
    """
    If you are confused about what's inside a checkpoint,
    call this after safe_torch_load().

    Example:
      payload = safe_torch_load("ckpt.pt")
      print_checkpoint_keys(payload)
    """
    if not isinstance(payload, dict):
        print("Checkpoint is not a dict. Type:", type(payload))
        return

    keys = list(payload.keys())
    print(f"Checkpoint has {len(keys)} top-level keys:")
    for k in keys[:max_items]:
        print(" -", k)
    if len(keys) > max_items:
        print(" ... (more keys not shown)")