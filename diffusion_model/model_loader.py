"""Checkpoint loading and parameter freezing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union

import torch
import torch.nn as nn


def _safe_torch_load(path: str, map_location: Union[str, torch.device] = "cpu") -> Any:
    """Safely load a torch checkpoint while supporting multiple PyTorch versions.

    Args:
        path: Checkpoint path.
        map_location: Torch map location.

    Returns:
        Loaded checkpoint payload.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _best_overlap_score(keys_a: Iterable[str], keys_b: Iterable[str]) -> int:
    """Count overlap size between two key collections.

    Args:
        keys_a: First key iterable.
        keys_b: Second key iterable.

    Returns:
        Number of shared keys.
    """
    set_a = set(keys_a)
    set_b = set(keys_b)
    return len(set_a.intersection(set_b))


def _state_dict_match_score(candidate: Dict[str, torch.Tensor], model_state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """Compute compatibility score between a candidate and model state dict.

    Args:
        candidate: Candidate state dict.
        model_state: Target model state dict.

    Returns:
        Tuple ``(shape_matches, key_overlap)``.
    """
    overlap = _best_overlap_score(candidate.keys(), model_state.keys())
    shape_matches = 0
    for key, tensor in candidate.items():
        if key in model_state:
            target = model_state[key]
            if hasattr(tensor, "shape") and hasattr(target, "shape") and tuple(tensor.shape) == tuple(target.shape):
                shape_matches += 1
    return shape_matches, overlap


def _resolve_state_dict(payload: Any, model: nn.Module) -> Dict[str, torch.Tensor]:
    """Resolve the state dict from a checkpoint payload.

    Args:
        payload: Loaded checkpoint payload.
        model: Target model for loading.

    Returns:
        State dict compatible with ``model.load_state_dict``.
    """
    if isinstance(payload, dict):
        model_state = model.state_dict()

        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]

        model_key = model.__class__.__name__.lower()
        if model_key in payload and isinstance(payload[model_key], dict):
            return payload[model_key]

        if all(isinstance(key, str) for key in payload.keys()):
            direct_shape_score, direct_overlap_score = _state_dict_match_score(payload, model_state)
            direct_tensor_values = all(hasattr(value, "shape") for value in payload.values())
            if direct_shape_score > 0 or (direct_overlap_score > 0 and direct_tensor_values):
                return payload

            candidate_key = None
            candidate_shape_score = 0
            candidate_overlap_score = 0
            for key, value in payload.items():
                if isinstance(value, dict) and all(isinstance(item, str) for item in value.keys()):
                    shape_score, overlap_score = _state_dict_match_score(value, model_state)
                    if shape_score > candidate_shape_score or (
                        shape_score == candidate_shape_score and overlap_score > candidate_overlap_score
                    ):
                        candidate_key = key
                        candidate_shape_score = shape_score
                        candidate_overlap_score = overlap_score
            if candidate_key is not None and (candidate_shape_score > 0 or candidate_overlap_score > 0):
                nested = payload[candidate_key]
                if isinstance(nested, dict):
                    return nested

    raise ValueError("Unable to resolve state_dict from checkpoint payload")


def load_checkpoint(
    path: str,
    model: nn.Module,
    strict: bool = True,
    map_location: Union[str, torch.device] = "cpu",
) -> Tuple[Any, list[str], list[str]]:
    """Load model weights from a checkpoint with diagnostics.

    Args:
        path: Checkpoint path.
        model: Target model.
        strict: Passed to ``load_state_dict``.
        map_location: Torch map location.

    Returns:
        Tuple ``(payload, missing_keys, unexpected_keys)``.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = _safe_torch_load(str(checkpoint_path), map_location=map_location)
    state_dict = _resolve_state_dict(payload, model)
    incompatible = model.load_state_dict(state_dict, strict=strict)

    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)

    print(f"[load_checkpoint] path={checkpoint_path}")
    print(f"[load_checkpoint] strict={strict}")
    print(f"[load_checkpoint] missing_keys={len(missing_keys)}")
    if missing_keys:
        for key in missing_keys:
            print(f"  - missing: {key}")
    print(f"[load_checkpoint] unexpected_keys={len(unexpected_keys)}")
    if unexpected_keys:
        for key in unexpected_keys:
            print(f"  - unexpected: {key}")

    return payload, missing_keys, unexpected_keys


def freeze_module(module: nn.Module) -> None:
    """Freeze all parameters in a module and verify freezing.

    Args:
        module: Module to freeze.
    """
    for parameter in module.parameters():
        parameter.requires_grad = False
    module.eval()
    verify_frozen(module)


def verify_frozen(module: nn.Module) -> bool:
    """Verify all module parameters are frozen.

    Args:
        module: Module to verify.

    Returns:
        ``True`` if all parameters are frozen.
    """
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            raise AssertionError(f"Parameter is not frozen: {name}")
    return True
