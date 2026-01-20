import torch


def get_device() -> str:
    """Detect the best available device for model inference."""
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    assert device in ("mps", "cuda", "cpu"), f"Unknown device: {device}"
    return device


def get_dtype(device: str) -> torch.dtype:
    """Get the appropriate dtype for the given device."""
    assert device in ("mps", "cuda", "cpu"), f"Unknown device: {device}"

    if device == "mps":
        return torch.float16
    elif device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    else:
        return torch.float32
