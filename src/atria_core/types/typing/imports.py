TORCH_AVAILABLE = True
try:
    import torch  # type: ignore[import]  # noqa: F401
except ImportError:
    TORCH_AVAILABLE = False
