from __future__ import annotations

import os
import numpy as np
import torch


def load_phase_npy(path: str, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.load(path)).to(device=device, dtype=dtype)


def load_target_image(path: str, *, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    try:
        from scipy.ndimage import zoom
    except Exception as exc:
        raise RuntimeError("scipy is required for target resizing.") from exc

    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".npy":
        image = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        key = next((k for k in ("target", "arr_0") if k in data.files), data.files[0])
        image = data[key]
    else:
        import matplotlib.image as mpimg

        image = mpimg.imread(path)

    image = np.asarray(image)
    if image.ndim == 3:
        image = image[..., :3].mean(axis=-1)
    if image.shape[-2:] != (size, size):
        image = zoom(image, (size / image.shape[0], size / image.shape[1]), order=1)
    image = image.astype(np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image -= image.min()
    image /= image.max() + 1e-12
    return torch.from_numpy(image).to(device=device, dtype=dtype)


def save_intensity_png(field_sum: torch.Tensor, path: str) -> None:
    import matplotlib.pyplot as plt

    image = field_sum.detach().abs().float().cpu().numpy() ** 2
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=500)
    plt.close()


def save_phase_png(phase: torch.Tensor, path: str) -> None:
    import matplotlib.pyplot as plt

    image = torch.remainder(phase.detach(), 2 * torch.pi).cpu().numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="twilight", vmin=0.0, vmax=2 * np.pi)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=500)
    plt.close()


def save_loss_plot(loss_history: list[float], lr_history: list[float], path: str) -> None:
    import matplotlib.pyplot as plt

    if not loss_history:
        return
    epochs = np.arange(len(loss_history))
    fig, ax_loss = plt.subplots(figsize=(7, 4))
    ax_loss.plot(epochs, loss_history, linewidth=1.5, color="C0")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss", color="C0")
    ax_loss.tick_params(axis="y", labelcolor="C0")
    ax_loss.grid(True, alpha=0.3)
    if len(lr_history) == len(loss_history):
        ax_lr = ax_loss.twinx()
        ax_lr.plot(epochs, lr_history, linewidth=1.2, color="C1")
        ax_lr.set_ylabel("learning rate", color="C1")
        ax_lr.tick_params(axis="y", labelcolor="C1")
    fig.tight_layout()
    plt.savefig(path, dpi=500)
    plt.close()
