import torch
import numpy as np

def img_crop(Xg: torch.Tensor, Yg: torch.Tensor, img: torch.Tensor, crop_size: int = 101) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Global peak in full image
    max_idx = torch.tensor(np.unravel_index(img.argmax().item(), img.shape))
    cy, cx = max_idx.tolist()
    half = crop_size // 2

    # Crop around global peak
    y0 = max(cy - half, 0)
    x0 = max(cx - half, 0)
    y1 = min(y0 + crop_size, img.shape[0])
    x1 = min(x0 + crop_size, img.shape[1])
    y0 = max(y1 - crop_size, 0)
    x0 = max(x1 - crop_size, 0)

    img_c = img[y0:y1, x0:x1]
    Xg_c = Xg[y0:y1, x0:x1]
    Yg_c = Yg[y0:y1, x0:x1]

    # Peak inside cropped image
    peak_c = torch.tensor(np.unravel_index(img_c.argmax().item(), img_c.shape))
    py, px = peak_c.tolist()

    center_x_mm = Xg_c[py, px].detach().cpu().item()
    center_y_mm = Yg_c[py, px].detach().cpu().item()

    print(f"PSF center (um): x = {center_x_mm * 1000:.2f}, y = {center_y_mm * 1000:.2f}")

    return Xg_c, Yg_c, img_c