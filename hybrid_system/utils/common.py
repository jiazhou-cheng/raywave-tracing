import torch
from typing import Optional, Tuple

def set_step_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def fft2c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))


def nan_to_num_complex(z: torch.Tensor) -> torch.Tensor:
    return torch.complex(torch.nan_to_num(z.real), torch.nan_to_num(z.imag))


def complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
    return torch.complex64 if real_dtype == torch.float32 else torch.complex128


def empty_ray_bundle(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cplx_dtype = complex_dtype_for(dtype)
    return (
        torch.zeros((0, 3), device=device, dtype=dtype),
        torch.zeros((0, 3), device=device, dtype=dtype),
        torch.zeros((0,), device=device, dtype=cplx_dtype),
        torch.zeros((0,), device=device, dtype=dtype),
    )


def generate_collimated_rays(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    radius_mm: float,
    num_rays: int,
    z_start_mm: float,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    set_step_seed(seed)
    x0, y0 = sample_ray_origins_pixel_centers(X, Y, radius_mm, num_rays)
    z0 = torch.full_like(x0, float(z_start_mm))
    o = torch.stack([x0, y0, z0], dim=-1)
    d = torch.zeros_like(o)
    d[:, 2] = 1.0
    return o, d


@torch.no_grad()
def sample_ray_origins_pixel_centers(
    X: torch.Tensor,
    Y: torch.Tensor,
    radius_mm: float,
    num_rays: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xf = X.reshape(-1)
    yf = Y.reshape(-1)
    inside = xf * xf + yf * yf <= float(radius_mm) ** 2
    if not torch.any(inside):
        raise RuntimeError("No grid samples are inside the aperture.")
    idx_all = torch.nonzero(inside, as_tuple=False).squeeze(-1)
    if num_rays <= idx_all.numel():
        idx = idx_all[torch.randperm(idx_all.numel(), device=idx_all.device)[:num_rays]]
    else:
        idx = idx_all[torch.randint(0, idx_all.numel(), (num_rays,), device=idx_all.device)]
    return xf[idx], yf[idx]