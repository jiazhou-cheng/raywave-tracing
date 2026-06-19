import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import math

from deeplens.config import (
    DEFAULT_WAVE,
    PSF_KS,
)
from deeplens.light.ray import Ray

class RayWave:

    def huygens_psf(
        self,
        ray: Ray,
        sensor_pitch: float = None, 
        sensor_grid_size: int = None,
        oversamp: float = 1.5,
    ):
        """
        Compute the Huygens PSF for a ray bundle before tracing to the sensor.
        """
        ray_sensor = self.trace2sensor(ray)
        wvln_um = ray_sensor.wvln.item()
        o = ray_sensor.o.squeeze()       # [N,3]
        device = o.device
        real_dtype = o.dtype
        cplx_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
        N = o.shape[0]
        d = ray_sensor.d.squeeze()       # [N,3]
        opl = ray_sensor.opl.squeeze()   # [N]
        if hasattr(ray_sensor, "amp"):
            amps = ray_sensor.amp.squeeze()
        else:
            amps = torch.ones((N,), dtype=cplx_dtype, device=device)

        # ----- build spatial grid (just for returning coords) -----
        if sensor_pitch is None:
            sensor_pitch = self.pixel_size
        if sensor_grid_size is None:
            sensor_grid_size = self.sensor_res[0]
        pp = sensor_pitch
        ks = sensor_grid_size
        
        xs = (torch.arange(ks, device=self.device, dtype=real_dtype) - (ks - 1) / 2) * pp
        ys = (torch.arange(ks, device=self.device, dtype=real_dtype) - (ks - 1) / 2) * pp
        Xg, Yg = torch.meshgrid(xs, ys, indexing="xy")

        extent = (ks // 2) * pp
        on_sensor = (
            (o[:, 0] >= -extent) & (o[:, 0] <= extent)
            & (o[:, 1] >= -extent) & (o[:, 1] <= extent)
        )
        if not torch.any(on_sensor):
            return Xg, Yg, torch.zeros((ks, ks), dtype=cplx_dtype, device=device), torch.zeros((ks, ks), dtype=real_dtype, device=device)

        o = o[on_sensor]
        d = d[on_sensor]
        opl = opl[on_sensor]
        amps = amps[on_sensor]
        N = o.shape[0]

        # ----- complex weights A_n = amp * exp(i k opl) * exp(-i (kx ox + ky oy)) -----
        k = 2.0 * math.pi / float(wvln_um * 1e-3)
        kx_s = k * d[:, 0]
        ky_s = k * d[:, 1]

        ox, oy = o[:, 0], o[:, 1]
        piston = -(kx_s * ox + ky_s * oy)  # real

        if amps is None:
            amps_c = torch.ones((N,), dtype=cplx_dtype, device=device)
        else:
            amps_c = amps
            if amps_c.ndim > 1:
                amps_c = amps_c.squeeze()
            if not torch.is_complex(amps_c):
                amps_c = amps_c.to(cplx_dtype)
            else:
                amps_c = amps_c.to(cplx_dtype)

        # ----- obliquity / inclination factor <n, d> -----
        # For a flat sensor whose normal is +z:
        sensor_normal = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=real_dtype)

        # Make sure ray directions are normalized
        d_norm = d / torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(1e-12)

        # <n, d_i>
        obliquity = torch.sum(d_norm * sensor_normal[None, :], dim=-1)

        # Optional: remove backward-facing rays
        obliquity = torch.clamp(obliquity, min=0.0)

        phase = torch.exp(
            1j * (k * opl).to(cplx_dtype)
            + 1j * piston.to(cplx_dtype)
        )

        A = amps_c * obliquity.to(cplx_dtype) * phase

        # phase = torch.exp(1j * (k * opl).to(cplx_dtype) + 1j * piston.to(cplx_dtype))
        # A = amps_c * phase  # [N] complex

        # ----- define k-grid (possibly oversampled) -----
        K_os = int(math.ceil(ks * float(oversamp)))

        fx = torch.fft.fftshift(torch.fft.fftfreq(K_os, d=float(pp))).to(device, dtype=real_dtype)
        kx_grid = 2 * math.pi * fx
        dk = float(kx_grid[1] - kx_grid[0])
        kmax = math.pi / float(pp)

        # tighten band a bit so we don't land on last bin
        kmax_eff = kmax - dk

        inband = (kx_s.abs() < kmax_eff) & (ky_s.abs() < kmax_eff)
        if not torch.any(inband):
            return Xg, Yg, torch.zeros((ks, ks), dtype=cplx_dtype, device=device), torch.zeros((ks, ks), dtype=real_dtype, device=device)

        kx_s = kx_s[inband]
        ky_s = ky_s[inband]
        A = A[inband]

        # ----- map to FFTSHIFTED grid indices -----
        center = K_os // 2
        ix_f = (kx_s / dk) + center
        iy_f = (ky_s / dk) + center

        # IMPORTANT: require a valid 4-neighbor footprint: ix0 in [0, K_os-2], iy0 in [0, K_os-2]
        # i.e., ix_f must be in [0, K_os-1) but for bilinear we need <= K_os-2 + 1 => <= K_os-1, and ix0 <= K_os-2
        valid_splat = (ix_f >= 0) & (ix_f <= (K_os - 2)) & (iy_f >= 0) & (iy_f <= (K_os - 2))
        if not torch.any(valid_splat):
            raise ValueError("No valid rays in the band.")

        ix_f = ix_f[valid_splat]
        iy_f = iy_f[valid_splat]
        A = A[valid_splat]

        # ----- bilinear splat into Ehat(kx, ky) -----
        ix0 = torch.floor(ix_f).long()
        iy0 = torch.floor(iy_f).long()
        tx = (ix_f - ix0.to(ix_f.dtype))  # in [0,1)
        ty = (iy_f - iy0.to(iy_f.dtype))

        ix1 = ix0 + 1
        iy1 = iy0 + 1

        w00 = (1 - tx) * (1 - ty)
        w10 = tx * (1 - ty)
        w01 = (1 - tx) * ty
        w11 = tx * ty

        def lin(ix, iy):
            return iy * K_os + ix  # row-major flatten

        idx00 = lin(ix0, iy0)
        idx10 = lin(ix1, iy0)
        idx01 = lin(ix0, iy1)
        idx11 = lin(ix1, iy1)

        Ehat_r = torch.zeros((K_os * K_os,), dtype=real_dtype, device=device)
        Ehat_i = torch.zeros((K_os * K_os,), dtype=real_dtype, device=device)

        Ar = A.real.to(real_dtype)
        Ai = A.imag.to(real_dtype)

        Ehat_r.index_add_(0, idx00, w00 * Ar)
        Ehat_r.index_add_(0, idx10, w10 * Ar)
        Ehat_r.index_add_(0, idx01, w01 * Ar)
        Ehat_r.index_add_(0, idx11, w11 * Ar)

        Ehat_i.index_add_(0, idx00, w00 * Ai)
        Ehat_i.index_add_(0, idx10, w10 * Ai)
        Ehat_i.index_add_(0, idx01, w01 * Ai)
        Ehat_i.index_add_(0, idx11, w11 * Ai)

        Ehat_os = torch.complex(Ehat_r, Ehat_i).view(K_os, K_os)

        # ----- iFFT to spatial field -----
        field_os = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(Ehat_os)))

        # ----- crop back to ks x ks -----
        if K_os != ks:
            c0 = (K_os - ks) // 2
            field = field_os[c0:c0 + ks, c0:c0 + ks].contiguous()
        else:
            field = field_os

        psf = field.abs()**2 

        return Xg, Yg, field, psf
