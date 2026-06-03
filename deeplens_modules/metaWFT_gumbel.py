import torch
import torch.nn.functional as F
import numpy as np

from ..geometric_surface.aspheric import Aspheric
from ..ray import Ray
try:
    from ..basics import EPSILON as EPS
except Exception:
    EPS = 1e-12


# -----------------------
# Utils
# -----------------------
def _fft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered 2D FFT on the last 2 dims; preserves input dtype (complex64/128)."""
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1)
    )

def _grid_sample_fp32(img: torch.Tensor, grid: torch.Tensor, *, align_corners=True) -> torch.Tensor:
    """
    Run grid_sample in float32 (CUDA-friendly), preserve autograd, cast back to img.dtype.

    Args:
        img  : [N,C,H,W], real tensor
        grid : [N,H_out,W_out,2], real tensor in [-1,1]
    """
    out_dtype = img.dtype
    img32  = img.to(torch.float32)
    grid32 = grid.to(torch.float32)
    out32  = F.grid_sample(img32, grid32, mode="bilinear",
                           align_corners=align_corners, padding_mode="border")
    return out32.to(out_dtype)

def _nan_to_num_complex(z: torch.Tensor, nan=0.0, posinf=0.0, neginf=0.0) -> torch.Tensor:
    """nan_to_num for complex tensors by applying to real & imag separately."""
    return torch.complex(
        torch.nan_to_num(z.real, nan=nan, posinf=posinf, neginf=neginf),
        torch.nan_to_num(z.imag, nan=nan, posinf=posinf, neginf=neginf)
    )

def _real_dtype_like(x: torch.Tensor) -> torch.dtype:
    """Return a real dtype matching 'x' precision (float32 for complex64/float32; float64 for complex128/float64)."""
    if x.is_complex():
        return x.real.dtype
    return x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32

def _complex_dtype_from_real(real_dtype: torch.dtype) -> torch.dtype:
    return torch.complex64 if real_dtype == torch.float32 else torch.complex128


class AsphericMetaWFT(Aspheric):
    """
    Aspheric metasurface with discretized *complex* field map on a design plane (⊥ optical axis).

    • Geometry (intersect/normal) inherited from Aspheric.
    • At each hit: build tangent frame (u,v,n), resample a local P×P complex field patch,
      window + zero-pad + FFT → angular spectrum.
    • For spr<=1: use weighted-mean direction; for spr>=2: importance sample via **Gumbel-Softmax**.

    CONVENTION:
      origin_xy = (x0, y0) is the PHYSICAL CENTER of the map.
      Pixel centers are at ((W-1)/2, (H-1)/2) ↔ (x0, y0).
    """

    def __init__(
        self,
        d: float,
        r: float = 5.0,
        c: float = 0.0,
        k: float = 0.0,
        ai=None,
        mat2: str = "air",
        device: str = "cpu",
        # metasurface grid (COMPLEX field expected; phase-only also accepted and converted)
        phase_mat: torch.Tensor | None = None,   # [H,W], complex (preferred) or real (radians)
        dx: float | None = None,                 # mm/px (x)
        dy: float | None = None,                 # mm/px (y)
        origin_xy: tuple[float, float] = (0.0, 0.0),  # (x0,y0) in mm at z=d (design plane), CENTER
        Xg: torch.Tensor | None = None,
        Yg: torch.Tensor | None = None,
        # WFT params
        patch_px: int = 64,                      # local patch size P×P
        pad_factor: int = 4,                     # zero-pad extent ≈ pad_factor*(P-1), odd-enforced
        window: str = "none",                    # "none"|"hann"|"hamming"|"blackman"
        spr: int = 0,                            # samples per ray for MC; 0/1 => mean only
        # Gumbel-Softmax params
        tau: float = 5.0,                        # default temperature
        straight_through: bool = True,           # use ST estimator (argmax hard, soft backward)
    ):
        super().__init__(r=r, d=d, c=c, k=k, ai=ai, mat2=mat2, device=device)
        self.device = torch.device(device)
        self.phase_mat = None
        self.dx, self.dy = dx, dy
        self.origin_xy = (float(origin_xy[0]), float(origin_xy[1]))  # CENTER
        self.patch_px = int(patch_px)
        self.pad_factor = int(pad_factor)
        self.window = window
        self.spr = int(spr)
        self.tau = float(tau)
        self.straight_through = bool(straight_through)
        self._is_complex_field = False  # set in set_grid

        if (Xg is None) != (Yg is None):
            raise ValueError("Both Xg and Yg must be set together, or both must be None.")

        if phase_mat is not None and Xg is None and Yg is None:
            self.set_grid(phase_mat, dx, dy, origin_xy)
        elif phase_mat is not None and Xg is not None and Yg is not None:
            self.Xg, self.Yg = Xg, Yg
            if phase_mat.is_complex():
                self.phase_mat = phase_mat
                self._is_complex_field = True
            else:
                real_dtype = _real_dtype_like(phase_mat)
                cplx_dtype = _complex_dtype_from_real(real_dtype)
                self.phase_mat = torch.exp(1j * phase_mat.to(cplx_dtype))
                self._is_complex_field = True  # now complex

    # ----------------- Grid API -----------------
    def set_grid(self, phase_mat: torch.Tensor, dx: float, dy: float, origin_xy=(0.0, 0.0)):
        """
        Register the metasurface discretized map on the design plane, with (x0,y0) at the CENTER.
        Accepts either:
          • complex field  [H,W] (preferred)
          • real phase     [H,W] (radians) — converted to complex via exp(i*phi)
        """
        assert phase_mat.ndim == 2, "phase_mat must be [H,W]"
        phase_mat = phase_mat.to(self.device)

        if phase_mat.is_complex():
            self.phase_mat = phase_mat
            self._is_complex_field = True
        else:
            real_dtype = _real_dtype_like(phase_mat)
            cplx_dtype = _complex_dtype_from_real(real_dtype)
            self.phase_mat = torch.exp(1j * phase_mat.to(cplx_dtype))
            self._is_complex_field = True  # now complex

        self.Xg, self.Yg = self._xy_axes_centered(phase_mat.shape[0], phase_mat.shape[1])

    def has_grid(self) -> bool:
        return (self.phase_mat is not None) and (self.dx is not None) and (self.dy is not None)

    # ----------------- Centered axes & normalization -----------------
    def _xy_axes_centered(self, H: int, W: int):
        """Return x- and y-axes (in mm) with origin_xy at the CENTER pixel."""
        real_dtype = _real_dtype_like(self.phase_mat.real)
        x0, y0 = self.origin_xy
        dx, dy = self.dx, self.dy
        xs = x0 + (torch.arange(W, device=self.device, dtype=real_dtype) - (W - 1) / 2) * dx
        ys = y0 + (torch.arange(H, device=self.device, dtype=real_dtype) - (H - 1) / 2) * dy
        return xs, ys

    def _xy_to_norm_centered(self, xy: torch.Tensor, H: int, W: int):
        """
        Map physical (x,y) (mm) to grid_sample normalized coords in [-1,1],
        with origin_xy at the CENTER of the image.
        """
        x0, y0 = self.origin_xy
        dx, dy  = self.dx, self.dy
        x = xy[..., 0]
        y = xy[..., 1]
        # pixel coordinates with center at (W-1)/2, (H-1)/2
        I = (x - x0) / dx + (W - 1) / 2
        J = (y - y0) / dy + (H - 1) / 2
        # normalize for grid_sample (align_corners=True)
        In = (I / (W - 1) - 0.5) * 2.0
        Jn = (J / (H - 1) - 0.5) * 2.0
        return In, Jn

    # ----------------- Public pipeline entry -----------------
    def ray_reaction(self, ray, n1=1.0, n2=1.0):
        """
        1) Intersect with asphere
        2) Compute surface normals
        3) Refract via WFT (mean or Gumbel-Softmax MC depending on spr)
        """
        ray = self.intersect(ray, n1)
        n_hat = self.normal_vec(ray)
        return self.refract_wft(ray, n1, n2, n_hat)


    def refract_wft(self, ray, n1: float, n2: float, n_hat: torch.Tensor, *, seed: int | None = None):
        """
        Compute outgoing directions by WFT on patches. For spr<=1: mean direction.
        For spr>=2: Gumbel-Softmax sampling with temperature self.tau (default 5.0).
        """
        assert self.has_grid(), "Call set_grid(phase_mat, dx, dy, origin_xy) first."
        valid = (ray.valid > 0)
        if not torch.any(valid):
            return ray

        # Subset valid rays
        o       = ray.o[valid]                       # [Nv,3] hit points (mm)
        d_in    = ray.d[valid]                       # [Nv,3]
        wvln_um = ray.wvln[valid].squeeze(-1)        # [Nv]
        nh      = F.normalize(n_hat[valid], dim=-1)  # [Nv,3]
        Nv      = o.shape[0]

        # Orient normal consistently with incoming direction
        sign = torch.sign(torch.sum(d_in * nh, dim=-1, keepdim=True))  # [+1 or -1]
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        sign = sign.detach()
        n_loc = nh * sign                                              # [Nv,3]

        # Build tangent frame
        u_hat, v_hat = self._build_tangent_uv(n_loc, d_in)             # [Nv,3] each

        # Local complex patch (tangent aligned)
        U_patch, (du, dv) = self._extract_local_patches(o[..., :2], u_hat, v_hat, self.patch_px)

        # Optional window
        if self.window != "none":
            real_dtype = _real_dtype_like(self.phase_mat.real)
            w2 = self._make_window(self.patch_px, self.window, device=self.device, real_dtype=real_dtype)
            U_patch = U_patch * w2.unsqueeze(0).to(U_patch.dtype)  # [Nv,P,P]

        # Zero-pad
        P = self.patch_px
        pad_size = int(self.pad_factor * max(0, P - 1))
        if pad_size % 2 == 0:
            pad_size += 1
        tpad = (pad_size - P)
        pad_before = tpad // 2
        pad_after  = tpad - pad_before
        U_pad = F.pad(U_patch, (pad_before, pad_after, pad_before, pad_after))  # [Nv, Ppad, Ppad]
        _, Ppad, _ = U_pad.shape

        # Frequency axes in cycles/mm along local (u,v)
        real_dtype = _real_dtype_like(self.phase_mat.real)
        fu = torch.fft.fftshift(torch.fft.fftfreq(Ppad, d=du)).to(self.device, dtype=real_dtype)  # [Nv]?→broadcast OK if du scalar
        fv = torch.fft.fftshift(torch.fft.fftfreq(Ppad, d=dv)).to(self.device, dtype=real_dtype)
        # If du,dv are tensors, above creates per-ray vectors; we need shared grids -> use means for speed:
        if torch.is_tensor(du):
            fu = torch.fft.fftshift(torch.fft.fftfreq(Ppad, d=du.mean())).to(self.device, dtype=real_dtype)
        if torch.is_tensor(dv):
            fv = torch.fft.fftshift(torch.fft.fftfreq(Ppad, d=dv.mean())).to(self.device, dtype=real_dtype)

        Fu, Fv = torch.meshgrid(fu, fv, indexing="xy")                                            # [Ppad,Ppad]

        # Angular spectrum & power
        spec  = _fft2c(U_pad)                           # [Nv,Ppad,Ppad], complex
        mag   = torch.abs(spec)
        power = (mag ** 2)                              # [Nv,Ppad,Ppad] ≥0
        power = torch.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize per-ray (pre-mask)
        sumP = power.sum(dim=(1, 2), keepdim=True)      # [Nv,1,1]
        dead0 = (sumP <= 0)
        if dead0.any():
            cy = Ppad // 2; cx = Ppad // 2
            power[dead0.expand_as(power)] = 0.0
            power[dead0.squeeze(-1).squeeze(-1), cy, cx] = 1.0
            sumP = power.sum(dim=(1, 2), keepdim=True)
        power = power / (sumP + EPS)

        # -------- Add incident transverse k offset (Snell frame) --------
        # k0 in 1/mm
        wvln_mm = (wvln_um * 1e-3).to(real_dtype).clamp_min(1e-12)  # [Nv]
        k0 = (2 * torch.pi) / wvln_mm                               # [Nv]
        if torch.is_tensor(n1):
            n1v = n1[valid].reshape(-1).to(real_dtype)
        else:
            n1v = torch.full_like(wvln_mm, float(n1))
        if torch.is_tensor(n2):
            n2v = n2[valid].reshape(-1).to(real_dtype)
        else:
            n2v = torch.full_like(wvln_mm, float(n2))

        # Incident transverse k (local u,v)
        du_in = torch.sum(d_in * u_hat, dim=-1)    # [Nv]
        dv_in = torch.sum(d_in * v_hat, dim=-1)    # [Nv]
        Ku0 = (n1v * k0 * du_in).view(Nv, 1, 1)    # [Nv,1,1]
        Kv0 = (n1v * k0 * dv_in).view(Nv, 1, 1)    # [Nv,1,1]

        # Exit medium total k magnitude
        k_exit = (n2v * k0).view(Nv, 1, 1)         # [Nv,1,1]

        # Total transverse grids (per-ray offset)
        Kx_grid = Ku0 + (2 * torch.pi * Fu).unsqueeze(0)  # [Nv,Ppad,Ppad]
        Ky_grid = Kv0 + (2 * torch.pi * Fv).unsqueeze(0)  # [Nv,Ppad,Ppad]

        # Evanescent mask
        mask = (Kx_grid**2 + Ky_grid**2) < (k_exit**2)    # [Nv,Ppad,Ppad]
        power = power * mask
        power = torch.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)
        power = torch.relu(power)
        sumP  = power.sum(dim=(1, 2), keepdim=True)
        dead  = (sumP <= 0)
        if dead.any():
            cy = Ppad // 2; cx = Ppad // 2
            power[dead.expand_as(power)] = 0.0
            power[dead.squeeze(-1).squeeze(-1), cy, cx] = 1.0
            sumP = power.sum(dim=(1, 2), keepdim=True)
        power = power / (sumP + EPS)  # final normalized pdf per ray

        # ============== spr <= 1 : weighted-mean direction (deterministic) ==============
        if self.spr <= 1:
            Kx_mean = (power * Kx_grid).sum(dim=(1, 2))            # [Nv]
            Ky_mean = (power * Ky_grid).sum(dim=(1, 2))            # [Nv]

            k_exit_v = k_exit.view(Nv)                             # [Nv]
            du_out = Kx_mean / (k_exit_v + EPS)                    # [Nv]
            dv_out = Ky_mean / (k_exit_v + EPS)                    # [Nv]

            rho2 = (du_out**2 + dv_out**2).clamp(0.0, 1.0 - 1e-12)
            dw_out = torch.sqrt(1.0 - rho2)

            d_out = (
                du_out.unsqueeze(-1) * u_hat +
                dv_out.unsqueeze(-1) * v_hat +
                dw_out.unsqueeze(-1) * n_loc
            )
            d_out = d_out / (torch.norm(d_out, dim=-1, keepdim=True) + EPS)
            d_out = torch.nan_to_num(d_out, nan=0.0, posinf=0.0, neginf=0.0)

            # Write back
            ray.d[valid] = d_out

            # OPL tweak from center field (keeps coherence)
            if ray.coherent:
                center = self._sample_field_at(o[..., :2])     # complex [Nv]
                center = _nan_to_num_complex(center)
                center_phi = torch.angle(center)
                opd_mm = (center_phi * (wvln_um * 1e-3) / (2 * torch.pi)).view(-1, 1)
                ray.opl[valid] = ray.opl[valid] + opd_mm

            return ray

        # ============== spr >= 2 : Gumbel-Softmax sampling (differentiable) ==============
        spr = int(max(2, self.spr))
        HW = Ppad * Ppad

        # Build sampling logits from |spec| with mask; use log to expand dynamic range
        mag = torch.abs(spec)                                  # [Nv,Ppad,Ppad]
        samp = mag * mask                                      # masked magnitude
        samp = torch.nan_to_num(samp, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize to avoid scale issues; any monotone rescale is OK for softmax
        Z = samp.sum(dim=(1, 2), keepdim=True).clamp_min(1e-30)
        pdf = samp / Z
        tiny = torch.tensor(1e-30, dtype=real_dtype, device=self.device)
        logits = torch.log(pdf + tiny)                         # [Nv,Ppad,Ppad]
        logits = logits.reshape(Nv, HW)                        # [Nv,HW]

        # Repeat for S = Nv * spr samples
        S = Nv * spr
        logits_exp = logits.repeat_interleave(spr, dim=0)      # [S,HW]

        # Gumbel noise
        if seed is not None:
            g_state = torch.random.get_rng_state()
            torch.manual_seed(int(seed))
        U = torch.rand((S, HW), device=self.device, dtype=real_dtype).clamp_min(1e-9)
        G = -torch.log(-torch.log(U))                          # [S,HW]
        if seed is not None:
            torch.random.set_rng_state(g_state)

        # Soft sample
        y_soft = F.softmax((logits_exp + G) / max(self.tau, 1e-6), dim=-1)  # [S,HW]

        # Straight-through: hard in forward, soft in backward
        if self.straight_through:
            idx = y_soft.argmax(dim=-1)                        # [S]
            y_hard = torch.zeros_like(y_soft).scatter_(1, idx.unsqueeze(1), 1.0)
            y = y_hard + (y_soft - y_soft.detach())
        else:
            y = y_soft

        # Flatten frequency grids & spectra
        Fu_f = Fu.reshape(-1)          # [HW] cycles/mm
        Fv_f = Fv.reshape(-1)          # [HW]
        Kx_bins = (2 * torch.pi) * Fu_f  # [HW] rad/mm
        Ky_bins = (2 * torch.pi) * Fv_f  # [HW]

        spec_flat = spec.reshape(Nv, HW)                     # complex
        power_flat = power.reshape(Nv, HW)                   # used only if needed

        # Expand per sample
        spec_exp = spec_flat.repeat_interleave(spr, dim=0)   # [S,HW]
        # Expected complex amplitude per sample (soft selection)
        spec_s = (y * spec_exp.real).sum(dim=1) + 1j * (y * spec_exp.imag).sum(dim=1)  # [S] complex

        # Expected pdf value at the soft pick (for importance weight)
        pdf_flat = pdf.reshape(Nv, HW)
        pdf_exp = pdf_flat.repeat_interleave(spr, dim=0)     # [S,HW]
        pdf_s = (y * pdf_exp).sum(dim=1).clamp_min(1e-20)    # [S]

        # Sample transverse k via soft expectation of bins
        Kx_s = (y * Kx_bins.unsqueeze(0)).sum(dim=1)         # [S]
        Ky_s = (y * Ky_bins.unsqueeze(0)).sum(dim=1)         # [S]

        # Add incident offsets (per parent ray)
        Ku0_s = Ku0.view(Nv).repeat_interleave(spr)          # [S]
        Kv0_s = Kv0.view(Nv).repeat_interleave(spr)          # [S]
        Kx_s = Kx_s + Ku0_s
        Ky_s = Ky_s + Kv0_s

        # Convert to direction cosines in exit medium
        k_exit_s = k_exit.view(Nv).repeat_interleave(spr)    # [S]
        du_s = Kx_s / (k_exit_s + EPS)
        dv_s = Ky_s / (k_exit_s + EPS)
        rho2 = (du_s**2 + dv_s**2).clamp(0.0, 1.0 - 1e-12)
        dw_s = torch.sqrt(1.0 - rho2)

        # World directions
        u_rep = u_hat.view(Nv,1,3).repeat_interleave(spr, dim=1).reshape(-1,3)  # [S,3]
        v_rep = v_hat.view(Nv,1,3).repeat_interleave(spr, dim=1).reshape(-1,3)
        n_rep = n_loc.view(Nv,1,3).repeat_interleave(spr, dim=1).reshape(-1,3)

        d_world = (du_s.unsqueeze(-1) * u_rep +
                   dv_s.unsqueeze(-1) * v_rep +
                   dw_s.unsqueeze(-1) * n_rep)
        d_world = d_world / (torch.norm(d_world, dim=-1, keepdim=True) + EPS)

        # Complex amplitudes: unbiased w.r.t. ∑ spec Δk with PDF ∝ |spec|
        # Keep spatial ΔxΔy factor (du*dv); divide by spr to maintain energy across fan-out
        if torch.is_tensor(du):
            du_eff = du.mean()
        else:
            du_eff = du
        if torch.is_tensor(dv):
            dv_eff = dv.mean()
        else:
            dv_eff = dv
        amps = spec_s * ((du_eff * dv_eff) / pdf_s) / spr    # [S] complex

        # Build new fan-out Ray
        o_s      = o.unsqueeze(1).repeat(1, spr, 1).reshape(-1, 3)                 # [S,3]
        wvln_s   = wvln_um.view(Nv, 1, 1).repeat(1, spr, 1).reshape(-1, 1)         # [S,1]
        parent_opl = ray.opl[valid].view(Nv, 1, 1).repeat(1, spr, 1).reshape(-1,1) # [S,1]
        parent_en  = ray.en [valid].view(Nv, 1, 1).repeat(1, spr, 1).reshape(-1,1) # [S,1]

        new_ray = Ray(o_s, d_world, wvln=wvln_s, coherent=(ray.coherent or True), device=self.device)
        new_ray.valid      = torch.ones(o_s.shape[0], device=self.device)
        new_ray.en         = parent_en
        new_ray.opl        = parent_opl
        new_ray.is_forward = (new_ray.d[..., 2].unsqueeze(-1) > 0)

        # Carry complex amplitudes + lineage
        new_ray.amp       = amps.reshape(-1)  # complex, [S]
        new_ray.parent_ix = torch.arange(Nv, device=self.device).repeat_interleave(spr)

        return new_ray

    # -------- Provide direct access to the MC bundle (no ray mutation) --------
    @torch.no_grad()
    def scatter_wft_samples(self, ray, n1: float, n2: float, spr: int | None = None, *, seed: int | None = None):
        """
        Return Monte-Carlo samples (origins, dirs, amps) for the *currently valid* rays.
        Does not modify 'ray'. If spr is None, uses self.spr (forces spr>=2).
        """
        spr_prev = self.spr
        if spr is not None:
            self.spr = int(spr)
        tmp_ray = type("Tmp", (), {})()
        # shallow copy minimal fields used
        tmp_ray.o = ray.o.clone()
        tmp_ray.d = ray.d.clone()
        tmp_ray.wvln = ray.wvln.clone()
        tmp_ray.valid = ray.valid.clone()
        tmp_ray.coherent = getattr(ray, "coherent", False)
        tmp_ray.opl = getattr(ray, "opl", torch.zeros_like(ray.wvln))

        tmp_ray = self.intersect(tmp_ray, n1)
        n_hat = self.normal_vec(tmp_ray)

        # Ensure we build a bundle
        if self.spr <= 1:
            self.spr = 2
        tmp_ray = self.refract_wft(tmp_ray, n1, n2, n_hat, seed=seed)
        bundle = getattr(tmp_ray, "meta", {}).get("wft_bundle", None)
        self.spr = spr_prev
        return bundle

    # ----------------- Helpers -----------------
    def _build_tangent_uv(self, n_hat: torch.Tensor, d_in: torch.Tensor):
        """Stable tangent frame: û ≈ proj(-d_in) onto tangent plane; v̂ = n̂×û."""
        n = F.normalize(n_hat, dim=-1)  # (N,3)
        # Project -d_in onto tangent plane
        t = (-d_in) - (n * torch.sum(-d_in * n, dim=-1, keepdim=True))
        t_norm = torch.norm(t, dim=-1, keepdim=True)
        fallback = torch.tensor([1.0, 0.0, 0.0], device=self.device)  # (3,)
        t = torch.where(t_norm < 1e-12, fallback.expand_as(t), t)
        u = F.normalize(t, dim=-1)
        v = F.normalize(torch.cross(n, u, dim=-1), dim=-1)
        u = F.normalize(torch.cross(v, n, dim=-1), dim=-1)  # re-orthogonalize
        return u, v

    def _extract_local_patches(self, o_xy: torch.Tensor, u_hat: torch.Tensor, v_hat: torch.Tensor, P: int):
        """
        Tangent-aligned P×P *complex field* patch centered at each hit.
        Returns: patch [Nv,P,P] complex, (du,dv) = (dx,dy)
        """
        assert self.phase_mat is not None and self._is_complex_field
        H, W = self.phase_mat.shape
        dx, dy = self.dx, self.dy
        Nv = o_xy.shape[0]

        du, dv = dx, dy
        half_u = (P - 1) * 0.5 * du
        half_v = (P - 1) * 0.5 * dv

        real_dtype = _real_dtype_like(self.phase_mat.real)
        u_lin = torch.linspace(-half_u, half_u, P, device=self.device, dtype=real_dtype)
        v_lin = torch.linspace(-half_v, half_v, P, device=self.device, dtype=real_dtype)
        V_rows, U_cols = torch.meshgrid(v_lin, u_lin, indexing="ij")  # [P,P]
        Ue = U_cols.view(1, P, P, 1)  # columns vary with u (x)
        Ve = V_rows.view(1, P, P, 1)  # rows    vary with v (y)
        u_e = u_hat.view(Nv, 1, 1, 3)
        v_e = v_hat.view(Nv, 1, 1, 3)

        # World XY (mm) from tangent displacements around o_xy
        XY = o_xy.view(Nv, 1, 1, 2) + (Ue * u_e[..., :2] + Ve * v_e[..., :2])  # [Nv,P,P,2]

        # Centered pixel coords → normalized for grid_sample
        In, Jn = self._xy_to_norm_centered(XY, H, W)
        grid = torch.stack([In, Jn], dim=-1)  # [Nv,P,P,2]

        img_r = self.phase_mat.real.view(1, 1, H, W)
        img_i = self.phase_mat.imag.view(1, 1, H, W)
        samp_r = _grid_sample_fp32(img_r.expand(Nv, 1, H, W), grid, align_corners=True).squeeze(1)
        samp_i = _grid_sample_fp32(img_i.expand(Nv, 1, H, W), grid, align_corners=True).squeeze(1)
        patch  = torch.complex(samp_r.to(self.phase_mat.real.dtype),
                               samp_i.to(self.phase_mat.real.dtype))  # [Nv,P,P]
        patch  = _nan_to_num_complex(patch, nan=0.0, posinf=0.0, neginf=0.0)
        return patch, (du, dv)

    def _sample_field_at(self, xy: torch.Tensor) -> torch.Tensor:
        """Sample complex field at scattered (x,y), with origin_xy as CENTER."""
        H, W = self.phase_mat.shape
        In, Jn = self._xy_to_norm_centered(xy, H, W)
        grid = torch.stack([In, Jn], dim=-1).view(1, -1, 1, 2)

        img_r = self.phase_mat.real.view(1, 1, H, W)
        img_i = self.phase_mat.imag.view(1, 1, H, W)
        out_r = _grid_sample_fp32(img_r, grid, align_corners=True).view(-1)
        out_i = _grid_sample_fp32(img_i, grid, align_corners=True).view(-1)
        out   = torch.complex(out_r.to(self.phase_mat.real.dtype),
                              out_i.to(self.phase_mat.real.dtype))
        return _nan_to_num_complex(out, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _make_window(P: int, kind: str, device="cpu", real_dtype=torch.float32):
        """Real 2D apodization window with dtype tied to 'real_dtype'."""
        t = torch.linspace(0, 1, P, device=device, dtype=real_dtype)
        if kind == "hann":
            w = 0.5 - 0.5 * torch.cos(2 * torch.pi * t)
        elif kind == "hamming":
            w = 0.54 - 0.46 * torch.cos(2 * torch.pi * t)
        elif kind == "blackman":
            w = 0.42 - 0.5 * torch.cos(2 * torch.pi * t) + 0.08 * torch.cos(4 * torch.pi * t)
        else:
            return torch.ones((P, P), device=device, dtype=real_dtype)
        return (w[:, None] * w[None, :])

    # ----------------- Serialization -----------------
    @classmethod
    def init_from_dict(cls, surf_dict):
        d = surf_dict["d"]
        r = surf_dict.get("r", 5.0)
        c = surf_dict.get("c", 0.0)
        k = surf_dict.get("k", 0.0)
        ai = surf_dict.get("ai", None)
        mat2 = surf_dict.get("mat2", "air")
        device = surf_dict.get("device", "cpu")
        phase = surf_dict.get("phase_mat", None)
        dx = surf_dict.get("dx", None)
        dy = surf_dict.get("dy", None)
        origin_xy = tuple(surf_dict.get("origin_xy", (0.0, 0.0)))  # CENTER
        patch_px = int(surf_dict.get("patch_px", 64))
        pad_factor = int(surf_dict.get("pad_factor", 4))
        window = surf_dict.get("window", "hamming")
        spr = int(surf_dict.get("spr", 0))
        tau = float(surf_dict.get("tau", 5.0))
        straight_through = bool(surf_dict.get("straight_through", True))
        return cls(d=d, r=r, c=c, k=k, ai=ai, mat2=mat2, device=device,
                   phase_mat=phase, dx=dx, dy=dy, origin_xy=origin_xy,
                   patch_px=patch_px, pad_factor=pad_factor, window=window, spr=spr,
                   tau=tau, straight_through=straight_through)

    def surf_dict(self):
        base = super().surf_dict()
        base.update({
            "type": "AsphericMetaWFT",
            "dx": self.dx,
            "dy": self.dy,
            "origin_xy": self.origin_xy,   # CENTER
            "patch_px": self.patch_px,
            "pad_factor": self.pad_factor,
            "window": self.window,
            "spr": self.spr,
            "tau": self.tau,
            "straight_through": self.straight_through,
        })
        return base

    def draw_widget(self, ax, color="black", linestyle="-"):
        super().draw_widget(ax, color, linestyle)
        # Plot the map extent with origin_xy at the CENTER
        if self.phase_mat is not None:
            H, W = self.phase_mat.shape
            xs, ys = self._xy_axes_centered(H, W)
            extent = [float(xs[0] - self.dx * 0.5), float(xs[-1] + self.dx * 0.5),
                      float(ys[0] - self.dy * 0.5), float(ys[-1] + self.dy * 0.5)]
            ax.imshow(torch.angle(self.phase_mat).detach().cpu().numpy(),
                      extent=extent, origin='lower', cmap='twilight', alpha=0.25)
        d_cpu = float(self.d.cpu()) if hasattr(self.d, "cpu") else float(self.d)
        ax.plot([d_cpu - 0.1, d_cpu + 0.1], [0, 0], color=color, linestyle=linestyle, linewidth=1.5)
