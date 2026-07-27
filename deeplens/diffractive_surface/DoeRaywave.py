""" RayWave tracing at DOE surface, non-differentiable version.

The surface stores a complex DOE field on a physical design plane built on Aspheric. At each ray
intersection, a local tangent-aligned field patch is Fourier transformed to an
angular spectrum and converted into an outgoing ray direction.
"""

import torch
import torch.nn.functional as F

from deeplens.geometric_surface.aspheric import Aspheric
from deeplens.light.ray import Ray

try:
    from deeplens.config import EPSILON as EPS
except Exception:
    EPS = 1e-12


def _fft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered 2D FFT over the last two dimensions."""
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )


def _grid_sample_fp32(
    img: torch.Tensor,
    grid: torch.Tensor,
    *,
    align_corners=True,
) -> torch.Tensor:
    """Run grid_sample in float32, then cast back to the input dtype."""
    out_dtype = img.dtype
    img32 = img.to(torch.float32)
    grid32 = grid.to(torch.float32)
    out32 = F.grid_sample(
        img32,
        grid32,
        mode="bilinear",
        align_corners=align_corners,
        padding_mode="zeros",
    )
    return out32.to(out_dtype)


def _nan_to_num_complex(z: torch.Tensor, nan=0.0, posinf=0.0, neginf=0.0) -> torch.Tensor:
    """Apply nan_to_num to the real and imaginary parts of a complex tensor."""
    return torch.complex(
        torch.nan_to_num(z.real, nan=nan, posinf=posinf, neginf=neginf),
        torch.nan_to_num(z.imag, nan=nan, posinf=posinf, neginf=neginf),
    )


def _real_dtype_like(x: torch.Tensor) -> torch.dtype:
    """Return the matching real dtype for a real or complex tensor."""
    if x.is_complex():
        return x.real.dtype
    return x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32


def _complex_dtype_from_real(real_dtype: torch.dtype) -> torch.dtype:
    """Return the matching complex dtype for a real dtype."""
    return torch.complex64 if real_dtype == torch.float32 else torch.complex128


class DoeRaywave(Aspheric):
    """
    Aspheric DOE surface using a local RayWave/WFT angular-spectrum model.

    The DOE field is registered as a complex map on a plane perpendicular to
    the optical axis. Pixel centers are defined so that ``origin_xy`` is the
    physical center of the map, corresponding to image index
    ``((W - 1) / 2, (H - 1) / 2)``.
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
        field: torch.Tensor | None = None,
        dx: float | None = None,
        dy: float | None = None,
        origin_xy: tuple[float, float] = (0.0, 0.0),
        Xg: torch.Tensor | None = None,
        Yg: torch.Tensor | None = None,
        patch_px: int = 64,
        pad_factor: int = 4,
        window: str = "none",
        spr: int = 0,
        pad_field_for_patches: bool = False,
    ):
        """Initialize a RayWave DOE surface.

        Args:
            d (float): Surface distance. [mm]
            r (float): Aperture radius. [mm]
            c (float): Curvature. [1/mm]
            k (float): Conic constant.
            ai (list): Aspheric coefficients [a2, a4, a6, ...].
            mat2 (str): Material after the surface.
            device (str): Device for tensors and ray tracing.
            field (Tensor): Complex field map [H, W], or phase map in radians.
            dx (float): DOE pixel pitch along x. [mm/pixel]
            dy (float): DOE pixel pitch along y. [mm/pixel]
            origin_xy (tuple): Physical map center on the design plane. [mm]
            Xg (Tensor): Optional precomputed x-axis samples. [mm]
            Yg (Tensor): Optional precomputed y-axis samples. [mm]
            patch_px (int): Local WFT patch size in pixels.
            pad_factor (int): Zero-padding factor for angular-spectrum sampling.
            window (str): Apodization window: none, hann, hamming, or blackman.
            spr (int): Samples per ray. Values <= 1 use the mean direction.
            pad_field_for_patches (bool): If True, pad the DOE field to a
                patch-aligned grid plus a half-patch guard band and enlarge
                the ray-hit aperture radius to the padded field half-extent.
        """
        super().__init__(r=r, d=d, c=c, k=k, ai=ai, mat2=mat2, device=device)
        self.device = torch.device(device)
        self.field = None
        self.dx, self.dy = dx, dy
        self.origin_xy = (float(origin_xy[0]), float(origin_xy[1]))
        self.patch_px = int(patch_px)
        self.pad_factor = int(pad_factor)
        self.window = window
        self.spr = int(spr)
        self.pad_field_for_patches = bool(pad_field_for_patches)
        self.design_r = float(r)
        self._is_complex_field = False

        if self.patch_px <= 0:
            raise ValueError("patch_px must be positive.")

        if (Xg is None) != (Yg is None):
            raise ValueError("Both Xg and Yg must be set together, or both must be None.")

        if field is not None and Xg is None and Yg is None:
            self.set_grid(field, dx, dy, origin_xy)
        elif field is not None and Xg is not None and Yg is not None:
            self.Xg, self.Yg = Xg, Yg
            if field.is_complex():
                self.field = field.to(self.device)
                self._is_complex_field = True
            else:
                real_dtype = _real_dtype_like(field)
                cplx_dtype = _complex_dtype_from_real(real_dtype)
                self.field = torch.exp(1j * field.to(self.device, dtype=cplx_dtype))
                self._is_complex_field = True
            self._update_patch_encoded_radius()

    def set_grid(self, field: torch.Tensor, dx: float, dy: float, origin_xy=(0.0, 0.0)):
        """Register a complex field map or a real phase map on the design plane."""
        if field.ndim != 2:
            raise ValueError("field must have shape [H, W].")
        if dx is None or dy is None:
            raise ValueError("dx and dy must be provided when setting the DOE grid.")

        self.dx = float(dx)
        self.dy = float(dy)
        self.origin_xy = (float(origin_xy[0]), float(origin_xy[1]))
        field = field.to(self.device)

        if field.is_complex():
            complex_field = field
            self._is_complex_field = True
        else:
            real_dtype = _real_dtype_like(field)
            cplx_dtype = _complex_dtype_from_real(real_dtype)
            complex_field = torch.exp(1j * field.to(cplx_dtype))
            self._is_complex_field = True

        if self.pad_field_for_patches:
            complex_field = self._pad_field_for_patch_centers(complex_field)

        self.field = complex_field
        self.Xg, self.Yg = self._xy_axes_centered(complex_field.shape[0], complex_field.shape[1])
        self._update_patch_encoded_radius()

    def has_grid(self) -> bool:
        """Return True when the DOE field and physical sampling are available."""
        return (self.field is not None) and (self.dx is not None) and (self.dy is not None)

    def _pad_field_for_patch_centers(self, field: torch.Tensor) -> torch.Tensor:
        """
        Pad the field so valid ray-hit centers uniformly cover the original
        field plus a half-patch guard band.

        The inner padding makes both dimensions a multiple of patch_px. The
        outer guard band lets a P x P WFT patch be centered at hits that lie
        just outside the original design radius without clipping the sampled
        field asymmetrically.
        """
        H, W = field.shape
        P = self.patch_px
        pad = P

        H_mul = ((H + P - 1) // P) * P
        W_mul = ((W + P - 1) // P) * P
        pad_h = H_mul - H
        pad_w = W_mul - W
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        z = F.pad(field, (pad, pad, pad, pad), mode="constant", value=0)

        # margin = P // 2
        # if margin > 0:
        #     z = F.pad(z, (margin, margin, margin, margin), mode="constant", value=0)
        return z

    def _update_patch_encoded_radius(self):
        """Use the padded field half-extent as the ray-hit aperture radius."""
        if not self.has_grid():
            return
        H, W = self.field.shape
        rx = W * self.dx * 0.5
        ry = H * self.dy * 0.5
        self.r = float(min(rx, ry))

    def _xy_axes_centered(self, H: int, W: int):
        """Return physical x and y axes using edge-inclusive convention."""
        real_dtype = _real_dtype_like(self.field.real)
        x0, y0 = self.origin_xy
        dx, dy = self.dx, self.dy

        xs = x0 + (torch.arange(W + 1, device=self.device, dtype=real_dtype) - W / 2) * dx
        ys = y0 + (torch.arange(H + 1, device=self.device, dtype=real_dtype) - H / 2) * dy

        return xs, ys

    def _xy_to_norm_centered(self, xy: torch.Tensor, H: int, W: int):
        """Map physical x-y coordinates to grid_sample coordinates in [-1, 1]."""
        x0, y0 = self.origin_xy
        dx, dy = self.dx, self.dy
        x = xy[..., 0]
        y = xy[..., 1]
        I = (x - x0) / dx + (W - 1) / 2
        J = (y - y0) / dy + (H - 1) / 2
        In = (I / (W - 1) - 0.5) * 2.0
        Jn = (J / (H - 1) - 0.5) * 2.0
        return In, Jn

    def ray_reaction(self, ray, n1=1.0, n2=1.0):
        """Intersect the asphere, then apply RayWave/WFT refraction."""
        ray = self.intersect(ray, n1)
        n_hat = self.normal_vec(ray)
        return self.refract_wft(ray, n1, n2, n_hat)

    def refract_wft(self, ray, n1: float, n2: float, n_hat: torch.Tensor):
        """Refract rays by converting local DOE patches into angular spectra."""
        assert self.has_grid(), "Call set_grid(field, dx, dy, origin_xy) first."
        n_slots = ray.o[..., 0].numel()
        valid_1d = (ray.is_valid > 0).reshape(n_slots)
        if not torch.any(valid_1d):
            return ray

        o = ray.o.reshape(-1, 3)[valid_1d]
        d_in = ray.d.reshape(-1, 3)[valid_1d]
        wvln_um = ray.wvln.item()
        nh = F.normalize(n_hat.view(-1, 3)[valid_1d], dim=-1)
        Nv = o.shape[0]

        sign = torch.sign(torch.sum(d_in * nh, dim=-1, keepdim=True))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        sign = sign.detach()
        n_loc = nh * sign

        # Sample each field patch in a local tangent frame.
        u_hat, v_hat = self._build_tangent_uv(n_loc, d_in)
        U_patch, (du, dv) = self._extract_local_patches(o[..., :2], u_hat, v_hat, self.patch_px)

        if self.window != "none":
            real_dtype = _real_dtype_like(self.field.real)
            w2 = self._make_window(self.patch_px, self.window, device=self.device, real_dtype=real_dtype)
            U_patch = U_patch * w2.unsqueeze(0).to(U_patch.dtype)

        # Zero-padding refines the angular frequency grid without changing the
        # physical patch sampling pitch.
        P = self.patch_px
        pad_size = int(self.pad_factor * max(0, P - 1))
        # if pad_size % 2 == 0:
        #     pad_size += 1
        tpad = (pad_size - P)
        pad_before = tpad // 2
        pad_after = tpad - pad_before
        U_pad = F.pad(U_patch, (pad_before, pad_after, pad_before, pad_after))
        _, Ppad, _ = U_pad.shape

        # Frequency axes in cycles/mm along local u and v.
        real_dtype = _real_dtype_like(self.field.real)
        fu = torch.fft.fftshift(torch.fft.fftfreq(Ppad, d=du)).to(self.device, dtype=real_dtype)
        fv = torch.fft.fftshift(torch.fft.fftfreq(Ppad, d=dv)).to(self.device, dtype=real_dtype)
        Fu, Fv = torch.meshgrid(fu, fv, indexing="xy")

        spec = _fft2c(U_pad)
        power = torch.abs(spec) ** 2
        power = torch.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)

        sumP = power.sum(dim=(1, 2), keepdim=True)
        dead0 = (sumP <= 0)
        if dead0.any():
            cy = Ppad // 2
            cx = Ppad // 2
            power[dead0.expand_as(power)] = 0.0
            power[dead0.squeeze(-1).squeeze(-1), cy, cx] = 1.0
            sumP = power.sum(dim=(1, 2), keepdim=True)
        power = power / (sumP + EPS)

        # Offset the local spectrum by the incident transverse wavevector, then
        # discard evanescent components in the exit medium.
        wvln_mm = (wvln_um * 1e-3)
        k0 = (2 * torch.pi) / wvln_mm
        if torch.is_tensor(n1):
            n1v = n1.item()
        else:
            n1v = float(n1)
        if torch.is_tensor(n2):
            n2v = n2.item()
        else:
            n2v = float(n2)

        du_in = torch.sum(d_in * u_hat, dim=-1)
        dv_in = torch.sum(d_in * v_hat, dim=-1)
        Ku0 = (n1v * k0 * du_in).view(Nv, 1, 1)
        Kv0 = (n1v * k0 * dv_in).view(Nv, 1, 1)

        k_exit = n2v * k0

        Kx_grid = Ku0 + (2 * torch.pi * Fu).unsqueeze(0)
        Ky_grid = Kv0 + (2 * torch.pi * Fv).unsqueeze(0)

        mask = (Kx_grid**2 + Ky_grid**2) < (k_exit**2)
        power = power * mask
        power = torch.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)
        power = torch.relu(power)
        sumP = power.sum(dim=(1, 2), keepdim=True)
        dead = (sumP <= 0)
        if dead.any():
            cy = Ppad // 2
            cx = Ppad // 2
            power[dead.expand_as(power)] = 0.0
            power[dead.squeeze(-1).squeeze(-1), cy, cx] = 1.0
            sumP = power.sum(dim=(1, 2), keepdim=True)
        power = power / (sumP + EPS)

        if self.spr <= 1:
            # Mean transverse wavevector gives one deterministic outgoing ray.
            Kx_mean = (power * Kx_grid).sum(dim=(1, 2))
            Ky_mean = (power * Ky_grid).sum(dim=(1, 2))

            k_exit_v = k_exit
            du_out = Kx_mean / (k_exit_v + EPS)
            dv_out = Ky_mean / (k_exit_v + EPS)

            rho2 = (du_out**2 + dv_out**2).clamp(0.0, 1.0 - 1e-12)
            dw_out = torch.sqrt(1.0 - rho2)

            d_out = (
                du_out.unsqueeze(-1) * u_hat +
                dv_out.unsqueeze(-1) * v_hat +
                dw_out.unsqueeze(-1) * n_loc
            )
            d_out = d_out / (torch.norm(d_out, dim=-1, keepdim=True) + EPS)
            d_out = torch.nan_to_num(d_out, nan=0.0, posinf=0.0, neginf=0.0)

            d_new = ray.d.reshape(-1, 3)
            d_new[valid_1d] = d_out
            ray.d = d_new.reshape(ray.d.shape)

            if ray.is_coherent:
                center = self._sample_field_at(o[..., :2])
                center = _nan_to_num_complex(center)
                center_phi = torch.angle(center)
                opd_mm = (center_phi * (wvln_um * 1e-3) / (2 * torch.pi)).view(-1, 1)
                opl_new = ray.opl.view(n_slots, -1)
                opl_new[valid_1d] = opl_new[valid_1d] + opd_mm
                ray.opl = opl_new.reshape(ray.opl.shape)

            return ray

        else:
            spr = int(max(2, self.spr))
            Nv, Ppad = power.shape[0], power.shape[-1]

            # Sample from the spectrum magnitude so amplitudes can carry phase.
            mag = torch.abs(spec)
            pdf = mag * mask
            pdf = torch.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)

            sumM = pdf.sum(dim=(1, 2), keepdim=True)
            dead = (sumM <= 0)
            if dead.any():
                cy = Ppad // 2
                cx = Ppad // 2
                pdf[dead.expand_as(pdf)] = 0.0
                pdf[dead.squeeze(-1).squeeze(-1), cy, cx] = 1.0
                sumM = pdf.sum(dim=(1, 2), keepdim=True)
            pdf = pdf / (sumM + EPS)

            flat_pdf = pdf.reshape(Nv, -1)
            idx = torch.multinomial(flat_pdf, spr, True)
            y_idx = idx // Ppad
            x_idx = idx % Ppad

            fx_s = Fu[y_idx, x_idx]
            fy_s = Fv[y_idx, x_idx]

            Kx_s = Ku0.view(Nv, 1) + (2 * torch.pi) * fx_s
            Ky_s = Kv0.view(Nv, 1) + (2 * torch.pi) * fy_s
            du_s = Kx_s / (k_exit + EPS)
            dv_s = Ky_s / (k_exit + EPS)
            dw_s = torch.sqrt((1 - du_s**2 - dv_s**2).clamp_min(0))

            d_world = (
                du_s.unsqueeze(-1) * u_hat.unsqueeze(1)
                + dv_s.unsqueeze(-1) * v_hat.unsqueeze(1)
                + dw_s.unsqueeze(-1) * n_loc.unsqueeze(1)
            )
            d_world = d_world / (torch.norm(d_world, dim=-1, keepdim=True) + EPS)

            ray_ix = torch.arange(Nv, device=self.device).unsqueeze(1)
            spec_s = spec[ray_ix, y_idx, x_idx]
            pdf_s = pdf[ray_ix, y_idx, x_idx].clamp_min(1e-12)

            # The spatial sampling factor keeps the unnormalized FFT convention
            # consistent across different DOE pixel pitches.
            amps = spec_s * ((du * dv) / pdf_s.detach()) / spr

            o_s = o.unsqueeze(1).repeat(1, spr, 1).reshape(-1, 3)
            d_s = d_world.reshape(-1, 3)
            wvln_s = wvln_um
            parent_opl = ray.opl.view(n_slots, -1)[valid_1d].view(Nv, 1, 1).repeat(1, spr, 1).reshape(-1, 1)
            parent_en = ray.en.repeat(1, spr, 1).reshape(-1, 1)

            new_ray = Ray(o_s, d_s, wvln=wvln_s, is_coherent=ray.is_coherent, device=self.device)
            new_ray.is_valid = torch.ones(o_s.shape[0], device=self.device)
            new_ray.en = parent_en
            new_ray.opl = parent_opl
            new_ray.is_forward = (new_ray.d[..., 2].unsqueeze(-1) > 0)

            new_ray.amp = amps.reshape(-1)
            new_ray.parent_ix = torch.arange(Nv).repeat_interleave(spr)

            return new_ray

    def _build_tangent_uv(self, n_hat: torch.Tensor, d_in: torch.Tensor):
        """Build a stable tangent frame from the local normal and incident ray."""
        n = F.normalize(n_hat, dim=-1)
        t = (-d_in) - (n * torch.sum(-d_in * n, dim=-1, keepdim=True))
        t_norm = torch.norm(t, dim=-1, keepdim=True)
        fallback = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        t = torch.where(t_norm < 1e-12, fallback.expand_as(t), t)
        u = F.normalize(t, dim=-1)
        v = F.normalize(torch.cross(n, u, dim=-1), dim=-1)
        u = F.normalize(torch.cross(v, n, dim=-1), dim=-1)
        return u, v

    def _extract_local_patches(self, o_xy: torch.Tensor, u_hat: torch.Tensor, v_hat: torch.Tensor, P: int):
        """
        Tangent-aligned P×P *complex field* patch centered at each hit.
        Returns: patch [Nv,P,P] complex, (du,dv) = (dx,dy)
        """
        assert self.field is not None and self._is_complex_field
        H, W = self.field.shape
        dx, dy = self.dx, self.dy
        Nv = o_xy.shape[0]

        du, dv = dx, dy
        half_u = (P - 1) * 0.5 * du
        half_v = (P - 1) * 0.5 * dv

        real_dtype = _real_dtype_like(self.field.real)
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

        img_r = self.field.real.view(1, 1, H, W)
        img_i = self.field.imag.view(1, 1, H, W)
        samp_r = _grid_sample_fp32(img_r.expand(Nv, 1, H, W), grid, align_corners=True).squeeze(1)
        samp_i = _grid_sample_fp32(img_i.expand(Nv, 1, H, W), grid, align_corners=True).squeeze(1)
        patch  = torch.complex(samp_r.to(self.field.real.dtype),
                               samp_i.to(self.field.real.dtype))  # [Nv,P,P]
        patch  = _nan_to_num_complex(patch, nan=0.0, posinf=0.0, neginf=0.0)
        return patch, (du, dv)

    def _sample_field_at(self, xy: torch.Tensor) -> torch.Tensor:
        """Sample complex field at scattered (x,y), with origin_xy as CENTER."""
        H, W = self.field.shape
        In, Jn = self._xy_to_norm_centered(xy, H, W)
        grid = torch.stack([In, Jn], dim=-1).view(1, -1, 1, 2)

        img_r = self.field.real.view(1, 1, H, W)
        img_i = self.field.imag.view(1, 1, H, W)
        out_r = _grid_sample_fp32(img_r, grid, align_corners=True).view(-1)
        out_i = _grid_sample_fp32(img_i, grid, align_corners=True).view(-1)
        out   = torch.complex(out_r.to(self.field.real.dtype),
                              out_i.to(self.field.real.dtype))
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
        r = surf_dict.get("design_r", surf_dict.get("r", 5.0))
        c = surf_dict.get("c", 0.0)
        k = surf_dict.get("k", 0.0)
        ai = surf_dict.get("ai", None)
        mat2 = surf_dict.get("mat2", "air")
        device = surf_dict.get("device", "cpu")
        field = surf_dict.get("field", None)
        dx = surf_dict.get("dx", None)
        dy = surf_dict.get("dy", None)
        origin_xy = tuple(surf_dict.get("origin_xy", (0.0, 0.0)))  # CENTER
        patch_px = int(surf_dict.get("patch_px", 64))
        pad_factor = int(surf_dict.get("pad_factor", 4))
        window = surf_dict.get("window", "hamming")
        spr = int(surf_dict.get("spr", 0))
        pad_field_for_patches = bool(surf_dict.get("pad_field_for_patches", True))
        return cls(d=d, r=r, c=c, k=k, ai=ai, mat2=mat2, device=device,
                   field=field, dx=dx, dy=dy, origin_xy=origin_xy,
                   patch_px=patch_px, pad_factor=pad_factor, window=window, spr=spr,
                   pad_field_for_patches=pad_field_for_patches)

    def surf_dict(self):
        base = super().surf_dict()
        base.update({
            "type": "DoeRaywave",
            "dx": self.dx,
            "dy": self.dy,
            "origin_xy": self.origin_xy,   # CENTER
            "patch_px": self.patch_px,
            "pad_factor": self.pad_factor,
            "window": self.window,
            "spr": self.spr,
            "design_r": self.design_r,
            "pad_field_for_patches": self.pad_field_for_patches,
        })
        return base

    def draw_widget(self, ax, color="black", linestyle="-"):
        super().draw_widget(ax, color, linestyle)
        # Plot the map extent with origin_xy at the CENTER
        if self.field is not None:
            H, W = self.field.shape
            xs, ys = self._xy_axes_centered(H, W)
            extent = [float(xs[0] - self.dx * 0.5), float(xs[-1] + self.dx * 0.5),
                      float(ys[0] - self.dy * 0.5), float(ys[-1] + self.dy * 0.5)]
            ax.imshow(torch.angle(self.field).detach().cpu().numpy(),
                      extent=extent, origin='lower', cmap='twilight', alpha=0.25)
        # Surface indicator
        d_cpu = float(self.d.cpu()) if hasattr(self.d, "cpu") else float(self.d)
        ax.plot([d_cpu - 0.1, d_cpu + 0.1], [0, 0], color=color, linestyle=linestyle, linewidth=1.5)
