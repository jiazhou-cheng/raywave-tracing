""" Generalized Snell's Law (GSL) diffractive surface for DOE.

DoeGSL diffractive surface implementation based on Aspheric class.
Put it under deeplens/diffractive_surface/DoeGSL.py. 
"""

import numpy as np
import torch
from deeplens.geometric_surface.aspheric import Aspheric


class DoeGSL(Aspheric):
    """
    Aspheric metasurface with parametrized phase profile and generalized Snell's law implementation.
    Combines aspheric surface geometry with metasurface phase modulation.
    Uses generalized Snell's law: n2*sin(θ2) - n1*sin(θ1) = (1/k0) * ∇φ
    """

    def __init__(
        self,
        d,
        r=5.0,
        c=0.0,
        k=0.0,
        ai=None,
        mat2="air",
        phase_func=None,
        device="cpu",
    ):
        """Initialize DoeGSL.

        Args:
            d (float): Distance of the metasurface. [mm]
            r (float): Aperture radius. [mm]
            c (float): Curvature (1/radius of curvature). [1/mm]
            k (float): Conic constant.
            ai (list): Aspheric coefficients [a2, a4, a6, ...].
            mat2 (str): Material name.
            phase_func (callable): Phase function phi(x, y, wvln).
            device (str): Device to run on.
        """
        super().__init__(r=r, d=d, c=c, k=k, ai=ai, mat2=mat2, device=device)
        self.phase_func = phase_func if phase_func is not None else (lambda x, y, wvln: torch.zeros_like(x))
        self._analytic_grad = None

    def phi(self, x, y, wvln):
        """Phase function."""
        return self.phase_func(x, y, wvln)

    def dphi_dxy(self, x, y, wvln):
        """Phase gradient. Prefer analytic gradient if available; otherwise use adaptive FD."""
        if self._analytic_grad is not None:
            return self._analytic_grad(x, y, wvln)

        eps_base = 0.0001
        dgdx, dgdy = self._dfdxy(x, y)
        eps_x = eps_base / torch.sqrt(1 + dgdx**2)
        eps_y = eps_base / torch.sqrt(1 + dgdy**2)

        phi_xp = self.phase_func(x + eps_x, y, wvln.squeeze(-1))
        phi_xm = self.phase_func(x - eps_x, y, wvln.squeeze(-1))
        dphidx = (phi_xp - phi_xm) / (2 * eps_x)

        phi_yp = self.phase_func(x, y + eps_y, wvln.squeeze(-1))
        phi_ym = self.phase_func(x, y - eps_y, wvln.squeeze(-1))
        dphidy = (phi_yp - phi_ym) / (2 * eps_y)

        return dphidx, dphidy

    def ray_reaction(self, ray, n1=1.0, n2=1.0):
        """Intersect aspheric surface, then refract with GSL."""
        ray = self.intersect(ray, n1) # Same as Aspheric class, already implemented in DeepLens
        n = self.normal_vec(ray) # already implemented in DeepLens
        return self.refract_gsl(ray, n1, n2, n) # Our own implementation

    def refract_gsl(self, ray, n1, n2, normal):
        """Refract rays using generalized Snell's law."""
        valid = ray.is_valid > 0.0

        dphidx, dphidy = self.dphi_dxy(ray.o[..., 0], ray.o[..., 1], ray.wvln) # Our own implementation
        k0 = 2 * np.pi / (ray.wvln * 1e-3)

        # Snell's law
        d_in = ray.d
        n_vec = normal
        cos_theta1 = torch.sum(d_in * n_vec, dim=-1)
        grazing_mask = torch.abs(cos_theta1) < 1e-10
        cos_theta1 = torch.where(grazing_mask, torch.sign(cos_theta1) * 1e-10, cos_theta1)
        sin_theta1_vec = d_in - cos_theta1.unsqueeze(-1) * n_vec

        # Generalized Snell's law
        grad_phi = torch.stack([dphidx, dphidy, torch.zeros_like(dphidx)], dim=-1)
        grad_phi_normal = torch.sum(grad_phi * n_vec, dim=-1, keepdim=True)
        grad_phi_parallel = grad_phi - grad_phi_normal * n_vec
        sin_theta2_vec = (n1 * sin_theta1_vec + grad_phi_parallel / k0) / n2
        sin2_theta2 = torch.sum(sin_theta2_vec * sin_theta2_vec, dim=-1)

        tir_threshold = 1.0 - 1e-6
        tir_mask = sin2_theta2 > tir_threshold

        new_d = ray.d.clone()

        if tir_mask.any():
            d_reflected = d_in - 2 * cos_theta1.unsqueeze(-1) * n_vec
            new_d[tir_mask] = d_reflected[tir_mask]
            if hasattr(ray, "ra"):
                ray.ra[tir_mask] *= 0.1

        # Transmitted rays and update ray direction
        transmitted_mask = ~tir_mask & valid
        if transmitted_mask.any():
            sin2_theta2_clamped = torch.clamp(sin2_theta2, 0.0, tir_threshold)
            cos_theta2 = torch.sqrt(1 - sin2_theta2_clamped)
            forward = cos_theta1 > 0
            cos_theta2 = torch.where(forward, cos_theta2, -cos_theta2)

            d_out = sin_theta2_vec + cos_theta2.unsqueeze(-1) * n_vec
            d_out_norm = torch.norm(d_out, dim=-1, keepdim=True)
            d_out_norm = torch.clamp(d_out_norm, min=1e-10)
            d_out = d_out / d_out_norm

            new_d[transmitted_mask] = d_out[transmitted_mask]

        new_d[~valid] = d_in[~valid]
        ray.d = new_d

        # update optical path length
        if ray.is_coherent:
            phi = self.phi(ray.o[..., 0], ray.o[..., 1], ray.wvln.squeeze(-1))
            new_opl = ray.opl + (phi * (ray.wvln.squeeze(-1) * 1e-3) / (2 * np.pi)).unsqueeze(-1)
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        return ray

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Initialize DoeGSL from a dict."""
        d = surf_dict["d"]
        r = surf_dict.get("r", 5.0)
        c = surf_dict.get("c", 0.0)
        k = surf_dict.get("k", 0.0)
        ai = surf_dict.get("ai", None)
        mat2 = surf_dict.get("mat2", "air")

        return cls(
            d=d,
            r=r,
            c=c,
            k=k,
            ai=ai,
            mat2=mat2,
        )

    def get_optimizer_params(self, lr=[1e-4, 1e-4, 1e-1, 1e-2], decay=0.01):
        """Get parameters for optimization."""
        return super().get_optimizer_params(lr, decay)

    def surf_dict(self):
        """Return a dict of surface parameters."""
        surf_dict = super().surf_dict()
        surf_dict.update({"type": "DoeGSL"})
        return surf_dict

    def draw_widget(self, ax, color="black", linestyle="-"):
        """Draw 2D widget in the plot."""
        super().draw_widget(ax, color, linestyle)
        d_cpu = float(self.d.cpu().item()) if isinstance(self.d, torch.Tensor) else float(self.d)
        ax.plot([d_cpu - 0.1, d_cpu + 0.1], [0, 0], color=color, linestyle=linestyle, linewidth=1.5)
