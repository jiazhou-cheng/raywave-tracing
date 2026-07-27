# Differentiable Ray–Wave Modeling for Hybrid Optical Systems

Hybrid refractive–diffractive systems are hard to model: rays and waves live at different spatial scales. This repo provides a **differentiable ray–wave** framework that plugs into standard ray tracing, supports **planar and curved** DOEs, and enables end-to-end inverse design at a single wavelength.

<p align="center">
  <img src="figures/figure1.png" width="70%" alt="Ray–wave overview">
</p>

## Plug-and-play and lightweight implementation

Ray tracing builds on open-source [**DeepLens**](https://github.com/vccimaging/DeepLens) (`deeplens/`). Our plug-and-play ray–wave modules:

| Module | Role |
|--------|------|
| [`deeplens/raywave.py`](deeplens/raywave.py) | Huygens PSF / coherent sensor rendering from traced rays |
| [`deeplens/diffractive_surface/DoeRaywavePlane.py`](deeplens/diffractive_surface/DoeRaywavePlane.py) | Planar diffractive optical element (DOE) surface, full-field rule-of-thumb |
| [`deeplens/diffractive_surface/DoeRaywave.py`](deeplens/diffractive_surface/DoeRaywave.py) | General DOE with aspheric surface sag, patch-based implementation |

Standalone demos also use the lightweight package `src_lightweight/`.

---

## Benchmark (Fig. 5)

Three systems vs. conventional methods (λ = 0.7 µm): (a) grating + lens, (b) free-space hologram, (c) hologram + lens (no standard reference).

<p align="center">
  <img src="figures/figure5.png" width="85%" alt="Figure 5 benchmark">
</p>

| Setup | Reference | Result |
|-------|-----------|--------|
| (a) Grating + lens | Generalized refraction | MSE 1.5×10⁻⁹, NCC 0.985 |
| (b) Planar hologram | Angular spectrum (ASM) | MSE 4.4×10⁻¹⁰, NCC 0.997 |
| (c) Hologram + lens | — | Non-paraxial speckles missed by paraxial design |

Notebooks: [`benchmarks/`](benchmarks/) (DeepLens + `raywave`).

---

## Inverse design (Fig. 6 and 7)

### Fig. 6 — Hybrid flat DOE + lens

Paraxial design → NCC 0.462; end-to-end ray–wave optimization → **NCC 0.952**.

<p align="center">
  <img src="figures/figure6.png" width="90%" alt="Figure 6 inverse design">
</p>

```bash
python -m hybrid_system_fig6.optim   # configs/hybrid_system_fig6
```

### Fig. 7 — Conformal reflective DOE

Curved substrate (NA = 0.5, λ = 1 µm): two-focus beamsplitter and Stanford “S” hologram (NCC 0.909).

<p align="center">
  <img src="figures/figure7.png" width="80%" alt="Figure 7 conformal DOE">
</p>

```bash
python -m conformal_doe_fig7.optim_beamsplitter   # configs/conformal_doe_bs_fig7
python -m conformal_doe_fig7.optim_hologram       # configs/conformal_doe_hologram_fig7
```

---

## Layout

```
deeplens/            # DeepLens fork + ray–wave plug-ins (raywave.py, DoeRaywave*)
src_lightweight/     # standalone ray–wave primitives
benchmarks/          # Fig. 5 notebooks
hybrid_system_fig6/  # Fig. 6 inverse design
conformal_doe_fig7/  # Fig. 7 conformal DOE
configs/             # demo configs
figures/             # paper figures
```

## Acknowledgment

This project uses [DeepLens](https://github.com/vccimaging/DeepLens) (Yang et al.) as the differentiable geometric ray-tracing backend for benchmark simulations.
