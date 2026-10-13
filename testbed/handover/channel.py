"""3GPP TR 38.901 (v17) urban macro (UMa) large-scale channel, for signal traces.

Implemented (section numbers of TR 38.901):
  7.4.1   UMa path loss, LOS and NLOS, with the breakpoint distance d'_BP (Note 1: h_E = 1 m)
  7.4.2   UMa LOS probability (h_UT <= 13 m)
  7.4.1   shadow fading sigma: LOS 4 dB, NLOS 6 dB
  7.5-6   shadowing decorrelation distance: LOS 37 m, NLOS 50 m (Gudmundson exponential model)
  7.6.3.3 spatially consistent LOS state (procedure A: a correlated Gaussian mapped to a uniform,
          LOS if uniform < P_LOS(d_2D)), correlation distance 50 m for UMa
Simplifications (stated in the report): omnidirectional single-sector cells, independent shadowing
per cell, no fast fading (RSRP is averaged over REs and L1 samples), 2D distance >= 10 m.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

C_LIGHT = 3.0e8


def hex_sites(n_cells: int, isd_m: float) -> np.ndarray:
    """Central site plus rings of a hexagonal grid. 7 cells = centre + first ring."""
    pts = [(0.0, 0.0)]
    if n_cells > 1:
        for k in range(6):
            a = math.pi / 6 + k * math.pi / 3
            pts.append((isd_m * math.cos(a), isd_m * math.sin(a)))
    if n_cells > 7:
        raise ValueError("only up to 7 cells supported")
    return np.array(pts[:n_cells])


def los_probability_uma(d2d: np.ndarray, h_ut: float = 1.5) -> np.ndarray:
    d2d = np.maximum(d2d, 1e-3)
    c = 0.0 if h_ut <= 13 else ((h_ut - 13) / 10) ** 1.5
    p = (18 / d2d + np.exp(-d2d / 63) * (1 - 18 / d2d)) * \
        (1 + c * 5 / 4 * (d2d / 100) ** 3 * np.exp(-d2d / 150))
    return np.where(d2d <= 18, 1.0, np.clip(p, 0, 1))


def pathloss_uma_los(d2d, h_bs, h_ut, fc_ghz):
    d3d = np.sqrt(d2d**2 + (h_bs - h_ut) ** 2)
    h_e = 1.0
    d_bp = 4 * (h_bs - h_e) * (h_ut - h_e) * fc_ghz * 1e9 / C_LIGHT
    pl1 = 28.0 + 22 * np.log10(d3d) + 20 * np.log10(fc_ghz)
    pl2 = (28.0 + 40 * np.log10(d3d) + 20 * np.log10(fc_ghz)
           - 9 * np.log10(d_bp**2 + (h_bs - h_ut) ** 2))
    return np.where(d2d <= d_bp, pl1, pl2)


def pathloss_uma_nlos(d2d, h_bs, h_ut, fc_ghz):
    d3d = np.sqrt(d2d**2 + (h_bs - h_ut) ** 2)
    pl_nlos = 13.54 + 39.08 * np.log10(d3d) + 20 * np.log10(fc_ghz) - 0.6 * (h_ut - 1.5)
    return np.maximum(pathloss_uma_los(d2d, h_bs, h_ut, fc_ghz), pl_nlos)


def correlated_gaussian_along_path(dist_steps: np.ndarray, decorr_m: float,
                                   rng: np.random.Generator, n: int) -> np.ndarray:
    """n independent unit-variance Gaussian processes along a path with exponential
    autocorrelation exp(-d/decorr) (Gudmundson). dist_steps: (T,) distance moved per step."""
    T = len(dist_steps)
    rho = np.exp(-dist_steps / decorr_m)                 # (T,)
    x = np.empty((T, n))
    x[0] = rng.standard_normal(n)
    e = rng.standard_normal((T, n))
    s = np.sqrt(np.maximum(0.0, 1 - rho**2))
    for t in range(1, T):
        x[t] = rho[t] * x[t - 1] + s[t] * e[t]
    return x


def smooth_gaussian_along_path(dist_steps: np.ndarray, ell_m: float, rng: np.random.Generator,
                               n: int, grid_m: float = 0.5) -> np.ndarray:
    """n unit-variance processes along the travelled distance with Gaussian autocorrelation
    exp(-d^2 / ell^2) -- differentiable, unlike the exponential model, so a threshold on it
    produces clean transitions instead of flicker. White noise on a 0.5 m distance grid is
    convolved with a Gaussian kernel (std ell/sqrt(2)) and interpolated at the UE positions."""
    s = np.cumsum(dist_steps)
    sig = ell_m / np.sqrt(2)
    pad = 4 * sig
    grid = np.arange(-pad, s[-1] + pad + grid_m, grid_m)
    w = rng.standard_normal((len(grid), n))
    k_x = np.arange(-pad, pad + grid_m, grid_m)
    kern = np.exp(-0.5 * (k_x / sig) ** 2)
    kern /= np.sqrt((kern**2).sum())                     # unit variance after convolution
    from scipy.signal import fftconvolve
    field = fftconvolve(w, kern[:, None], mode="same", axes=0)
    return np.stack([np.interp(s, grid, field[:, j]) for j in range(n)], axis=1)


def large_scale_rsrp(pos: np.ndarray, sites: np.ndarray, dep: dict, ch: dict,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """pos: (T, 2) UE positions. Returns RSRP (T, C) in dBm per resource element and LOS (T, C)."""
    T = len(pos)
    d2d = np.maximum(np.linalg.norm(pos[:, None, :] - sites[None, :, :], axis=2), 10.0)  # (T, C)
    step = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1)]
    C = len(sites)
    # Spatially consistent LOS state (procedure A): hard rule  LOS <=> g < Phi^-1(P_LOS).
    # A hard switch makes RSRP jump by 25-40 dB between two samples a few cm apart, which no
    # policy can react to and which is not physical (diffraction around a building edge takes
    # metres). We therefore use a SOFT state s in [0, 1] (in the spirit of the soft LOS state of
    # TR 38.901 7.6.3.3): s = Phi((Phi^-1(P_LOS) - g) / eps). g must be SMOOTH: an exponentially
    # correlated g is rough at small scales and flickers across any threshold. With a Gaussian
    # correlation over 50 m, |dg/ds| ~ 0.03 /m, so eps = 0.1 gives ~3-5 m transitions.
    # eps -> 0 recovers the hard rule.
    g = smooth_gaussian_along_path(step, ch.get("los_decorr_m", 50), rng, C)
    eps = ch.get("soft_los_eps", 0.1)
    p_los = np.clip(los_probability_uma(d2d, dep["h_ut_m"]), 1e-6, 1 - 1e-6)
    s = norm.cdf((norm.ppf(p_los) - g) / eps) if eps > 0 else (norm.cdf(g) < p_los).astype(float)
    los = s > 0.5
    pl_l = pathloss_uma_los(d2d, dep["h_bs_m"], dep["h_ut_m"], dep["fc_ghz"])
    pl_n = pathloss_uma_nlos(d2d, dep["h_bs_m"], dep["h_ut_m"], dep["fc_ghz"])
    pl = s * pl_l + (1 - s) * pl_n
    if ch.get("shadowing", True):
        z_los = correlated_gaussian_along_path(step, ch["sf_decorr_m"]["los"], rng, C)
        z_nlos = correlated_gaussian_along_path(step, ch["sf_decorr_m"]["nlos"], rng, C)
        pl = pl + s * 4.0 * z_los + (1 - s) * 6.0 * z_nlos
    n_re = 12 * dep["n_rb"]
    p_re = dep["tx_power_dbm"] - 10 * np.log10(n_re) + dep.get("antenna_gain_dbi", 0.0)
    return p_re - pl, los


def noise_per_re_dbm(dep: dict) -> float:
    return -174.0 + 10 * np.log10(dep["scs_khz"] * 1e3) + dep["noise_figure_db"]


def sinr_db(rsrp_dbm: np.ndarray, serving: np.ndarray, noise_dbm: float) -> np.ndarray:
    """Full-load downlink SINR on the serving cell, all other cells interfere.
    rsrp_dbm: (..., C); serving: (...) int."""
    lin = 10 ** (rsrp_dbm / 10)
    s = np.take_along_axis(lin, serving[..., None], axis=-1)[..., 0]
    i = lin.sum(axis=-1) - s
    return 10 * np.log10(s / (i + 10 ** (noise_dbm / 10)))
