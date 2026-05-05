"""One-shot helper to swap the rdg_spectrum + plot_structure_factor cell.

Run once, then delete this file.
"""
import json
import sys
from pathlib import Path

NB_PATH = Path(__file__).with_name("20250903_create_h5_from_ends.ipynb")
TARGET_CELL_ID = "f59b2abb"

NEW_SRC = '''def rdg_spectrum(
    eps,
    box_size,
    n_bg=1.0,
    wavelengths=None,
    freqs=None,
    eps_bg=None,
    n_theta=500,
    volume_density=None,
    n_lowq_skip=5,
    n_lowq_fit=25,
    normalize_S=True,
):
    """
    RDG differential scattering cross section spectrum from a 3D permittivity.

    For each wavelength:
      q(θ)  = 2k sin(θ/2)              elastic scattering vector
      dσ/dΩ ∝ k⁴ · S(q(θ))
      σ      = ∫ dσ/dΩ dΩ
      g      = <cosθ>
      l_s    = 1 / (ρ · σ)
      l*     = l_s / (1 − g)

    S(q) is the radial average of |F(q)|² from the full 3D FFT of the
    mean-subtracted contrast `deps = eps - eps_bg`. The first `n_lowq_skip`
    bins near q≈0 are noisy (very few samples per bin in 3D) so they are
    replaced by a polynomial S(q) = a*q² + b*q⁴ fit (no constant term ⇒
    S(0)=0 enforced) using the next `n_lowq_fit` bins as the trustworthy
    window. Everything past the noisy region is preserved exactly.

    If `normalize_S` is True (default), |F(q)|² is divided by
    N_voxels · σ²_deps so that white-noise contrast gives S(q)≈1 and
    overall values are O(1) — publication-friendly. The shape of all
    derived curves (g, λ/l_s, λ/l*) is unchanged by the normalization.
    """
    import warnings

    eps = np.asarray(eps, dtype=float)
    if eps.ndim != 3:
        raise ValueError("eps must be 3D")
    if (wavelengths is None) == (freqs is None):
        raise ValueError("Provide exactly one of wavelengths or freqs")

    if eps_bg is None:
        eps_bg = n_bg ** 2
    else:
        n_bg = float(np.sqrt(np.real(eps_bg)))

    Nx, Ny, Nz = eps.shape
    if np.isscalar(box_size):
        Lx = Ly = Lz = float(box_size)
    else:
        Lx, Ly, Lz = map(float, box_size)
    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
    V = Lx * Ly * Lz

    if wavelengths is not None:
        wavelengths = np.asarray(wavelengths, dtype=float)
        freqs = 1.0 / wavelengths
    else:
        freqs = np.asarray(freqs, dtype=float)
        wavelengths = 1.0 / freqs

    rho = 1.0 / V if volume_density is None else float(volume_density)

    # RDG validity checks
    m = np.sqrt(np.max(np.real(eps))) / n_bg
    rdg_contrast = abs(m - 1.0)
    d_char = V ** (1.0 / 3.0)
    rdg_phase = rdg_contrast * 2 * np.pi * n_bg * d_char / float(np.min(wavelengths))
    if rdg_contrast > 0.1:
        warnings.warn(f"RDG contrast |m-1|={rdg_contrast:.3f} > 0.1. "
                      "Single-scattering may not be valid.")
    if rdg_phase > 1.0:
        warnings.warn(f"RDG phase k*d*|m-1|={rdg_phase:.3f} > 1. "
                      "Multiple scattering likely significant.")

    # --- FFT ---
    deps = eps - eps_bg
    deps -= deps.mean()

    F3d = np.fft.fftshift(np.fft.fftn(deps))
    P3d = np.abs(F3d) ** 2

    # Standard structure-factor normalization: rescales y-axis only.
    # σ² = mean(deps²) is the variance after mean subtraction; for
    # white-noise contrast <|F|²> = N·σ², so S(q) → 1 in that limit.
    if normalize_S:
        sigma2 = float(np.mean(deps ** 2))
        if sigma2 > 0.0:
            P3d /= (deps.size * sigma2)

    # kz=0 slice for the diagnostic plot
    iz_k = Nz // 2
    F2d_power = P3d[:, :, iz_k]
    qx2d = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
    qy2d = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny, d=dy))

    # S(q) radial average from the full 3D power
    qx3 = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
    qy3 = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny, d=dy))
    qz3 = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nz, d=dz))
    Qx3, Qy3, Qz3 = np.meshgrid(qx3, qy3, qz3, indexing="ij")
    qmag_3d = np.sqrt(Qx3**2 + Qy3**2 + Qz3**2)

    qmax = np.max(qmag_3d)
    q_bins = np.linspace(0, qmax, 1000)
    q_centers = 0.5 * (q_bins[:-1] + q_bins[1:])

    sums, _   = np.histogram(qmag_3d.ravel(), bins=q_bins, weights=P3d.ravel())
    counts, _ = np.histogram(qmag_3d.ravel(), bins=q_bins)
    S_q_raw = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float),
                        where=counts > 0)

    # Low-q stabilization: only the first n_lowq_skip bins are touched.
    # Fit S(q)=a*q²+b*q⁴ (no constant) on the next n_lowq_fit bins, then
    # evaluate the fit at q_centers[:n_lowq_skip]. S(0)=0 is enforced
    # structurally by omitting the constant term.
    S_q = S_q_raw.copy()
    if n_lowq_skip > 0 and n_lowq_skip + n_lowq_fit < len(q_centers):
        qf = q_centers[n_lowq_skip:n_lowq_skip + n_lowq_fit]
        Sf = S_q_raw[n_lowq_skip:n_lowq_skip + n_lowq_fit]
        A = np.column_stack([qf ** 2, qf ** 4])
        coeffs, *_ = np.linalg.lstsq(A, Sf, rcond=None)
        qr = q_centers[:n_lowq_skip]
        S_q[:n_lowq_skip] = coeffs[0] * qr ** 2 + coeffs[1] * qr ** 4

    # Anchor q=0 → S=0 in the interpolant used for the scattering integral.
    q_plot = np.r_[0.0, q_centers]
    S_plot = np.r_[0.0, S_q]

    S_interp = interp1d(
        q_plot,
        S_plot,
        bounds_error=False,
        fill_value=(0.0, 0.0)
        )

    # --- Per-wavelength integration ---
    theta = np.linspace(0.0, np.pi, n_theta)
    mu = np.cos(theta)
    sin_theta = np.sin(theta)

    g_list, sig_list, ls_list, lstar_list, ds_list = [], [], [], [], []

    for lam in wavelengths:
        k = 2.0 * np.pi * n_bg / lam
        q_scatter = 2.0 * k * np.sin(theta / 2.0)
        ang = 0.5 * (1.0 + np.cos(theta)**2)
        Sq = S_interp(q_scatter)
        dsigma = k**4 * ang * Sq
        sigma = 2.0 * np.pi * np.trapz(dsigma * sin_theta, theta)

        if sigma > 0.0:
            g      = 2.0 * np.pi * np.trapz(dsigma * sin_theta * mu, theta) / sigma
            l_s    = 1.0 / (rho * sigma)
            gc     = np.clip(g, -1.0 + 1e-10, 1.0 - 1e-10)
            l_star = l_s / (1.0 - gc)
        else:
            g, l_s, l_star = np.nan, np.inf, np.inf

        g_list.append(g);   sig_list.append(sigma)
        ls_list.append(l_s); lstar_list.append(l_star)
        ds_list.append(dsigma)

    g_arr     = np.array(g_list)
    sigma_arr = np.array(sig_list)
    ls_arr    = np.array(ls_list)
    lstar_arr = np.array(lstar_list)
    ds_arr    = np.array(ds_list)

    with np.errstate(divide="ignore", invalid="ignore"):
        ds_norm = np.where(sigma_arr[:, None] > 0,
                           ds_arr / sigma_arr[:, None], 0.0)

    return dict(
        wavelengths=wavelengths, freqs=freqs,
        g=g_arr, sigma=sigma_arr, l_s=ls_arr, l_star=lstar_arr,
        dsigma_dOmega=ds_arr, dsigma_dOmega_norm=ds_norm,
        theta=theta, mu=mu,
        S_q=S_q, S_q_raw=S_q_raw, q_centers=q_centers,
        F2d_power=F2d_power, qx2d=qx2d, qy2d=qy2d,
    )


def plot_structure_factor(results, figsize=(10, 4)):
    """2D kz=0 power map + radial average S(q): raw (blue) vs stabilized (red)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    F2d_power = results["F2d_power"]
    pos  = F2d_power[F2d_power > 0]
    vmin = np.percentile(pos, 1) if pos.size else 1e-10
    vmax = np.percentile(F2d_power, 99.9)
    ax.imshow(F2d_power.T, origin="lower",
              norm=mcolors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax),
              cmap="viridis")
    ax.set_title(r"$|F(k_x, k_y)|^2$  at  $k_z = 0$")
    ax.set_xlabel("$k_x$ index"); ax.set_ylabel("$k_y$ index")

    ax = axes[1]
    ax.plot(results["q_centers"], results["S_q_raw"],
            color="steelblue", lw=0.8, label="raw")
    ax.plot(results["q_centers"], results["S_q"],
            color="crimson", lw=1.0, label="stabilized")
    ax.set_xlabel(r"$|q|$  [rad / length unit]")
    ax.set_ylabel(r"$S(q)$")
    ax.set_title(r"Radial average of $|F(q)|^2$")
    ax.legend()

    plt.tight_layout()
    return fig


def plot_rdg_results(results, characteristic_length=None, figsize=(8, 11)):
    """
    Replicate Image 2 — four-panel RDG summary.
      Panel 1: dσ/dΩ(μ,λ)/σ(λ)   heat map
      Panel 2: λ/l_s              log scale
      Panel 3: g
      Panel 4: λ/l*
    """
    wavelengths = results["wavelengths"]
    g       = results["g"]
    l_s     = results["l_s"]
    l_star  = results["l_star"]
    ds_norm = results["dsigma_dOmega_norm"]
    mu      = results["mu"]

    if characteristic_length is not None:
        lam_axis = wavelengths / characteristic_length
        xlabel   = r"wavelength  $\\lambda$"
    else:
        lam_axis = wavelengths
        xlabel   = r"wavelength  $\\lambda$"

    fig, axes = plt.subplots(4, 1, figsize=figsize,
                             gridspec_kw={"hspace": 0.08})

    ax = axes[0]
    lam_grid, mu_grid = np.meshgrid(lam_axis, mu[::-1])
    data = ds_norm[:, ::-1].T
    vmax = np.nanpercentile(data, 99)
    im = ax.pcolormesh(lam_grid, mu_grid, data,
                       cmap="hot",
                       norm=mcolors.Normalize(vmin=0, vmax=vmax),
                       shading="auto", rasterized=True)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$d\\sigma/d\\Omega\\,/\\,\\sigma$", fontsize=8)
    ax.set_ylabel(r"$\\mu = \\cos\\theta$")
    ax.set_title(r"$d\\sigma/d\\Omega(\\mu,\\lambda)\\,/\\,\\sigma(\\lambda)$")
    ax.set_ylim(-1, 1); ax.set_xlim(lam_axis[0], lam_axis[-1])
    ax.tick_params(labelbottom=False)

    ax = axes[1]
    ax.semilogy(lam_axis, wavelengths / l_s)
    ax.set_ylabel(r"$\\lambda\\,/\\,l_s$")
    ax.set_xlim(lam_axis[0], lam_axis[-1])
    ax.tick_params(labelbottom=False)

    ax = axes[2]
    ax.plot(lam_axis, g)
    ax.axhline(0.0, color="k", lw=0.5, ls="--")
    ax.set_ylabel(r"$g$")
    ax.set_xlim(lam_axis[0], lam_axis[-1])
    ax.tick_params(labelbottom=False)

    ax = axes[3]
    ax.plot(lam_axis, wavelengths / l_star)
    ax.set_ylabel(r"$\\lambda\\,/\\,l^*$")
    ax.set_xlim(lam_axis[0], lam_axis[-1])
    ax.set_xlabel(xlabel)

    plt.tight_layout()
    return fig'''


def main() -> int:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    target = None
    for cell in nb["cells"]:
        if cell.get("id") == TARGET_CELL_ID:
            target = cell
            break
    if target is None:
        print(f"ERROR: cell id {TARGET_CELL_ID!r} not found", file=sys.stderr)
        return 1

    # Notebook source is a list of lines, each ending with '\n' except the last.
    lines = NEW_SRC.splitlines(keepends=True)
    target["source"] = lines
    # Clear stale outputs/exec count so the diff is clean.
    target["outputs"] = []
    target["execution_count"] = None

    NB_PATH.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Updated cell {TARGET_CELL_ID} ({len(lines)} lines).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
