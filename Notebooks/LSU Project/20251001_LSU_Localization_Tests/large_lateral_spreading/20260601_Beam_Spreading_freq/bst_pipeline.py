"""
bst_pipeline.py  --  validated broadband-FFT beam-spreading analysis pipeline
=============================================================================

Audit-hardened replacement for the ad-hoc analysis in
``20260602_IPR_Calculation_FFT.ipynb`` / ``20251008_IPR_Calculation.ipynb``.
Every function documents the physical quantity it computes, its convention, and
the systematic it controls. See ``Claude/AUDIT_REPORT.md`` and
``Claude/METHODS.md`` for the derivations and literature anchors.

KEY CONVENTIONS (verified 2026-07-17 against Tidy3D 2.9.1 source + the data):
  * Stored HDF5 fields are the FREQUENCY-DOMAIN running DFT of a FieldMonitor,
        Ehat_c(x,y,f) = (dt/sqrt(2pi)) sum_n E_c(t_n) exp(+2 pi i f t_n),
    UN-normalized (Simulation.normalize_index = None), so they still carry the
    Gaussian source-spectrum envelope |S(f)|.  Axis order is (x, y, z=1, f),
    f strictly ASCENDING and uniform (df).
  * Time reconstruction: because of the +i sign, scipy.fft.fft over the f axis
    yields a CAUSAL E(t) on t_n = n*dt, dt = 1/(N*df), period T = 1/df.
    (scipy.fft.ifft would time-mirror it.)  The optical carrier exp(-2 pi i f0 t)
    is irrelevant to |E(t)|^2 and is dropped.
  * A narrow-band response at nu_c is I_W(x,y,t) = sum_c |FFT_f[W(f) Ehat_c]|^2,
    with W a spectral window centred at f_c.  Because the data is scale-invariant
    per bin under the estimators here, the source envelope cancels; there is NO
    source deconvolution (doing so would be a double-deconvolution -- see report).

PHYSICS TARGETS (Cherroret, Skipetrov & van Tiggelen, PRE 82, 056603 (2010)):
    diffusive     sigma^2(t) = 4 D t           [profile ~ exp(-rho^2 / 4Dt)]
    localized     sigma^2(t) -> 2 L xi (1 - xi/L) <= L^2/2   (saturation)
    mobility edge sigma^2(t) -> ~ L^2
    absorption is a pure multiplicative exp(-t/tau_a) -> cancels in sigma^2(t)
    BUT rho-dependent loss (transverse absorbers, aperture truncation, wrap) does
    NOT cancel and biases sigma^2 DOWN (fake saturation) -- see valid-window logic.

HARD STRUCTURAL LIMIT for THIS dataset (12x-tiled base cell, transverse period
    = L = 14.3 um = 5.58 a):  transverse transport is Bloch-periodic beyond one
    tile, so sigma^2(t) is physically interpretable only while sigma <~ 1 tile,
    i.e. sigma^2 <~ 30 a^2.  Any 'saturation' at sigma^2 >~ 30 a^2 is tiling /
    aperture / wrap, NOT a localization length.  Fits are capped accordingly.
"""

from __future__ import annotations
import numpy as np
import h5py
import scipy.fft

A_UM = 2.562629142772549          # characteristic length a [um]
C_UM_S = 2.99792458e14            # speed of light [um/s]   (= 299.792458 um/ps)
L_SLAB_UM = 14.3                  # slab thickness [um]  (= 5.580 a)
FIELD_SCALE = 1e16                # raw |E| ~ 1e-15 (Tidy3D un-normalized units);
                                  # rescale at read so I=|E|^2 ~ O(1) and I^2 does
                                  # NOT underflow float32 -> 0 (which NaNs the PR).
                                  # Any global scale cancels in every estimator here.
GAP_NU = (0.43214708, 0.46009945) # MPB pseudogap edges in nu = a/lambda

# ---------------------------------------------------------------------------
#  Axis / metadata helpers
# ---------------------------------------------------------------------------

def load_axes(h5path, group="2.90"):
    """Return dict with x, y (um), f (Hz), nu (=a f/c), df (Hz), T (ps), dt (ps),
    N, and 1D trapezoidal integration weights wx, wy for the (possibly
    non-uniform) monitor grid.  Reads only the small coordinate datasets."""
    with h5py.File(h5path, "r") as h:
        g = h[group]
        x = np.asarray(g["x"][()], float)
        y = np.asarray(g["y"][()], float)
        f = np.asarray(g["f"][()], float)
    N = f.size
    df = float(np.mean(np.diff(f)))
    T_ps = 1.0 / df * 1e12                      # FFT period [ps]
    dt_ps = T_ps / N                            # reconstruction step [ps]
    nu = A_UM * f / C_UM_S                       # a f / c  (f in Hz, c in um/s)
    return dict(x=x, y=y, f=f, nu=nu, df=df, N=N, T_ps=T_ps, dt_ps=dt_ps,
                wx=_trap_weights(x), wy=_trap_weights(y))


def _trap_weights(coord):
    """1D trapezoidal quadrature weights so that sum_i w_i f_i ~ integral f dcoord
    on a non-uniform grid."""
    c = np.asarray(coord, float)
    w = np.empty_like(c)
    w[1:-1] = 0.5 * (c[2:] - c[:-2])
    w[0] = 0.5 * (c[1] - c[0])
    w[-1] = 0.5 * (c[-1] - c[-2])
    return w


# ---------------------------------------------------------------------------
#  Spectral windows
# ---------------------------------------------------------------------------

def spectral_window(nu, nu_c, dnu, shape="gauss"):
    """Narrow-band spectral window W(nu) centred at nu_c.

    dnu is the Gaussian STANDARD DEVIATION in nu for shape='gauss' (pulse
    duration t_p ~ 1/(2 pi dnu_f), dnu_f = dnu * c/a).  For 'tukey'/'box' dnu is
    the HALF-WIDTH of the pass band.  Returns a float array, max 1.
    Gaussian is preferred for multi-decade decay (no sidelobes)."""
    nu = np.asarray(nu, float)
    if shape == "gauss":
        return np.exp(-0.5 * ((nu - nu_c) / dnu) ** 2)
    if shape == "box":
        return (np.abs(nu - nu_c) <= dnu).astype(float)
    if shape == "tukey":
        # flat top with cosine tapers over the outer 50% of the half-width
        x = (nu - nu_c) / dnu
        w = np.zeros_like(x)
        inside = np.abs(x) <= 1.0
        taper = np.abs(x) > 0.5
        w[inside] = 1.0
        t = (np.abs(x) - 0.5) / 0.5
        w[inside & taper] = 0.5 * (1 + np.cos(np.pi * t[inside & taper]))
        return w
    raise ValueError(f"unknown window shape {shape!r}")


def pulse_duration_ps(dnu, shape="gauss"):
    """Approximate temporal pulse duration (1/e half-width for gauss) of a window
    of width dnu in nu.  t_p ~ 1/(2 pi dnu_f), dnu_f = dnu * c / a  [Hz]."""
    dnu_f = dnu * C_UM_S / A_UM
    return 1.0 / (2 * np.pi * dnu_f) * 1e12


# ---------------------------------------------------------------------------
#  Core streaming reconstruction: sigma^2(t) and friends for one window set
# ---------------------------------------------------------------------------

def reconstruct_windows(h5path, windows, group="2.90", center=(0.0, 0.0),
                        bg_rho_list=(25.0, 28.0, 30.0), edge_rho=28.0,
                        hu_radii=(5.0, 10.0, 15.0, 20.0), hu_dr=1.0,
                        row_block=48, verbose=True, nx_limit=None):
    """One streaming pass over the (nx,ny,1,N) field file that, for every window
    in ``windows``, reconstructs the narrow-band intensity movie I_W(x,y,t) and
    accumulates -- WITHOUT ever holding a full movie in RAM -- the time series:

      P(t)          = integral I dA                       (total transmitted power)
      M2(t)         = integral rho^2 I dA                 (2nd moment numerator)
      I2(t)         = integral I^2 dA                     (for the PR diameter)
      Pann(t; rho0) = integral_{rho>=rho0} I dA           (annulus background lever)
      Pedge(t)      = integral_{rho>edge_rho} I dA        (aperture/wrap clock)
      Ihu(t; rho)   = mean I over the thin annulus at rho (Hu ratio widths)
      Iaxis(t)      = mean I over rho < 2a                (Hu on-axis reference)
      Parseval:     sum_t P(t) dt   vs   sum_f |W Ehat|^2  (energy check)

    ``windows``: list of dicts {name, nu_c, dnu, shape}.
    Returns dict keyed by window name -> dict of arrays (all times in ps).
    """
    ax = load_axes(h5path, group)
    x, y, nu = ax["x"], ax["y"], ax["nu"]
    wx, wy, N = ax["wx"], ax["wy"], ax["N"]
    dt_ps, T_ps = ax["dt_ps"], ax["T_ps"]
    t_ps = np.arange(N) * dt_ps
    xc, yc = center
    a = A_UM

    # Precompute per-window spectral weights and Parseval spectral energy
    W = []
    for w in windows:
        Wf = spectral_window(nu, w["nu_c"], w["dnu"], w.get("shape", "gauss"))
        W.append(Wf.astype(np.float32))
    nW = len(windows)

    # Accumulators per window
    P = [np.zeros(N) for _ in range(nW)]
    M2 = [np.zeros(N) for _ in range(nW)]
    I2 = [np.zeros(N) for _ in range(nW)]
    Pann = [{r: np.zeros(N) for r in bg_rho_list} for _ in range(nW)]
    Pedge = [np.zeros(N) for _ in range(nW)]
    Ihu = [{r: np.zeros(N) for r in hu_radii} for _ in range(nW)]
    Ihu_area = [{r: 0.0 for r in hu_radii} for _ in range(nW)]
    Iaxis = [np.zeros(N) for _ in range(nW)]
    Iaxis_area = [0.0 for _ in range(nW)]
    spec_energy = [float(np.sum((W[k].astype(float)) ** 2)) for k in range(nW)]
    # (spec_energy is per-pixel-independent; multiplied by field power below)
    spec_par = [0.0 for _ in range(nW)]   # sum_f |W Ehat|^2 integrated over area

    ny = y.size

    with h5py.File(h5path, "r") as h:
        g = h[group]
        Ex, Ey, Ez = g["Ex"], g["Ey"], g["Ez"]
        nx_full = Ex.shape[0]
        if nx_limit is None:
            i_start, i_stop = 0, nx_full
        else:
            # centered band of nx_limit rows (for smoke tests over the beam core)
            i_start = max(0, nx_full // 2 - nx_limit // 2)
            i_stop = min(nx_full, i_start + nx_limit)
        for i0 in range(i_start, i_stop, row_block):
            i1 = min(i0 + row_block, i_stop)
            # sequential block read, rescaled to avoid float32 underflow of I^2
            bx = np.asarray(Ex[i0:i1, :, 0, :]) * FIELD_SCALE   # (nb, ny, N) complex64
            by = np.asarray(Ey[i0:i1, :, 0, :]) * FIELD_SCALE
            bz = np.asarray(Ez[i0:i1, :, 0, :]) * FIELD_SCALE
            xrows = x[i0:i1]                                     # (nb,)
            rho2 = (xrows[:, None] - xc) ** 2 + (y[None, :] - yc) ** 2   # (nb,ny)
            rho = np.sqrt(rho2)
            areaw = wx[i0:i1][:, None] * wy[None, :]             # (nb,ny)
            for k in range(nW):
                Wf = W[k]                                        # (N,) float32
                # narrow-band time reconstruction, vectorised over the whole block
                Iw = (np.abs(scipy.fft.fft(bx * Wf, axis=-1)) ** 2
                      + np.abs(scipy.fft.fft(by * Wf, axis=-1)) ** 2
                      + np.abs(scipy.fft.fft(bz * Wf, axis=-1)) ** 2)     # (nb,ny,N) f32
                aw = areaw[:, :, None]                           # (nb,ny,1)
                P[k]  += (Iw * aw).sum(axis=(0, 1), dtype=np.float64)
                M2[k] += (Iw * (rho2[:, :, None] * aw)).sum(axis=(0, 1), dtype=np.float64)
                I2[k] += (Iw ** 2 * aw).sum(axis=(0, 1), dtype=np.float64)
                for r in bg_rho_list:
                    m = rho >= r
                    if m.any():
                        Pann[k][r] += (Iw[m] * areaw[m][:, None]).sum(axis=0, dtype=np.float64)
                me = rho > edge_rho
                if me.any():
                    Pedge[k] += (Iw[me] * areaw[me][:, None]).sum(axis=0, dtype=np.float64)
                for r in hu_radii:
                    m = np.abs(rho - r) <= hu_dr
                    if m.any():
                        Ihu[k][r] += (Iw[m] * areaw[m][:, None]).sum(axis=0, dtype=np.float64)
                        Ihu_area[k][r] += float(areaw[m].sum())
                ma = rho < 2.0 * a
                if ma.any():
                    Iaxis[k] += (Iw[ma] * areaw[ma][:, None]).sum(axis=0, dtype=np.float64)
                    Iaxis_area[k] += float(areaw[ma].sum())
                # Parseval spectral side: sum_f |W Ehat|^2 integrated over area
                spec = ((np.abs(bx * Wf) ** 2 + np.abs(by * Wf) ** 2
                         + np.abs(bz * Wf) ** 2).sum(axis=-1) * areaw)    # (nb,ny)
                spec_par[k] += float(spec.sum(dtype=np.float64))
            if verbose:
                print(f"    rows {i1}/{i_stop}", end="\r", flush=True)
    if verbose:
        print()

    out = {}
    for k, w in enumerate(windows):
        # Parseval: time energy sum_n P(n)*dt vs spectral energy.
        # DFT pair E(t_n)=fft(Ehat)[n] gives sum_n |E_n|^2 = N * sum_f |Ehat_f|^2,
        # so compare sum_n P(n) to N * spec_par (dt factors cancel as a ratio).
        time_energy = float(P[k].sum())
        spectral_energy = float(N * spec_par[k])
        res = dict(
            name=w["name"], nu_c=w["nu_c"], dnu=w["dnu"],
            shape=w.get("shape", "gauss"),
            t_ps=t_ps, T_ps=T_ps, dt_ps=dt_ps,
            P=P[k], M2=M2[k], I2=I2[k],
            sigma2_raw=_safe_div(M2[k], P[k]) / a ** 2,      # <rho^2>/a^2, no bg sub
            Pann={r: Pann[k][r] for r in bg_rho_list},
            Pedge=Pedge[k],
            edge_frac=_safe_div(Pedge[k], P[k]),
            Ihu={r: _safe_div(Ihu[k][r], Ihu_area[k][r]) for r in hu_radii},
            Iaxis=_safe_div(Iaxis[k], Iaxis_area[k]),
            hu_radii=np.array(hu_radii),
            bg_rho_list=np.array(bg_rho_list),
            parseval_ratio=time_energy / spectral_energy if spectral_energy else np.nan,
            pulse_ps=pulse_duration_ps(w["dnu"], w.get("shape", "gauss")),
        )
        # background-subtracted sigma^2 for each background_rho (annulus mean)
        res["sigma2_bg"] = {}
        for r in bg_rho_list:
            # annulus mean background per frame = Pann / area(rho>=r)
            area_ann = _annulus_area(x, y, wx, wy, xc, yc, r)
            bg = Pann[k][r] / area_ann if area_ann > 0 else np.zeros(N)
            # subtract flat background: M2_sub = M2 - bg*<rho^2>_aperture*Area? -- do
            # it exactly by removing bg*integral(rho^2 dA) and bg*Area from moments
            IA = _aperture_area(wx, wy)
            Irho2 = _aperture_rho2_integral(x, y, wx, wy, xc, yc)
            P_sub = P[k] - bg * IA
            M2_sub = M2[k] - bg * Irho2
            res["sigma2_bg"][r] = _safe_div(M2_sub, P_sub) / a ** 2
        out[w["name"]] = res
    return out


# ---------------------------------------------------------------------------
#  small geometry integrals (computed once, cheap)
# ---------------------------------------------------------------------------

def _aperture_area(wx, wy):
    return float(wx.sum() * wy.sum())

def _aperture_rho2_integral(x, y, wx, wy, xc, yc):
    # integral over aperture of rho^2 dA  = integral (x-xc)^2 dA + integral (y-yc)^2 dA
    ix = float(((x - xc) ** 2 * wx).sum() * wy.sum())
    iy = float(wx.sum() * ((y - yc) ** 2 * wy).sum())
    return ix + iy

def _annulus_area(x, y, wx, wy, xc, yc, r0):
    X, Y = np.meshgrid(x, y, indexing="ij")
    rho = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
    Wt = np.outer(wx, wy)
    return float(Wt[rho >= r0].sum())

def _safe_div(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    out = np.full(np.broadcast(a, b).shape, np.nan)
    m = b != 0
    out[m] = a[m] / b[m]
    return out


# ---------------------------------------------------------------------------
#  PR diameter from the streamed moments
# ---------------------------------------------------------------------------

def pr_diameter_of_t(res):
    """Participation-ratio diameter d(t)/a = 2 sqrt(A_eff/pi)/a,
    A_eff = P^2 / I2.  Computed from the (background-UNsubtracted) streamed
    moments -- the PR is background-robust by construction and MUST be applied to
    the non-negative intensity (never to a signed background-subtracted frame)."""
    A_eff = _safe_div(res["P"] ** 2, res["I2"])       # um^2
    return 2.0 * np.sqrt(A_eff / np.pi) / A_UM


# ---------------------------------------------------------------------------
#  valid-window determination and diffusion fit
# ---------------------------------------------------------------------------

def valid_window(res, sigma2_key=("sigma2_bg", 28.0), sigma2_cap=25.0,
                 edge_frac_cap=0.01, arrival_frac=0.02):
    """Return (i_lo, i_hi) index bounds of the physically trustworthy fit window
    for the DIFFUSIVE GROWTH sigma^2(t) = 4Dt.

    CRITICAL (fixed 2026-07-17, after an adversarial verifier caught the bug): the
    clean 4Dt growth lives on the PRE-PEAK RISING EDGE of the transmitted pulse
    (early-arriving light has spread less, later light more; sigma^2 grows through
    the power peak). An earlier version started the fit at the power peak, which
    discarded the entire growth phase and fitted only the post-peak decay -> a
    spurious near-zero/negative D. i_lo is now the ARRIVAL (first rise above
    arrival_frac of peak power).

    i_lo = arrival (P first exceeds arrival_frac * P_peak).
    i_hi = earliest of: sigma^2 > sigma2_cap (~1 tile / aperture onset),
                        edge power fraction > edge_frac_cap (wrap/edge onset),
                        the power peak (past it the halo/aperture/background
                        corrupt sigma^2, especially the bg-subtracted version).
    """
    P = res["P"]; t = res["t_ps"]
    i_peak = int(np.argmax(P))
    i_arr = int(np.argmax(P > arrival_frac * P[i_peak]))   # first sample above threshold
    s2 = res[sigma2_key[0]][sigma2_key[1]] if isinstance(sigma2_key, tuple) else res[sigma2_key]
    ef = res["edge_frac"]
    i_cap = _first_true(s2[i_arr:] > sigma2_cap, i_arr, default=len(t) - 1)
    i_edge = _first_true(ef[i_arr:] > edge_frac_cap, i_arr, default=len(t) - 1)
    i_hi = min(i_cap, i_edge, i_peak)
    return i_arr, max(i_arr + 4, i_hi)


def _first_true(mask, offset, default):
    idx = np.nonzero(mask)[0]
    return offset + int(idx[0]) if idx.size else default


def fit_diffusion(res, sigma2_key=("sigma2_bg", 28.0), **vw):
    """Fit sigma^2(tau) = sigma0^2 + 4 D tau over the valid window.
    Returns dict with D [a^2/ps], slope 4D, R^2, the window (t_lo,t_hi), n_pts,
    and a CAUTION note that oversampled FFT frames are correlated (R^2 optimistic).
    Also reports whether sigma^2 stays below the cap (i.e. the fit is physical)."""
    i_lo, i_hi = valid_window(res, sigma2_key=sigma2_key, **vw)
    t = res["t_ps"]
    s2 = res[sigma2_key[0]][sigma2_key[1]] if isinstance(sigma2_key, tuple) else res[sigma2_key]
    tt = t[i_lo:i_hi]; ss = s2[i_lo:i_hi]
    m = np.isfinite(tt) & np.isfinite(ss)
    tt, ss = tt[m], ss[m]
    if tt.size < 4:
        return dict(D=np.nan, slope=np.nan, R2=np.nan, n=tt.size,
                    t_lo=float(t[i_lo]), t_hi=float(t[i_hi]),
                    note="too few valid points")
    A = np.vstack([np.ones_like(tt), tt]).T
    coef, *_ = np.linalg.lstsq(A, ss, rcond=None)
    fit = A @ coef
    ss_res = float(np.sum((ss - fit) ** 2))
    ss_tot = float(np.sum((ss - ss.mean()) ** 2))
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    slope = float(coef[1])
    return dict(D=slope / 4.0, slope=slope, sigma0_2=float(coef[0]), R2=R2,
                n=int(tt.size), t_lo=float(tt[0]), t_hi=float(tt[-1]),
                sigma2_max=float(np.nanmax(ss)),
                note="R^2 optimistic: oversampled FFT frames are correlated")


def xi_from_saturation(sigma2_inf_over_a2, L_over_a=L_SLAB_UM / A_UM):
    """Invert sigma^2_inf = 2 L xi (1 - xi/L) for xi/a (Cherroret Fig. 3 dashed
    line).  Real only for sigma^2_inf <= L^2/2 (= 15.57 a^2 here); returns nan
    otherwise (a signal that the 'plateau' is above the localization ceiling and
    is aperture/tiling, not xi)."""
    s = np.asarray(sigma2_inf_over_a2, float)
    Lc = L_over_a
    disc = 1.0 - 2.0 * s / Lc ** 2
    xi = np.where(disc >= 0, 0.5 * Lc * (1.0 - np.sqrt(np.clip(disc, 0, None))), np.nan)
    return xi


def valid_window_arrays(t, P, sigma2, edge_frac, sigma2_cap=25.0, edge_frac_cap=0.01,
                        arrival_frac=0.02):
    """Same rising-edge valid-window logic as ``valid_window`` but from bare arrays
    (so it can be applied to curves reloaded from an .npz). i_lo = arrival (NOT the
    power peak — see ``valid_window`` for why). Returns (i_lo, i_hi)."""
    P = np.asarray(P); t = np.asarray(t)
    i_peak = int(np.argmax(P))
    i_arr = int(np.argmax(P > arrival_frac * P[i_peak]))
    i_cap = _first_true(np.asarray(sigma2)[i_arr:] > sigma2_cap, i_arr, default=len(t) - 1)
    i_edge = _first_true(np.asarray(edge_frac)[i_arr:] > edge_frac_cap, i_arr, default=len(t) - 1)
    i_hi = min(i_cap, i_edge, i_peak)
    return i_arr, max(i_arr + 4, i_hi)


def fit_from_arrays(t, P, sigma2, edge_frac, **vw):
    """Diffusion fit sigma^2 = sigma0^2 + 4 D t over the valid window, from bare
    reloaded arrays.  Returns the same dict shape as ``fit_diffusion``."""
    i_lo, i_hi = valid_window_arrays(t, P, sigma2, edge_frac, **vw)
    t = np.asarray(t); s2 = np.asarray(sigma2)
    tt, ss = t[i_lo:i_hi], s2[i_lo:i_hi]
    m = np.isfinite(tt) & np.isfinite(ss)
    tt, ss = tt[m], ss[m]
    if tt.size < 4:
        return dict(D=np.nan, slope=np.nan, R2=np.nan, n=int(tt.size),
                    t_lo=float(t[i_lo]), t_hi=float(t[i_hi]), sigma2_max=np.nan,
                    note="too few valid points")
    A = np.vstack([np.ones_like(tt), tt]).T
    coef, *_ = np.linalg.lstsq(A, ss, rcond=None)
    fit = A @ coef
    ss_res = float(np.sum((ss - fit) ** 2)); ss_tot = float(np.sum((ss - ss.mean()) ** 2))
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(D=float(coef[1]) / 4.0, slope=float(coef[1]), sigma0_2=float(coef[0]),
                R2=R2, n=int(tt.size), t_lo=float(tt[0]), t_hi=float(tt[-1]),
                sigma2_max=float(np.nanmax(ss)),
                note="R^2 optimistic: oversampled FFT frames are correlated")


def hu_width2_of_t(res):
    """Hu ratio widths w_rho^2(t) = -rho^2 / ln[ I(rho,t)/I(0,t) ]  (Hu 2008).
    Background-free and aperture-free; localized transport shows w_rho^2 that
    SATURATES and INCREASES with rho (non-Gaussian), diffusion gives w^2=4Dt
    independent of rho.  Returns {rho: w2/a^2}."""
    I0 = res["Iaxis"]
    out = {}
    for r in res["hu_radii"]:
        ratio = _safe_div(res["Ihu"][float(r)], I0)
        with np.errstate(divide="ignore", invalid="ignore"):
            w2 = -(r ** 2) / np.log(ratio)
        w2[~np.isfinite(w2)] = np.nan
        w2[w2 <= 0] = np.nan
        out[float(r)] = w2 / A_UM ** 2
    return out
