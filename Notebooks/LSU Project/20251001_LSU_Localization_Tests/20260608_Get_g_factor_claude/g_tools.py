"""
g_tools.py
==========
Corrected helpers for computing the scattering anisotropy factor

        g = <cos theta>

of a correlated disordered slab from FDTD / structure data.  See
``LITERATURE_AND_METHOD.md`` for the physics and the list of issues in the
original ``20260212_Get_g_factor`` setup that these functions fix.

Design notes
------------
* ``FieldProjectionAngleMonitor`` projects onto the **tensor grid** ``theta x phi``
  (data dims ``(r, theta, phi, f)``).  Therefore the angular grid MUST be a regular
  product grid -- NOT a Fibonacci ``get_sphere`` set of paired directions.  Use
  :func:`regular_angle_grid` to build the monitor's ``theta`` / ``phi`` and
  :func:`asymmetry_parameter` to integrate with the correct ``sin(theta) dtheta dphi``
  quadrature.
* ``g`` is defined from the **diffuse** (incoherent) intensity
  ``I_d = <|E|^2> - |<E>|^2``.  Use :func:`coherent_incoherent` to form it from an
  ensemble of complex far fields (and/or use azimuthal averaging as a proxy).
* Method A (Born) reads ``P(q)`` straight from the ``.h5`` permittivity via
  :func:`power_spectrum_from_eps`; no cloud run required.

Dependencies: numpy, h5py, scipy (spherical Bessel for the optional Mie reference).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "regular_angle_grid",
    "asymmetry_parameter",
    "phase_function",
    "coherent_incoherent",
    "q_of_theta",
    "power_spectrum_from_eps",
    "power_spectrum_robust",
    "power_spectrum_regularized",
    "fit_hyperuniform_exponent",
    "g_born_from_eps",
    "g_from_phase_function_q",
    "mie_phase_function",
    "mie_g",
]


# ---------------------------------------------------------------------------
# Angular grid + asymmetry integration
# ---------------------------------------------------------------------------
def regular_angle_grid(n_theta: int = 181, n_phi: int = 72,
                       theta_min: float = 0.0, theta_max: float = np.pi):
    """Regular product grid for ``FieldProjectionAngleMonitor``.

    Returns ``(theta, phi)`` 1D arrays (radians) spanning ``[theta_min, theta_max]``
    and ``[0, 2*pi]``.  Pass ``list(theta)`` / ``list(phi)`` to the monitor and the
    SAME arrays to :func:`asymmetry_parameter`.

    Unlike a Fibonacci sphere, this matches how the monitor samples (tensor grid)
    and integrates exactly with ``sin(theta) dtheta dphi`` quadrature.
    """
    theta = np.linspace(theta_min, theta_max, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    return theta, phi


def _as_tpf(intensity):
    """Coerce intensity to shape ``(n_theta, n_phi, n_f)``; report if f was added."""
    intensity = np.asarray(intensity, dtype=float)
    if intensity.ndim == 2:
        return intensity[:, :, None], True
    if intensity.ndim == 3:
        return intensity, False
    raise ValueError(f"intensity must be 2D (theta,phi) or 3D (theta,phi,f); got {intensity.shape}")


def asymmetry_parameter(theta, phi, intensity):
    """``g = <cos theta>`` of an angular intensity on a regular ``(theta, phi)`` grid.

    Parameters
    ----------
    theta, phi : 1D arrays (radians), the monitor axes.
    intensity  : array ``(n_theta, n_phi)`` or ``(n_theta, n_phi, n_f)``.  Should be
                 the **diffuse** intensity ``I_d`` (see :func:`coherent_incoherent`).

    Returns
    -------
    g : scalar or array ``(n_f,)``.

    Notes
    -----
    ``g = [∫ I cosθ sinθ dθ dφ] / [∫ I sinθ dθ dφ]``.  Azimuth integration is done
    first (φ-independent weights factor out), then a sinθ/cosθ-weighted θ integral.
    Trapezoidal quadrature handles non-uniform grids; if ``phi`` does not include the
    endpoint ``2π`` (the default), a closing segment is added so the azimuth integral
    is complete.
    """
    theta = np.asarray(theta, float)
    phi = np.asarray(phi, float)
    I, squeeze = _as_tpf(intensity)

    # integrate over phi (periodic): append the wrap-around point if needed
    if not np.isclose(phi[-1] - phi[0], 2.0 * np.pi):
        phi_c = np.concatenate([phi, [phi[0] + 2.0 * np.pi]])
        I_c = np.concatenate([I, I[:, :1, :]], axis=1)
    else:
        phi_c, I_c = phi, I
    I_phi = np.trapz(I_c, phi_c, axis=1)                      # (n_theta, n_f)

    sin_t = np.sin(theta)[:, None]
    cos_t = np.cos(theta)[:, None]
    num = np.trapz(I_phi * cos_t * sin_t, theta, axis=0)      # (n_f,)
    den = np.trapz(I_phi * sin_t, theta, axis=0)
    g = num / np.where(den == 0, np.nan, den)
    return g[0] if squeeze else g


def phase_function(theta, phi, intensity, normalize: bool = True):
    """Azimuth-averaged phase function ``p(theta)`` (for plotting / inspection).

    Returns ``p`` with shape ``(n_theta,)`` or ``(n_theta, n_f)``.  If ``normalize``,
    ``∫ p sinθ dθ dφ = 1``.
    """
    theta = np.asarray(theta, float)
    phi = np.asarray(phi, float)
    I, squeeze = _as_tpf(intensity)
    p = I.mean(axis=1)                                        # average over phi
    if normalize:
        sin_t = np.sin(theta)[:, None]
        norm = 2.0 * np.pi * np.trapz(p * sin_t, theta, axis=0)
        p = p / np.where(norm == 0, np.nan, norm)
    return p[:, 0] if squeeze else p


# ---------------------------------------------------------------------------
# Coherent / incoherent (diffuse) separation
# ---------------------------------------------------------------------------
def coherent_incoherent(E):
    """Split an ensemble of complex far fields into coherent and diffuse intensity.

    Parameters
    ----------
    E : complex array ``(n_real, ...)`` -- realisation index first.  Typically each
        ``E[i]`` is ``E_theta`` or a stack; you may pass ``E_theta`` and ``E_phi``
        separately and add the returned ``diffuse`` parts.

    Returns
    -------
    dict with ``coherent`` ``|<E>|^2``, ``total`` ``<|E|^2>`` and
    ``diffuse`` ``<|E|^2> - |<E>|^2`` (all shaped like ``E[0]``).

    With a single realisation the coherent lobe cannot be subtracted by ensemble
    averaging; use azimuthal / sub-aperture averaging instead (the slab is
    statistically isotropic in-plane), or -- preferred -- supply several independent
    structures.
    """
    E = np.asarray(E)
    if E.ndim < 2:
        raise ValueError("Pass realisations along axis 0: E.shape == (n_real, ...).")
    mean_E = E.mean(axis=0)
    coherent = np.abs(mean_E) ** 2
    total = (np.abs(E) ** 2).mean(axis=0)
    diffuse = total - coherent
    return {"coherent": coherent, "total": total, "diffuse": diffuse}


# ---------------------------------------------------------------------------
# Structure: power spectrum P(q) and Born g
# ---------------------------------------------------------------------------
def q_of_theta(theta, wavelength, n_bg: float = 1.0):
    """Momentum transfer ``q = 2 k sin(theta/2)`` (rad / length-unit-of-wavelength).

    ``k = 2*pi*n_bg/wavelength``.  ``wavelength`` may be a scalar or array; returns
    shape ``(n_theta,)`` or ``(n_theta, n_lambda)``.
    """
    theta = np.asarray(theta, float)[:, None]
    lam = np.atleast_1d(np.asarray(wavelength, float))[None, :]
    k = 2.0 * np.pi * n_bg / lam
    q = 2.0 * k * np.sin(theta / 2.0)
    return q[:, 0] if q.shape[1] == 1 else q


def power_spectrum_from_eps(eps, L, nbins: int = 256):
    """Radially-averaged power spectrum ``P(q) = <|FFT(eps - <eps>)|^2>`` of a cubic
    permittivity volume.

    In the first Born approximation this is proportional to the single-scattering
    differential cross section of the *actual* index pattern (form factor x structure
    factor together), so it also works for connected networks.

    Parameters
    ----------
    eps   : 3D array (assumed (near-)cubic, isotropic statistics).
    L     : physical box side length (same length unit as the wavelengths you will use).
    nbins : number of radial q bins.

    Returns
    -------
    q : 1D array (rad / length), bin centres (q=0 excluded).
    P : 1D array, radially averaged power spectrum (arbitrary units).
    """
    eps = np.asarray(eps, float)
    delta = eps - eps.mean()
    P3 = np.abs(np.fft.fftn(delta)) ** 2

    # q-grid: fftfreq gives cycles/length -> multiply by 2*pi for rad/length
    axes = []
    for n in eps.shape:
        dx = L / n
        axes.append(2.0 * np.pi * np.fft.fftfreq(n, d=dx))
    QX, QY, QZ = np.meshgrid(*axes, indexing="ij")
    qmag = np.sqrt(QX ** 2 + QY ** 2 + QZ ** 2).ravel()
    Pflat = P3.ravel()

    qmax = qmag.max() / np.sqrt(3.0)        # avoid sparse corner bins
    bins = np.linspace(0.0, qmax, nbins + 1)
    which = np.digitize(qmag, bins) - 1     # 0..nbins-1 inside range
    valid = (which >= 0) & (which < nbins)
    # vectorised radial average: O(N) via bincount, not O(N * nbins)
    sums = np.bincount(which[valid], weights=Pflat[valid], minlength=nbins)
    counts = np.bincount(which[valid], minlength=nbins)
    P = sums / np.where(counts == 0, np.nan, counts)
    qc = 0.5 * (bins[1:] + bins[:-1])
    good = ~np.isnan(P) & (qc > 0)
    return qc[good], P[good]


def g_from_phase_function_q(theta, weight):
    """``g = <cos theta>`` for an azimuthally symmetric weight ``w(theta)``.

    ``weight`` is ``dσ/dΩ`` as a function of ``theta`` (e.g. ``P(q(theta))`` or
    ``|f|^2 S(q)``).  Shape ``(n_theta,)`` or ``(n_theta, n_lambda)``.
    """
    theta = np.asarray(theta, float)
    w = np.asarray(weight, float)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    if w.ndim == 1:
        num = np.trapz(w * cos_t * sin_t, theta)
        den = np.trapz(w * sin_t, theta)
        return num / den
    num = np.trapz(w * (cos_t * sin_t)[:, None], theta, axis=0)
    den = np.trapz(w * sin_t[:, None], theta, axis=0)
    return num / den


def power_spectrum_robust(eps, L, nbins: int = 160, min_modes: int = 20,
                          smooth: bool = True, log_bins: bool = True):
    """Robust radially-averaged power spectrum ``P(q) = <|FFT(eps - <eps>)|^2>``.

    Improvements over :func:`power_spectrum_from_eps` for the **low-q** region that
    matters for a hyperuniform medium (where ``P(q) -> 0`` as ``q -> 0``):

    * **log-spaced** radial bins resolve the small-q rise with many points
      (linear bins put only ~3 bins below the correlation peak and bias the exponent);
    * bins with ``< min_modes`` Fourier modes are dropped (kills the ``q_min`` shell
      counting-noise jitter);
    * returns the per-bin standard error ``P_err = std(|F|^2)/sqrt(count)``;
    * optional shape-preserving Savitzky-Golay smoothing of ``log P`` (kept >= 0).

    Returns ``(q, P, P_err)`` with ``q`` ascending and only well-populated bins.

    Note: windowing/apodizing the field before the FFT was tested and **rejected** --
    mean subtraction already zeroes DC and a disordered field has no edge ramp to leak,
    so a window only smears the correlation peak into the suppressed low-q region and
    raises the effective ``q_min``. Proper binning is what cleans the low-q trend.
    """
    from scipy.signal import savgol_filter
    eps = np.asarray(eps, float)
    delta = eps - eps.mean()
    P3 = np.abs(np.fft.fftn(delta)) ** 2
    axes = [2.0 * np.pi * np.fft.fftfreq(n, d=L / n) for n in eps.shape]
    QX, QY, QZ = np.meshgrid(*axes, indexing="ij")
    qmag = np.sqrt(QX ** 2 + QY ** 2 + QZ ** 2).ravel()
    Pflat = P3.ravel()
    nz = qmag > 0
    qmag, Pflat = qmag[nz], Pflat[nz]
    qmax = qmag.max() / np.sqrt(3.0)         # avoid sparse cube-corner shells

    if log_bins:
        bins = np.geomspace(qmag.min() * 0.999, qmax, nbins + 1)
        qc = np.sqrt(bins[1:] * bins[:-1])   # geometric bin centre
    else:
        bins = np.linspace(0.0, qmax, nbins + 1)
        qc = 0.5 * (bins[1:] + bins[:-1])
    which = np.digitize(qmag, bins) - 1
    valid = (which >= 0) & (which < nbins)
    counts = np.bincount(which[valid], minlength=nbins)
    sums = np.bincount(which[valid], weights=Pflat[valid], minlength=nbins)
    sumsq = np.bincount(which[valid], weights=Pflat[valid] ** 2, minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = sums / np.where(counts == 0, np.nan, counts)
        var = np.clip(sumsq / np.where(counts == 0, np.nan, counts) - P ** 2, 0, None)
        P_err = np.sqrt(var / np.where(counts == 0, np.nan, counts))

    keep = counts >= min_modes
    qc, P, P_err = qc[keep], P[keep], P_err[keep]

    if smooth and P.size >= 7:
        win = min(11, P.size if P.size % 2 == 1 else P.size - 1)
        if win >= 5:
            logP = np.log(np.clip(P, 1e-300, None))
            P = np.clip(np.exp(savgol_filter(logP, win, 3)), 0.0, None)
    return qc, P, P_err


def fit_hyperuniform_exponent(q, P, L=None, q_lo=None, q_hi=None, hi_frac: float = 0.6):
    """Fit the small-q power law ``P(q) ~ C q^alpha`` and return ``(alpha, C, g_inf)``.

    ``g_inf = -alpha/(alpha+4)`` is the long-wavelength Born limit IF the ``q^alpha``
    suppression actually continues to ``q -> 0``.

    CAVEAT: the default fit window ``[2*pi/L, hi_frac*q_peak]`` lies ABOVE the box floor
    ``q_box = 2*pi/L`` and is really the rising flank of the first correlation peak, NOT
    the asymptotic ``q->0`` scaling (the finite box cannot reach ``q->0``). So ``alpha``
    here does NOT by itself establish hyperuniformity. For a non-hyperuniform medium with
    finite ``S(0)`` (e.g. an LSU / local-self-uniform network) the true limit may instead
    be ``g -> 0``. Treat ``g_inf`` as a model value, not a measurement.
    """
    q = np.asarray(q, float)
    P = np.asarray(P, float)
    if q_lo is None:
        q_lo = (2.0 * np.pi / L) if L else q[q > 0].min()
    if q_hi is None:
        pk = (q > 2.0) & (q < 3.4)
        q_peak = q[pk][np.argmax(P[pk])] if np.any(pk) else q[np.nanargmax(P)]
        q_hi = hi_frac * q_peak
    m = (q >= q_lo) & (q <= q_hi) & (P > 0) & np.isfinite(P)
    if m.sum() < 3:
        raise ValueError("too few low-q points to fit; raise nbins or widen the window")
    slope, intercept = np.polyfit(np.log(q[m]), np.log(P[m]), 1)
    alpha = float(slope)
    return alpha, float(np.exp(intercept)), -alpha / (alpha + 4.0)


def power_spectrum_regularized(eps, L, nbins: int = 200, q_cross=None,
                               hi_frac: float = 0.6, n_subgrid: int = 400,
                               return_meta: bool = False):
    """Spectrum with a physically-constrained low-q model enforcing ``P(q) -> 0``.

    Above ``q_cross`` the measured (robust) spectrum is used unchanged; below it the
    box-floor-limited bins are blended (logistic) into the fitted power law
    ``C q^alpha``, and a dense sub-grid on ``[0, q_first)`` carrying the pure power law
    is prepended so the long-wavelength Born integral resolves and tends to
    ``g_inf = -alpha/(alpha+4)``. By default ``q_cross = 2*pi/L`` (model only the
    genuinely unmeasurable sub-floor region).

    *** CAVEAT -- MODEL, NOT DATA: the finite box supports no Fourier modes below
    ``q_box = 2*pi/L``. Every value returned for ``q < q_box`` is pure extrapolation
    encoding the ASSUMPTION that the ``q^alpha`` suppression continues to ``q->0``. That
    assumption holds for hyperuniform media but is NOT established for LSU networks (whose
    ``S(0)`` may be finite, in which case the true long-wavelength limit is ``g->0``). It
    is not a measurement. ***

    Returns ``(q, P)`` (with ``q[0]=0, P[0]=0``), or ``(q, P, meta)`` if ``return_meta``.
    """
    q, P, _ = power_spectrum_robust(eps, L, nbins=nbins, min_modes=10, smooth=False)
    q_box = 2.0 * np.pi / L
    pk = (q > 2.0) & (q < 3.4)
    q_peak = q[pk][np.argmax(P[pk])] if np.any(pk) else q[np.nanargmax(P)]
    alpha, C, g_inf = fit_hyperuniform_exponent(q, P, q_lo=q_box, q_hi=hi_frac * q_peak)
    if q_cross is None:
        q_cross = q_box
    q_cross = max(q_cross, q_box)
    P_law = C * q ** alpha
    width = max(0.25 * q_cross, 0.5 * (q[1] - q[0]))
    s = 1.0 / (1.0 + np.exp(-(q - q_cross) / width))         # 0 below q_cross -> 1 above
    P_blend = (1.0 - s) * P_law + s * P
    qsub = np.linspace(0.0, q[0], n_subgrid, endpoint=False)
    q_out = np.concatenate([qsub, q])
    P_out = np.concatenate([C * qsub ** alpha, P_blend])
    if return_meta:
        return q_out, P_out, dict(alpha=alpha, C=C, q_peak=q_peak,
                                  q_cross=q_cross, q_box=q_box, g_inf=g_inf)
    return q_out, P_out


def g_born_from_eps(eps, L, wavelengths, n_bg: float = 1.0,
                    n_theta: int = 721, nbins: int = 256,
                    spectrum: str = "raw", allow_extrapolation: bool = False):
    """Born-approximation ``g(lambda)`` directly from a permittivity volume.

    Maps ``q(theta) = 2k sin(theta/2)`` per wavelength and integrates
    ``g = <cos theta>`` weighted by ``P(q(theta))``.

    ``spectrum``:
        * ``"raw"``         -- linear-binned :func:`power_spectrum_from_eps` (default;
          pure data, keeps all low-q bins; NaN below the box floor);
        * ``"robust"``      -- log-binned :func:`power_spectrum_robust` -- best for the
          spectrum plot / exponent, but it drops sparse low-q bins so the ``g`` integral
          clamps at low ``nu`` (use for diagnostics, not for ``g`` below ``nu ~ 0.4``);
        * ``"regularized"`` -- :func:`power_spectrum_regularized`, whose low-q model
          enforces ``P(0)=0`` so ``g`` is smooth into the long-wavelength limit
          (model-based below the box floor; pair with ``allow_extrapolation=True``).

    ``allow_extrapolation`` only applies to ``spectrum="regularized"``: if True, the
    sub-box-floor wavelengths are evaluated on the **model** spectrum (label as
    extrapolation); otherwise wavelengths with ``2k < 2*pi/L`` are returned as ``NaN``.

    Returns ``g`` shaped like ``np.atleast_1d(wavelengths)`` (scalar in -> scalar out).
    """
    if spectrum == "raw":
        qP, P = power_spectrum_from_eps(eps, L, nbins=nbins)
    elif spectrum == "robust":
        qP, P, _ = power_spectrum_robust(eps, L, nbins=nbins)
    elif spectrum == "regularized":
        qP, P = power_spectrum_regularized(eps, L, nbins=nbins)
    else:
        raise ValueError("spectrum must be 'raw', 'robust', or 'regularized'")

    lams = np.atleast_1d(np.asarray(wavelengths, float))
    theta = np.linspace(0.0, np.pi, n_theta)
    # Fundamental q supported by the finite box (smallest non-zero mode). Below 2k ~ q_box
    # the accessible window [0, 2k] contains essentially no Fourier modes and the result
    # would be a box-limited extrapolation -> flag those wavelengths as NaN, UNLESS the
    # regularized model is explicitly allowed to extrapolate there.
    q_box = 2.0 * np.pi / L
    model_ok = (spectrum == "regularized") and allow_extrapolation
    n_unres = 0
    g = np.empty(lams.size)
    for j, lam in enumerate(lams):
        k = 2.0 * np.pi * n_bg / lam
        if (2.0 * k < q_box) and not model_ok:               # long-wavelength / finite-box floor
            g[j] = np.nan
            n_unres += 1
            continue
        q = 2.0 * k * np.sin(theta / 2.0)
        w = np.interp(q, qP, P, left=P[0], right=P[-1])      # P(q(theta))
        g[j] = g_from_phase_function_q(theta, w)
    if n_unres:
        print(f"g_born_from_eps: {n_unres} wavelength(s) below the finite-box floor "
              f"(2k < q_box = {q_box:.3f} rad/length, box L = {L}); set to NaN. "
              f"Use a larger structure box to reach those frequencies.")
    return g[0] if np.isscalar(wavelengths) or lams.size == 1 else g


# ---------------------------------------------------------------------------
# Mie reference (validation of the FDTD -> NTFF pipeline). Bohren & Huffman.
# ---------------------------------------------------------------------------
def _mie_ab(x: float, m: complex, nmax: int | None = None):
    """Mie coefficients a_n, b_n for size parameter x and relative index m."""
    if nmax is None:
        nmax = int(np.ceil(x + 4.0 * x ** (1.0 / 3.0) + 2.0))
    n = np.arange(1, nmax + 1)
    mx = m * x

    # logarithmic derivative D_n(mx) by downward recurrence
    nmx = int(max(nmax, abs(mx)) + 16)
    D = np.zeros(nmx + 1, dtype=complex)
    for k in range(nmx, 0, -1):
        D[k - 1] = k / mx - 1.0 / (D[k] + k / mx)
    Dn = D[1:nmax + 1]

    # Riccati-Bessel psi, chi by upward recurrence
    psi = np.zeros(nmax + 1)
    chi = np.zeros(nmax + 1)
    psi_m1, chi_m1 = np.cos(x), -np.sin(x)      # psi_{-1}, chi_{-1}
    psi0, chi0 = np.sin(x), np.cos(x)           # psi_0, chi_0
    for k in range(1, nmax + 1):
        psi[k] = (2 * k - 1) / x * psi0 - psi_m1
        chi[k] = (2 * k - 1) / x * chi0 - chi_m1
        psi_m1, psi0 = psi0, psi[k]
        chi_m1, chi0 = chi0, chi[k]
    psin = psi[1:]
    psin_m1 = np.concatenate([[np.sin(x)], psi[1:nmax]])
    chin = chi[1:]
    chin_m1 = np.concatenate([[np.cos(x)], chi[1:nmax]])
    xi = psin - 1j * chin
    xi_m1 = psin_m1 - 1j * chin_m1

    a = ((Dn / m + n / x) * psin - psin_m1) / ((Dn / m + n / x) * xi - xi_m1)
    b = ((Dn * m + n / x) * psin - psin_m1) / ((Dn * m + n / x) * xi - xi_m1)
    return n, a, b


def mie_phase_function(theta, radius, m, wavelength, n_med: float = 1.0):
    """Unpolarised Mie phase function |S1|^2+|S2|^2 vs theta (for NTFF validation).

    ``radius``, ``wavelength`` in the same length unit; ``m = n_sphere/n_med``.
    Returns array shaped like ``theta``.
    """
    theta = np.asarray(theta, float)
    x = 2.0 * np.pi * n_med * radius / wavelength
    n, a, b = _mie_ab(x, m / n_med)
    cos_t = np.cos(theta)
    S1 = np.zeros(theta.shape, dtype=complex)
    S2 = np.zeros(theta.shape, dtype=complex)
    # angular functions pi_n, tau_n by standard upward recurrence
    pi_prev = np.zeros_like(theta)              # pi_0 = 0
    pi_cur = np.ones_like(theta)               # pi_1 = 1
    for idx, nn in enumerate(n):
        nn = int(nn)
        if nn > 1:
            pi_new = ((2 * nn - 1) / (nn - 1) * cos_t * pi_cur
                      - nn / (nn - 1) * pi_prev)
            pi_prev, pi_cur = pi_cur, pi_new
        # else pi_cur already holds pi_1 = 1
        tau = nn * cos_t * pi_cur - (nn + 1) * pi_prev
        fac = (2 * nn + 1) / (nn * (nn + 1))
        S1 += fac * (a[idx] * pi_cur + b[idx] * tau)
        S2 += fac * (a[idx] * tau + b[idx] * pi_cur)
    return (np.abs(S1) ** 2 + np.abs(S2) ** 2)


def mie_g(radius, m, wavelength, n_med: float = 1.0):
    """Analytic Mie asymmetry parameter g = <cos theta> (Bohren & Huffman 4.62)."""
    x = 2.0 * np.pi * n_med * radius / wavelength
    n, a, b = _mie_ab(x, m / n_med)
    nf = n.astype(float)
    qsca = (2.0 / x ** 2) * np.sum((2 * nf + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2))
    term1 = (nf[:-1] * (nf[:-1] + 2) / (nf[:-1] + 1)
             * np.real(a[:-1] * np.conj(a[1:]) + b[:-1] * np.conj(b[1:])))
    term2 = ((2 * nf + 1) / (nf * (nf + 1)) * np.real(a * np.conj(b)))
    g = (4.0 / (x ** 2 * qsca)) * (np.sum(term1) + np.sum(term2))
    return g
