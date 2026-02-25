import numpy as np
from sympy import I, exp, log, pi, sqrt

from devito import Eq, grad
from devito.tools import as_tuple

from fields import frequencies
from fields_exprs import sub_time


def func_name(freq=None, ic="as", is_viscoacoustic=False):
    """
    Get key for imaging condition/linearized source function
    """
    if is_viscoacoustic:
        if freq is None:
            return "%s_%s" % (ic, "visco")
        else:
            return "%s_%s_%s" % (ic, "visco", "freq")
    else:
        if freq is None:
            return str(ic)
        else:
            return "%s_%s" % (ic, "freq")


def grad_expr(gradm, u, v, model, w=None, f0=0.015, freq=None, dft_sub=None, ic="as"):
    """
    Gradient update stencil

    Parameters
    ----------
    u: TimeFunction or Tuple
        Forward wavefield (tuple of fields for TTI or dft)
    v: TimeFunction or Tuple
        Adjoint wavefield (tuple of fields for TTI)
    model: Model
        Model structure
    w: Float or Expr (optional)
        Weight for the gradient expression (default=1)
    freq: Array
        Array of frequencies for on-the-fly DFT
    factor: int
        Subsampling factor for DFT
    isic: Bool
        Whether or not to use inverse scattering imaging condition (not supported yet)
    """
    ic_func = ic_dict[func_name(freq=freq, ic=ic, is_viscoacoustic=model.is_viscoacoustic)]
    u, v = as_tuple(u), as_tuple(v)
    expr = ic_func(u, v, model, f0=f0, freq=freq, factor=dft_sub, w=w)
    eq_g = [Eq(gradm, gradm - expr, subdomain=model.physical)]
    return eq_g


def crosscorr_time(u, v, model, **kwargs):
    """
    Cross correlation of forward and adjoint wavefield

    Parameters
    ----------
    u: TimeFunction or Tuple
        Forward wavefield (tuple of fields for TTI or dft)
    v: TimeFunction or Tuple
        Adjoint wavefield (tuple of fields for TTI)
    model: Model
        Model structure
    """
    w = kwargs.get('w') or u[0].indices[0].spacing * model.irho
    return w * sum(vv.dt2 * uu for uu, vv in zip(u, v))


def crosscorr_freq(u, v, model, freq=None, dft_sub=None, **kwargs):
    """
    Standard cross-correlation imaging condition with on-th-fly-dft

    Parameters
    ----------
    u: TimeFunction or Tuple
        Forward wavefield (tuple of fields for TTI or dft)
    v: TimeFunction or Tuple
        Adjoint wavefield (tuple of fields for TTI)
    model: Model
        Model structure
    freq: Array
        Array of frequencies for on-the-fly DFT
    factor: int
        Subsampling factor for DFT
    """
    # Subsampled dft time axis
    time = model.grid.time_dim
    dt = time.spacing
    tsave, factor = sub_time(time, dft_sub)
    expr = 0

    fdim = as_tuple(u)[0][0].dimensions[0]
    f, _ = frequencies(freq, fdim=fdim)
    omega_t = 2*np.pi*f*tsave*factor*dt
    # Gradient weighting is (2*np.pi*f)**2/nt
    w = -(2*np.pi*f)**2/time.symbolic_max

    for uu, vv in zip(u, v):
        expr += w*uu*exp(1j*omega_t)*vv
    return expr


def isic_time(u, v, model, **kwargs):
    """
    Inverse scattering imaging condition

    Parameters
    ----------
    u: TimeFunction or Tuple
        Forward wavefield (tuple of fields for TTI or dft)
    v: TimeFunction or Tuple
        Adjoint wavefield (tuple of fields for TTI)
    model: Model
        Model structure
    """
    w = u[0].indices[0].spacing * model.irho
    ics = kwargs.get('icsign', 1)
    return w * sum(uu * vv.dt2 * model.m + ics * inner_grad(uu, vv)
                   for uu, vv in zip(u, v))


def isic_freq(u, v, model, **kwargs):
    """
    Inverse scattering imaging condition

    Parameters
    ----------
    u: TimeFunction or Tuple
        Forward wavefield (tuple of fields for TTI or dft)
    v: TimeFunction or Tuple
        Adjoint wavefield (tuple of fields for TTI)
    model: Model
        Model structure
    """
    ics = kwargs.get('icsign', 1)
    freq = kwargs.get('freq')
    # Subsampled dft time axis
    time = model.grid.time_dim
    dt = time.spacing
    tsave, factor = sub_time(time, kwargs.get('factor'))
    fdim = as_tuple(u)[0][0].dimensions[0]
    f, nfreq = frequencies(freq, fdim=fdim)
    omega_t = 2*np.pi*f*tsave*factor*dt
    w = -(2*np.pi*f)**2/time.symbolic_max
    w2 = ics * factor / time.symbolic_max

    expr = 0
    for uu, vv in zip(u, v):
        idftu = uu * exp(1j*omega_t)
        expr += w * idftu * vv * model.m - w2 * inner_grad(idftu, vv)
    return expr


def lin_src(model, u, ic="as"):
    """
    Source for linearized modeling

    Parameters
    ----------
    model: Model
        Model containing the perturbation dm
    u: TimeFunction or Tuple
        Forward wavefield (tuple of fields for TTI or dft)
    ic: String
        Imaging condition of which we compute the linearized source
    """
    ls_func = ls_dict[func_name(ic=str(ic), is_viscoacoustic=model.is_viscoacoustic)]
    return ls_func(model, as_tuple(u))


def basic_src(model, u, **kwargs):
    """
    Basic source for linearized modeling

    Parameters
    ----------
    model: Model
        Model containing the perturbation dm
    u: TimeFunction or Tuple
        Forward wavefield (tuple of fields for TTI or dft)
    """
    w = -model.dm * model.irho
    if model.is_tti:
        return (w * u[0].dt2, w * u[1].dt2)
    return w * u[0].dt2


def isic_src(model, u, **kwargs):
    """
    ISIC source for linearized modeling

    Parameters
    ----------
    model: Model
        Model containing the perturbation dm
    u: TimeFunction or Tuple
        Forward wavefield (tuple of fields for TTI or dft)
    """
    m, dm, irho = model.m, model.dm, model.irho
    ics = kwargs.get('icsign', 1)
    dus = []
    for uu in u:
        dus.append(dm * irho * uu.dt2 * m - ics * laplacian(uu, dm * irho))
    if model.is_tti:
        return (-dus[0], -dus[1])
    return -dus[0]


def inner_grad(u, v):
    """
    Inner product of the gradient of two fields

    Parameters
    ----------
    u: TimeFunction
        First field
    v: TimeFunction
        Second field
    """
    return grad(u, shift=.5).dot(grad(v, shift=.5))


def Loss(dsyn, dobs, dt, is_residual=False, misfit=None):
    """
    L2 loss and residual between the synthetic data dsyn and observed data dobs

    Parameters
    ----------

    dsyn: SparseTimeFunction or tuple
        Synthetic data or tuple (background, linearized) data
    dobs: SparseTimeFunction
        Observed data
    dt: float
        Time sampling rate
    is_residual: bool
        Whether input dobs is already the data residual
    misfit: function
        User provided function of the form:
        misifit(dsyn, dob) -> obj, adjoint_source
    """
    if misfit is not None:
        if isinstance(dsyn, tuple):
            f, r = misfit(dsyn[0].data._local, dobs - dsyn[1].data._local[:])
            dsyn[0].data._local[:] = r
            return dt * f, dsyn[0].data._local
        else:
            f, r = misfit(dsyn.data._local, dobs)
            dsyn.data._local[:] = r
            return dt * f, dsyn.data._local

    if not is_residual:
        if isinstance(dsyn, tuple):
            # input is observed data
            dsyn[0].data._local[:] -= dobs - dsyn[1].data._local[:]
            phi = .5 * dt * np.linalg.norm(dsyn[0].data._local)**2
            return phi, dsyn[0].data._local
        else:
            dsyn.data._local[:] -= dobs   # input is observed data
    else:
        dsyn.data._local[:] = dobs

    return .5 * dt * np.linalg.norm(dsyn.data._local)**2, dsyn.data._local


# =================== QFWI VISCO GRADIENTS ===================


def _beta_kf(freq_hz, f0_hz):
    """β(ω) = i - (2/π) log(ω/ω0) for the Kolsky-Futterman model."""
    return I - (2 / pi) * log(freq_hz / f0_hz)


def _keating_base_terms(p, p_adj, model, beta):
    """
    Symbolic kernels for gradients in Keating & Innanen (2019).

    Gradients are first formed in (s_c0, s_Q) with:
      s_c0 = γ c0^-2 = γ m,  s_Q = Q^-1 = 1/qp,
    then mapped by chain rule to (vp, qp) used by JUDI.
    """
    vp = 1 / sqrt(model.m)
    qp = model.qp
    sq = 1 / qp
    gamma = 1e6

    # Paper equations (6)-(7): dphi/ds_c0 and dphi/ds_Q kernels
    g_sc0 = (1 / gamma) * (1 + beta * sq) * p * p_adj
    g_sq = (beta * (gamma * model.m)) * p * p_adj

    # Chain rule to requested parameters vp and qp.
    dsc0_dvp = -2 * gamma / vp**3
    dsq_dqp = -1 / qp**2
    g_vp = g_sc0 * dsc0_dvp
    g_qp = g_sq * dsq_dqp
    return g_vp, g_qp


def isic_visco_time(u, v, model, **kwargs):
    """
    Time-domain QFWI gradients for Vp and Qp, based on Eq. (16) with τ(Qp).

    Returns a dict with keys used by `grad_expr_multi`:
      - ``grad_m``: Vp-sensitive update (stored in gradm slot)
      - ``grad_q``: Qp-sensitive update
    """
    w = kwargs.get('w') or as_tuple(u)[0].indices[0].spacing * model.irho
    p = as_tuple(u)[0]
    p_adj = as_tuple(v)[0]

    # Time-domain implementation: narrow-band approximation around f0,
    # where log(ω/ω0)=0 and β≈i.
    f0 = kwargs.get('f0', 0.015)
    beta0 = _beta_kf(f0, f0)

    # Use p_tt as the time-domain analogue of ω² u(ω).
    gv, gqp = _keating_base_terms(p.dt2, p_adj, model, beta0)
    return {"grad_m": w * gv, "grad_q": w * gqp}


def isic_visco_freq(u, v, model, freq=None, dft_sub=None, **kwargs):
    """
    Frequency-domain QFWI gradients (weighted IDFT accumulation).
    """
    p = as_tuple(u)[0]
    p_adj = as_tuple(v)[0]
    time = model.grid.time_dim
    tsave, factor = sub_time(time, dft_sub)
    fdim = as_tuple(u)[0][0].dimensions[0]
    f, _ = frequencies(freq, fdim=fdim)
    omega = 2 * np.pi * f
    f0 = kwargs.get('f0', 0.015)
    beta = _beta_kf(f, f0)

    omega_t = omega * tsave * factor * time.spacing
    idft_weight = exp(1j * omega_t)
    spectral_w = -(omega**2) / time.symbolic_max

    gv_base, gqp_base = _keating_base_terms(p, p_adj, model, beta)
    return {
        "grad_m": spectral_w * idft_weight * gv_base,
        "grad_q": spectral_w * idft_weight * gqp_base
    }


def isic_visco_src(model, u, **kwargs):
    """
    Linearized visco source term.

    For viscoacoustic modeling in this code path we inject perturbations through
    ``dm`` (Vp-related) as before, while Qp updates are handled in adjoint gradient
    accumulation (``grad_q``).
    """
    return isic_src(model, u, **kwargs)

def grad_expr_multi(grad_dict, u, v, model, w=None, f0=0.015, freq=None, dft_sub=None, ic="as"):
    """
    Gradient expression for multiple parameters (viscoacoustic extension).
    
    Handles both acoustic and viscoacoustic cases with proper parameter separation.
    """
    # For viscoacoustic models always use visco imaging condition
    ic_func = ic_dict[func_name(freq=freq, ic=ic, is_viscoacoustic=model.is_viscoacoustic)]
    
    # Compute gradient expressions
    u_tuple, v_tuple = as_tuple(u), as_tuple(v)
    
    # Basic kwargs
    ic_kwargs = {'w': w}
    if freq is not None:
        ic_kwargs['freq'] = freq
        ic_kwargs['factor'] = dft_sub
    ic_kwargs['f0'] = f0
    
    expr = ic_func(u_tuple, v_tuple, model, **ic_kwargs)
    
    # Create equations for each gradient parameter
    eqs = []
    if isinstance(expr, dict):
        # Multiple gradients (viscoacoustic case)
        for param_name, expr_part in expr.items():
            if param_name in grad_dict:
                eqs.append(Eq(grad_dict[param_name], 
                             grad_dict[param_name] - expr_part, 
                             subdomain=model.physical))
    else:
        # Single gradient (acoustic case) - backward compatibility
        if "grad_m" in grad_dict:
            eqs.append(Eq(grad_dict["grad_m"], 
                         grad_dict["grad_m"] - expr, 
                         subdomain=model.physical))
    
    return eqs


fwi_src = lambda *ar, **kw: isic_src(*ar, icsign=-1, **kw)
fwi_time = lambda *ar, **kw: isic_time(*ar, icsign=-1, **kw)
fwi_freq = lambda *ar, **kw: isic_freq(*ar, icsign=-1, **kw)

# For viscoacoustic AS path: reuse visco time/freq/source implementations
as_visco_time = lambda *ar, **kw: isic_visco_time(*ar, icsign=-1, **kw)
as_visco_freq = lambda *ar, **kw: isic_visco_freq(*ar, icsign=-1, **kw)
as_visco_src = lambda *ar, **kw: isic_visco_src(*ar, icsign=-1, **kw)

# For FWI (minimization): icsign = -1
fwi_visco_time = lambda *ar, **kw: isic_visco_time(*ar, icsign=-1, **kw)
fwi_visco_freq = lambda *ar, **kw: isic_visco_freq(*ar, icsign=-1, **kw)
fwi_visco_src = lambda *ar, **kw: isic_visco_src(*ar, icsign=-1, **kw)

# For RTM (imaging): icsign = 1
rtm_visco_time = lambda *ar, **kw: isic_visco_time(*ar, icsign=1, **kw)
rtm_visco_freq = lambda *ar, **kw: isic_visco_freq(*ar, icsign=1, **kw)
rtm_visco_src = lambda *ar, **kw: isic_visco_src(*ar, icsign=1, **kw)

ic_dict = {"isic_freq": isic_freq, 
           "as_freq": crosscorr_freq,
           "fwi": fwi_time, 
           "fwi_freq": fwi_freq,
           "isic": isic_time, 
           "as": crosscorr_time, 
           "isic_visco": isic_visco_time,
           "isic_visco_freq": isic_visco_freq,
           "as_visco": as_visco_time,
           "as_visco_freq": as_visco_freq,
           "fwi_visco": fwi_visco_time,
           "fwi_visco_freq": fwi_visco_freq,
           "rtm_visco": rtm_visco_time,
           "rtm_visco_freq": rtm_visco_freq}
ls_dict = {"isic": isic_src, 
           "fwi": fwi_src, 
           "as": basic_src,
           "isic_visco": isic_visco_src,
           "as_visco": as_visco_src,
           "fwi_visco": fwi_visco_src,
           "rtm_visco": rtm_visco_src}
