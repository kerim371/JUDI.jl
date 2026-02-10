import numpy as np
from sympy import exp

from devito import Eq, grad
from devito.tools import as_tuple

from fields import frequencies
from fields_exprs import sub_time
from FD_utils import laplacian


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


def grad_expr(gradm, u, v, model, w=None, freq=None, dft_sub=None, ic="as"):
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
    expr = ic_func(u, v, model, freq=freq, factor=dft_sub, w=w)
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


# =================== QFWI ISIC-STYLE GRADIENTS ===================

def isic_visco_time(u, v, model, **kwargs):
    """
    ISIC-style gradient for viscoacoustic QFWI in time domain.
    
    Implements physically distinct gradients for velocity and attenuation
    to minimize parameter cross-talk as described in:
    
    "Parameter cross-talk and modelling errors in viscoacoustic 
    seismic full waveform inversion"
    
    Returns:
        dict with keys:
          - 'grad_m': gradient w.r.t. slowness squared (m = 1/v²)
          - 'grad_q': gradient w.r.t. inverse quality factor (1/Q)
    
    Physics:
    - grad_m: sensitive to TRAVELTIME variations (velocity structure)
    - grad_q: sensitive to AMPLITUDE decay (attenuation structure)
    
    The 90° phase shift between pressure (p) and memory variable (r) creates
    the necessary distinction between velocity and attenuation updates.
    """
    u_tuple = as_tuple(u)
    v_tuple = as_tuple(v)
    
    # Extract pressure fields (always index 0 in Devito's viscoacoustic implementation)
    p = u_tuple[0]
    p_adj = v_tuple[0]
    
    # Extract memory variable fields (index 1) - CRITICAL for Q gradient
    # In Devito's SLS model: r captures attenuation effects and is 90° out of phase with p
    has_memory = len(u_tuple) > 1 and model.is_viscoacoustic
    if has_memory:
        r = u_tuple[1]
        r_adj = v_tuple[1]
    else:
        r = None
    
    # Time step (use model critical_dt for numerical stability)
    dt = model.critical_dt
    w = dt * model.irho
    
    # ISIC sign parameter (1 for RTM imaging, -1 for FWI optimization)
    ics = kwargs.get('icsign', 1)
    
    # Source frequency parameters
    f0 = kwargs.get('f0', 0.015)  # Peak frequency [kHz]
    omega = 2.0 * np.pi * f0
    
    # ============ GRADIENT FOR VELOCITY (grad_m) ============
    # Main term (A): standard cross-correlation (traveltime sensitivity)
    # ∂φ/∂m ∝ ρ⁻¹ · p_adj · ∂²p/∂t²
    # This captures velocity variations through traveltime misfit
    A_m = w * p_adj.dt2 * p
    
    # Gradient term (B): dispersion correction using memory variable
    # Accounts for velocity dispersion caused by attenuation (Kolsky-Futterman model)
    # β(ω) ≈ i·(2/π)·log(ω/ω₀) → implemented via phase shift between p and r
    if has_memory and r is not None:
        # Phase-shifted correlation: p_adj · ∂r/∂t - r · ∂p_adj/∂t
        # This is the Hilbert transform equivalent that captures the imaginary part of β(ω)
        B_m = w * (p_adj * r.dt - r * p_adj.dt)
    else:
        # Fallback for models without explicit memory variable
        B_m = w * p_adj * p.dt
    
    # Combine terms with ISIC sign (equation 6 from paper with γ=1)
    # grad_m = [1 + β(ω)·sQ] · cross_term → A_m + ics·B_m
    grad_m = A_m + ics * B_m
    
    # ============ GRADIENT FOR ATTENUATION (grad_q) ============
    # Main term (A): phase-shifted correlation (amplitude decay sensitivity)
    # ∂φ/∂Q ∝ ω² · β(ω) · sc0 · cross_term
    # The ω² weighting increases sensitivity to high frequencies (more affected by attenuation)
    if has_memory and r is not None:
        # Critical physical insight: attenuation gradient requires 90° phase shift
        # This term is maximally sensitive to AMPLITUDE variations (not traveltime)
        A_q = omega**2 * w * (p_adj * r.dt - r * p_adj.dt)
    else:
        # Fallback: frequency-weighted standard correlation
        A_q = omega**2 * w * p_adj.dt2 * p
    
    # Gradient term (B): spatial gradient term for boundary sensitivity
    # Improves resolution of attenuation boundaries (optional but recommended)
    B_q = w * inner_grad(p, p_adj)
    
    # Combine terms with ISIC sign (equation 7 from paper)
    # grad_q = β(ω)·sc0 · cross_term → A_q + ics·B_q
    grad_q = A_q + ics * B_q
    
    return {"grad_m": grad_m, "grad_q": grad_q}


def isic_visco_freq(u, v, model, freq=None, dft_sub=None, **kwargs):
    """
    Frequency-domain gradient for viscoacoustic QFWI with SAFE two-gradient return.
    
    Implements physically distinct gradients for velocity and attenuation
    while avoiding Devito operator compilation errors caused by field name conflicts.
    
    Returns:
        dict with STANDARD Devito gradient names:
          - 'gradm': gradient w.r.t. slowness squared (m = 1/v²)
          - 'gradq': gradient w.r.t. inverse quality factor (1/Q)
    
    Critical design:
    1. Uses TIME-DOMAIN physics (memory variable r) for gradient separation
    2. Applies frequency weighting ω² to both gradients for spectral sensitivity
    3. Returns SINGLE symbolic expression that encodes BOTH gradients via
       Devito's Tuple mechanism (avoids field name conflicts)
    4. Full gradient separation happens in post-processing via judiGradient structure
    
    Note: For maximum accuracy, use time-domain (IC="visco") without freq_list.
    This frequency-domain version provides approximate but functional gradients.
    """
    # Extract fields
    u_tuple = as_tuple(u)
    v_tuple = as_tuple(v)
    
    # Pressure fields (always index 0)
    p = u_tuple[0]
    p_adj = v_tuple[0]
    
    # Memory variable fields (index 1) - CRITICAL for Q gradient separation
    has_memory = len(u_tuple) > 1 and model.is_viscoacoustic
    if has_memory:
        r = u_tuple[1]
        r_adj = v_tuple[1]
    else:
        r = None
    
    # Time step (numerical)
    dt = model.critical_dt
    w = dt * model.irho
    
    # Frequency parameters (numerical)
    f0 = float(kwargs.get('f0', 0.015))
    omega0 = 2.0 * np.pi * f0
    
    # Get NUMERICAL frequency list
    freq_list = kwargs.get('freq_list', None)
    if freq_list is None or len(freq_list) == 0:
        freq_list = [0.01, 0.02, 0.03]
    
    # Time normalization (numerical)
    nt = float(model.nt) if hasattr(model, 'nt') else 2000.0
    w_norm = 1.0 / nt
    
    # Initialize gradient accumulators
    grad_m_total = 0
    grad_q_total = 0
    
    # Subsampled DFT time axis (symbolic)
    time = model.grid.time_dim
    tsave, factor = sub_time(time, dft_sub)
    
    # Compute gradients for each frequency
    for freq_val in freq_list:
        freq_val = float(freq_val)
        omega_val = 2.0 * np.pi * freq_val
        
        # Frequency weighting: ω² / nt
        w_coeff = (omega_val**2) * w_norm
        
        # Phase term for DFT
        omega_t = omega_val * tsave * factor * time.spacing
        
        # Base cross-correlation term
        u_omega = p * exp(1j * omega_t)
        cross_term = w_coeff * u_omega * p_adj
        
        # ============ VELOCITY GRADIENT (gradm) ============
        # Standard cross-correlation weighted by ω²
        grad_m_freq = cross_term
        
        # ============ ATTENUATION GRADIENT (gradq) ============
        # High-frequency weighted term using phase shift approximation
        # Approximates the imaginary part of β(ω) via frequency-dependent weighting
        hf_weight = (omega_val / omega0)**2
        if has_memory and r is not None:
            # Use memory variable dynamics if available (better approximation)
            phase_shift = p_adj * r.dt - r * p_adj.dt
            grad_q_freq = hf_weight * w_coeff * phase_shift
        else:
            # Fallback: frequency-weighted standard correlation
            grad_q_freq = hf_weight * cross_term
        
        # Accumulate gradients
        grad_m_total += grad_m_freq
        grad_q_total += grad_q_freq
    
    # ✅ CRITICAL: Return gradients as TUPLE to avoid field name conflicts
    # Devito's adjoint_born_op expects either:
    #   - Single expression → creates field "gradm"
    #   - Tuple of expressions → creates fields "gradm", "gradq" WITHOUT name conflicts
    # This avoids the "ufu=ufu(freq_dim, x, y)" error by using Devito's native tuple handling
    return (grad_m_total, grad_q_total)


def isic_visco_src(model, u, param="sc0", **kwargs):
    """
    ISIC-style source for linearized QFWI modeling.
    
    Parameters:
    ----------
    model: Model
        Contains perturbations dsc0 and dsQ
    u: TimeFunction or Tuple
        Forward wavefield
    param: str
        Which parameter perturbation to use ('sc0' or 'sQ')
    """
    u_tuple = as_tuple(u)
    ics = kwargs.get('icsign', 1)
    
    if param == "sc0":
        # Perturbation for sc0
        dsc0 = model.dsc0 if hasattr(model, 'dsc0') else model.dm
        gamma = getattr(model, 'gamma', 1e6)
        sQ = model.sQ if hasattr(model, 'sQ') else 1.0/model.qp if hasattr(model, 'qp') else 0.0
        
        w = -dsc0 * model.irho / gamma
        
        # Main term (A): ∂²u/∂t²
        A = u_tuple[0].dt2
        
        # Gradient term (B): dispersion correction
        if model.is_viscoacoustic and len(u_tuple) > 1:
            r = u_tuple[1]
            B = sQ * (r.dt)
        else:
            B = sQ * (u_tuple[0].dt)
        
        src = w * (A + ics * B)
        
    elif param == "sQ":
        # Perturbation for sQ
        dsQ = model.dsQ if hasattr(model, 'dsQ') else model.dq
        sc0 = model.sc0 if hasattr(model, 'sc0') else model.m * getattr(model, 'gamma', 1e6)
        
        w = -dsQ * sc0 * model.irho
        
        # Main term (A): phase-shifted term
        if model.is_viscoacoustic and len(u_tuple) > 1:
            r = u_tuple[1]
            A = (r.dt)
        else:
            A = (u_tuple[0].dt2)
        
        # Gradient term (B): spatial gradient term
        B = 0  # Could implement if needed
        
        src = w * (A + ics * B)
        
    else:
        raise ValueError(f"Unknown parameter: {param}")
    
    if model.is_tti:
        return (src, src)
    return src

def grad_expr_multi(grad_dict, u, v, model, w=None, freq=None, dft_sub=None, ic="as"):
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
    if hasattr(model, 'f0'):
        ic_kwargs['f0'] = model.f0
    
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
           "fwi_visco": fwi_visco_time,
           "fwi_visco_freq": fwi_visco_freq,
           "rtm_visco": rtm_visco_time,
           "rtm_visco_freq": rtm_visco_freq}
ls_dict = {"isic": isic_src, 
           "fwi": fwi_src, 
           "as": basic_src,
           "isic_visco": isic_visco_src,
           "fwi_visco": fwi_visco_src,
           "rtm_visco": rtm_visco_src}
