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



def isic_visco_time(u, v, model, f0=0.015, **kwargs):
    """
    Time-domain viscoacoustic gradients for (m, qp).

    The implementation follows the generalized-Zener split used in QFWI:
      - velocity-like term from the elastic component,
      - attenuation-like term from memory-variable coupling.

    Notes
    -----
    The current propagator stores a single pressure wavefield in forward mode.
    If a forward memory field is not available in `u`, we use `p.dt` as a stable
    proxy for the dissipative state variable.
    """
    p = as_tuple(u)[0]
    p_adj = as_tuple(v)[0]

    dtw = kwargs.get('w') or p.indices[0].spacing
    w = dtw * model.irho
    qinv = 1.0 / model.qp

    has_forward_memory = len(as_tuple(u)) > 1
    xi = as_tuple(u)[1] if has_forward_memory else p.dt

    # dJ/dm: dominant kinematic (elastic) contribution + visco coupling correction
    grad_m = w * (p_adj.dt2 * p - qinv * p_adj * xi)

    # dJ/d(1/Q) mapped to qp through dqinv/dqp = -1/qp^2
    grad_qinv = -w * model.m * p_adj * xi
    grad_q = grad_qinv / (model.qp**2)

    return {"grad_m": grad_m, "grad_q": grad_q}


def isic_visco_freq(u, v, model, f0=0.015, freq=None, dft_sub=None, **kwargs):
    """
    On-the-fly DFT variant of viscoacoustic gradients for (m, qp).
    """
    p = as_tuple(u)[0]
    p_adj = as_tuple(v)[0]

    time = model.grid.time_dim
    tsave, factor = sub_time(time, dft_sub)
    fdim = as_tuple(u)[0][0].dimensions[0]
    f, _ = frequencies(freq, fdim=fdim)
    omega = 2*np.pi*f
    omega_t = omega * tsave * factor * time.spacing

    qinv = 1.0 / model.qp
    w = -(omega**2) / time.symbolic_max
    idftu = p * exp(1j * omega_t)

    # In frequency mode no explicit memory state is carried in saved fields;
    # keep a consistent proxy through spectral weighting of pressure.
    xi = 1j * omega * idftu

    grad_m = w * (idftu * p_adj - qinv * p_adj * xi)
    grad_qinv = -w * model.m * p_adj * xi
    grad_q = grad_qinv / (model.qp**2)

    return {"grad_m": grad_m, "grad_q": grad_q}


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
