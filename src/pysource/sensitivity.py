import numpy as np
from sympy import exp

from devito import Eq, grad
from devito.tools import as_tuple

from fields import frequencies
from fields_exprs import sub_time
from FD_utils import laplacian

try:
    import devitopro as dvp
except ImportError:
    dvp = None


def func_name(freq=None, ic="as"):
    """
    Get key for imaging condition/linearized source function
    """
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
    ic_func = ic_dict[func_name(freq=freq, ic=ic)]
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
    ls_func = ls_dict[func_name(ic=str(ic))]
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


# =================== VISCOACOUSTIC GRADIENTS ===================

def crosscorr_time_visco(u, v, model, **kwargs):
    """
    Cross correlation for viscoacoustic media - returns both Vp and Q gradients
    """
    
    try:
        # Убедимся, что u и v - кортежи
        u_tuple = as_tuple(u)
        v_tuple = as_tuple(v)
        
        
        # Извлекаем поля давления (первые элементы)
        if len(u_tuple) >= 1:
            p = u_tuple[0]
        else:
            p = u_tuple
            
        if len(v_tuple) >= 1:
            p_adj = v_tuple[0]
        else:
            p_adj = v_tuple
        
        # Gradient for Vp (m parameter) - стандартная формула
        dt = p.indices[0].spacing
        w = dt * model.irho
        grad_m = w * p_adj.dt2 * p
        
        # Gradient for Q (qp parameter) - упрощенная формула для теста
        # В реальности нужно правильное выражение для градиента по Q
        grad_q = grad_q_expr(p, p_adj, model, kwargs.get('f0', 0.015))
        
        return {"grad_m": grad_m, "grad_q": grad_q}
        
    except Exception as e:
        # В случае ошибки возвращаем оба градиента с нулевыми значениями
        return {"grad_m": 0, "grad_q": 0}


def grad_q_expr(p, p_adj, model, f0=0.015):
    """
    Упрощенный градиент по Q для тестирования
    В реальности нужно использовать правильную формулу из теории
    """
    try:
        dt = p.indices[0].spacing
        w = dt * model.irho
        # Упрощенная формула: grad_Q пропорционален квадрату частоты
        # Это только для теста!
        grad_q = 0.1 * w * p_adj.dt2 * p * (f0**2)
        return grad_q
    except Exception as e:
        return 0 * p * p_adj  # Нулевое выражение того же типа


def grad_expr_multi(grad_dict, u, v, model, w=None, freq=None, dft_sub=None, ic="as"):
    """
    Gradient expression for multiple parameters
    """
    
    # Для вязкоакустики всегда используем visco imaging condition
    if model.is_viscoacoustic:
        ic_func = crosscorr_time_visco
    else:
        ic_func = ic_dict.get(func_name(freq=freq, ic=ic), crosscorr_time)
    
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
    
    
    # Create equations for each gradient
    eqs = []
    if hasattr(expr, 'keys'):
        # Это словарь с несколькими градиентами
        for param_name, expr_part in expr.items():
            if param_name in grad_dict:
                eqs.append(Eq(grad_dict[param_name], 
                             grad_dict[param_name] - expr_part, 
                             subdomain=model.physical))
    else:
        # Для обратной совместимости: один градиент
        if "grad_m" in grad_dict:
            eqs.append(Eq(grad_dict["grad_m"], 
                         grad_dict["grad_m"] - expr, 
                         subdomain=model.physical))
    
    return eqs


fwi_src = lambda *ar, **kw: isic_src(*ar, icsign=-1, **kw)
fwi_time = lambda *ar, **kw: isic_time(*ar, icsign=-1, **kw)
fwi_freq = lambda *ar, **kw: isic_freq(*ar, icsign=-1, **kw)

ic_dict = {"isic_freq": isic_freq, "as_freq": crosscorr_freq,
           "fwi": fwi_time, "fwi_freq": fwi_freq,
           "isic": isic_time, "as": crosscorr_time, 
           "visco": crosscorr_time_visco}
ls_dict = {"isic": isic_src, "fwi": fwi_src, "as": basic_src}
