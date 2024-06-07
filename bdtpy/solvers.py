import numpy as np
import importlib

def sde_solver(drift, diffusion, y0, tspan, solver=None):
    """
    Efficient solver for SDEs. Wraps functions within
    ``sdeint``. Compiled as `numba.jit` for easy speedup.

    Parameters
    ==========
    drift : function 
        function handle of deterministic part of the SDE, usually of the form

            f = -(a + y*b**2)*(1 - y**2)

    diffusion : function
        function handle of stochastic part of the SDE, usually of the form

            g = b*(1 - y**2)

    y0 : float
        Initial condition for y
    
    tspan : np.ndarray
        timepoints to solve SDE for
    
    solver : str, optional
        solver to use, 'ito' or 'strat' (Stratonovich). Default 'ito'

    """
    if solver is None:
        solver = 'itoint'

    sdeint = importlib.import_module('sdeint.integrate', package='sdeint')
    solver = getattr(sdeint, solver)
    
    if (isinstance(y0, np.ndarray) or isinstance(y0, list)) and len(y0) > 2:
        result = [solver(drift, diffusion, y0[i], tspan) for i in range(len(y0))]
        return np.array(result).squeeze()

    else:
        return solver(drift, diffusion, y0, tspan)