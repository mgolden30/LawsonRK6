import numpy as np
from scipy.linalg import expm

def rk6(x, t, steps, v):
    """
    Eight stage, sixth order Runge-Kutta integrator. 
    Integrates the system of ODEs dx/dt = v(x).

    Parameters
    ----------
    x : array_like
        Initial condition x(0).
    t : float
        Integration time.
    steps : int
        Number of integration steps.
    v : callable
        Function defining the ODE.

    Returns
    -------
    x : ndarray
        Final state x(t).
    """
    h = t/steps
    for _ in range(steps):
        k1 = h*v(x)
        k2 = h*v(x + k1/6)
        k3 = h*v(x + k1/12 + k2/12)
        k4 = h*v(x - k2*4/33 + k3*5/11)
        k5 = h*v(x - k1/4 - k2*29/44 + k3*31/22)
        k6 = h*v(x + k1*3/11 + k2*8/33 - k3*4/11 + k4/11 + k5*14/33)
        k7 = h*v(x - k1*17/48 - k2*5/12 + k3 + k4 - k5*13/12 + k6*11/16) 
        k8 = h*v(x + k1*20/39 + k2*12/39 - k3*31/39 - k4/39 + k5*34/39 - k6*11/39 + k7*16/39)
        x = x + 13/200*(k1 + k8) + 4/25*(k3 + k7) + 11/40*(k4 + k6)
    return x

def lawson_rk6(x, t, steps, g, A, diagonal=False):
    """
    Eight stage, sixth order Runge-Kutta integrator applied to Lawson integration. 
    Integrates the system of ODEs dx/dt = g(x) + A(x),
    where g is nonlinear and A is a linear operator.

    Parameters
    ----------
    x : array_like
        Initial condition x(0)
    t : float
        Integration time.
    steps : int
        Number of integration steps.
    g : callable
        Function defining the ODE.
    A : array_like
        Linear operator defining the ODE. Can be a matrix or an array with 
        the diagonal elements if A is diagonal.
    diagonal : boolean
        Flag for the shape of A.
            False -> A is a n-by-n matrix
            True -> A is an array with n elements

    Returns
    -------
    x : ndarray
        Final state x(t)
    """
    h = t/steps
    e = np.exp( h/6 * A ) if diagonal else expm( h/6 * A)

    mul = lambda x: e*x if diagonal else e@x
    for _ in range(steps):
        k1 = h*g(x)
        x = mul(x); k1 = mul(k1);
        k2 = h*g(x + k1/6)
        k3 = h*g(x + k1/12 + k2/12)
        x = mul(x); k1 = mul(k1); k2 = mul(k2); k3 = mul(k3);
        k4 = h*g(x - k2*4/33 + k3*5/11)
        x = mul(x); k1 = mul(k1); k2 = mul(k2); k3 = mul(k3); k4 = mul(k4);
        k5 = h*g(x - k1/4 - k2*29/44 + k3*31/22)
        x = mul(x); k1 = mul(k1); k2 = mul(k2); k3 = mul(k3); k4 = mul(k4); k5 = mul(k5);
        k6 = h*g(x + k1*3/11 + k2*8/33 - k3*4/11 + k4/11 + k5*14/33)
        x = mul(x); k1 = mul(k1); k2 = mul(k2); k3 = mul(k3); k4 = mul(k4); k5 = mul(k5); k6 = mul(k6);
        k7 = h*g(x - k1*17/48 - k2*5/12 + k3 + k4 - k5*13/12 + k6*11/16) 
        x = mul(x); k1 = mul(k1); k2 = mul(k2); k3 = mul(k3); k4 = mul(k4); k5 = mul(k5); k6 = mul(k6); k7= mul(k7);
        k8 = h*g(x + k1*20/39 + k2*12/39 - k3*31/39 - k4/39 + k5*34/39 - k6*11/39 + k7*16/39)
        x = x + 13/200*(k1 + k8) + 4/25*(k3 + k7) + 11/40*(k4 + k6)
    return x