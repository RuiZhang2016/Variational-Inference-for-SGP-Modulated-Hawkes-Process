import __init__
__nugget_scalar = 1e-7
__nugget = lambda n: __nugget_scalar * np.eye(n)
import autograd.scipy.special as special
from util import *
from my_hawkes import *
import equations
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
import wp
import time
from scipy.optimize import minimize
from tick.hawkes import HawkesKernelTimeFunc, SimuHawkes
from tick.base import TimeFunction
from joblib import Parallel, delayed
from scipy import interpolate

np.errstate(divide='ignore')
np.random.seed(100)

figsize=(8,6)
markersize = 20
labelsize = 28
fontsize = 28

def fit_vbhp(hps, nz, zmax, run_time, filename, start_from_ideal, gamma, alpha, support):

    nhp = len(hps)
    _minimize_method = 'L-BFGS-B'
    _minimize_options = dict(maxiter=10, disp=False, ftol=0, maxcor=20)
    ite = 0
    baseline = 10
    z = np.linspace(1e-6, zmax, nz).reshape((1, -1))
    nparams, len_dict, range_dict, names = dim2nparams(nz)
    # print('x', x.shape, np.min(x, axis=1), np.max(x, axis=1))
    print('z', z.shape, np.min(z, axis=1), np.max(z, axis=1))
    print(_minimize_method, _minimize_options)

    paramslin0 = np.ones(nparams) + 1.0
    params0 = unlinearise_params(paramslin0)
    params0.shape = 10*np.pi
    params0.scale = 1/np.pi
    S0 = 0.5 * equations.k(z, z, gamma, alpha)
    params0.L = np.linalg.cholesky(S0 + np.eye(S0.shape[0]) * 1e-6)
    params0.m[:] = 1
    paramslin0 = linearise_params(params0)

    lower = params0.copy()
    upper = params0.copy()
    lower.m = 1e-7 * np.ones(lower.m.shape)
    upper.m = 10 * np.ones(lower.m.shape) # 2
    lower.L = -3 * np.max(np.abs(params0.L.flatten())) * np.ones((nz, nz))
    upper.L = 3 * np.max(np.abs(params0.L.flatten())) * np.ones((nz, nz))
    lower.shape = 10
    upper.shape = 120
    lower.scale = 0.1
    upper.scale = 1.2
    bounds = list(zip(linearise_params(lower), linearise_params(upper)))
    bounds_params0 = np.hstack((np.array(bounds), paramslin0.reshape(-1, 1)))

    assert all(a <= b for a, b in bounds), bounds
    assert all(a <= p0 and p0 <= b for (a, b), p0 in zip(bounds, list(paramslin0)))
    assert len(bounds) == len(paramslin0)


    xplot = np.linspace(0, 3, 128).reshape((1, -1))
    paramslin = paramslin0
    taus_list = [[np.array([(x[0, i] - x[0, :i])[x[0, i] - x[0, :i]<=support]]) for i in range(1, x.shape[1])] for x in hps]
    g0_params = {'shape': 10*np.pi, 'scale': 1/np.pi}
    output = []

    # precompute
    
    precomp = equations.precompute(z, gamma, alpha, run_time, hps[0][0],taus_list[0])

    pij = np.zeros((hps[0].shape[1],hps[0].shape[1]))
    pij[0,0] = 1
    precomp = equations.precompute(z, gamma, alpha, run_time, hps[0][0],taus_list[0])
    while True:
        time0 = time.time()
        params = unlinearise_params(paramslin)

        pijs = [equations.update_pij(paramslin, taus_list[i],z, gamma, alpha,  pij,precomp.Kzzinv) for i in range(nhp)]
        pij_flattens = [np.concatenate([pijs[i][j, 1:taus_list[i][j-1].shape[1]+1] for j in range(1,hps[i].shape[1])]) for i in range(nhp)]
        
        fn = lambda p: np.sum([equations.objective(p, hps[i], z, pij_flattens[i], np.sum(pijs[i][:,0]), run_time, taus_list[i], gamma, alpha, g0_params,precomp) for i in range(nhp)])/nhp                                            
        dfn = grad(fn)

        res = minimize(fn, paramslin, jac=dfn, method=_minimize_method, bounds=bounds, options=_minimize_options)
        paramslin = res['x']

        ite += 1
        del res

        if ite > 10:
            lml = 0
            for i in range(len(hps)):
                lml += equations.log_marginal_likelihood(paramslin, hps[i], z, pijs[i],pij_flattens[i], np.sum(pijs[i][:,0]), run_time, taus_list[i], gamma,alpha,precomp)
            break

    return [lml,paramslin]


if __name__=='__main__':

    train('sin')