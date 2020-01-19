import autograd.scipy.special as special
import autograd.numpy as np
import wh, wp
import util
import gtilde
from scipy.integrate import quad
from scipy.stats import multivariate_normal
from autograd import primitive
import matplotlib.pyplot as plt
__nugget_scalar = 1e-7
__nugget = lambda n: __nugget_scalar * np.eye(n)
__verify = False
__euler_mascheroni = 0.57721566490153286060651209008240243104215933593992
from scipy.misc import logsumexp


def psi(x, gamma, alpha, trange):
    tmin, tmax = trange
    y = x
    d = x.shape[0]
    exp1 = ((np.pi * alpha / 4) ** (d/2)) * (gamma**2) * np.exp(-wh.sqdist(x,None) / (4*alpha))
    xbar = 0.5 * (x.reshape(x.shape[0],x.shape[1],1)+y.reshape(y.shape[0],1,y.shape[1]))
    d = special.erf((xbar-tmin) / np.sqrt(alpha)) - special.erf((xbar - tmax) / np.sqrt(alpha))

    prodd = np.prod(d, axis=0)
    rval = exp1 * prodd
    rval = 0.5 * (rval + rval.T)
    rval += 2 * gamma * __nugget_scalar
    rval += np.eye(rval.shape[0]) * __nugget_scalar ** 2
    return rval


def k(x, y, gamma, alpha):
    """kernel function"""
    d2 = wh.sqdist(x,y)
    rval = gamma * np.exp(-d2/ (2*alpha))
    if y is None:
        delta = np.eye(x.shape[1])
        rval = 0.5 * (rval + rval.T)
    else:
        delta = (d2 <= 1e-7)
    rval += (delta * __nugget_scalar)
    return rval


def kdiag(x, gamma):
    """diagonal elements of the kernel matrix"""
    return (gamma + __nugget_scalar) * np.ones(x.shape[1])


def kl_tril(L, m, Lzz,u):
    """KL divergence of q(u) and p(u)"""
    M = L.shape[0]
    traceterm = np.sum(np.linalg.solve(Lzz, L)**2)
    mkmterm = np.sum(np.linalg.solve(Lzz,u-m)**2)
    logdetk = 2 * np.sum(np.log(np.abs(np.diag(Lzz))))
    logdets = 2 * np.sum(np.log(np.abs(np.diag(L))))
    kl = 0.5 * (traceterm + logdetk - logdets - M + mkmterm)
    if __verify:
        S = L @ L.T
        Kzz = Lzz @ Lzz.T
        traceterm2 = np.trace(np.linalg.solve(Kzz, S))
        mkmterm2 = np.dot((u-m).T,np.linalg.solve(Kzz, u-m))[0,0]
        logdetk2 = np.log(np.linalg.det(Kzz))
        logdets2 = np.log(np.linalg.det(S))
        wh.assert_close(traceterm, traceterm2)
        wh.assert_close(mkmterm, mkmterm2)
        if not np.isinf(logdetk2) and not np.isnan(logdetk2):
            wh.assert_close(logdetk, logdetk2, rtol=5e-2)
        if not np.isinf(logdets2) and not np.isnan(logdets2):
            wh.assert_close(logdets, logdets2, rtol=5e-2)
    return kl


def expected_log_f2(mu, sigma):
    """expectation of log of f^2"""
    assert mu.shape == sigma.shape, (mu.shape, sigma.shape)
    return -gtilde.gtilde_ad( - (mu ** 2) / (2 * (sigma ** 2))), \
           np.log((sigma ** 2) / 2), \
           - __euler_mascheroni * np.ones(mu.shape, dtype=float)


def predictive(L, m, precomp, precomp_predictive):
    """mean and variance of q(f)"""
    kzzinv = precomp.Lzzinv.T @ precomp.Lzzinv
    kzzinv_m = kzzinv @ m
    mu = (precomp_predictive.Kxz @ kzzinv_m).flatten()
    sigmaa = precomp_predictive.Kxx_diag
    Lzz_inv_Kzx = np.linalg.solve(precomp.Lzz, precomp_predictive.Kxz.T)
    sigmab = np.sum(precomp_predictive.Kxz * np.linalg.solve(precomp.Lzz.T, Lzz_inv_Kzx).T, axis=1)
    Kzz_inv_Kzx = np.linalg.solve(precomp.Lzz.T, Lzz_inv_Kzx)
    sigmac = np.sum((L.T @ Kzz_inv_Kzx) ** 2, axis=0)
    sigma2 = sigmaa - sigmab + sigmac

    Sigmaa = precomp_predictive.Kxx
    Sigmab = precomp_predictive.Kxz @ kzzinv @ precomp_predictive.Kxz.T
    Sigmac = precomp_predictive.Kxz @ kzzinv @ L @ L.T @ kzzinv @ precomp_predictive.Kxz.T
    Sigma = Sigmaa - Sigmab + Sigmac
    wh.assert_close(np.diag(Sigma), sigma2, rtol=1e-3)

    if __verify:
        sigmab2 = np.diag(precomp_predictive.Kxz @ np.linalg.solve(precomp.Kzz, precomp_predictive.Kxz.T))
        sigmac2 = np.diag(precomp_predictive.Kxz @ np.linalg.solve(precomp.Kzz, L @ L.T @ np.linalg.solve(precomp.Kzz,
                                                                                                          precomp_predictive.Kxz.T)))
        wh.assert_close(sigmab, sigmab2, rtol=1e-3)
        wh.assert_close(sigmac, sigmac2, rtol=1e-3)
    return mu, sigma2, Sigma


def precompute(z, gamma, alpha, run_time, x, taus):
    """ calculating Kzz, Lzz and Psi"""
    precomp = wh.RaisingDotDict()
    taus = np.concatenate([tau.flatten() for tau in taus]).reshape((1, -1))
    precomp.Kzz = k(z, None, gamma, alpha)
    try:
        precomp.Lzz = np.linalg.cholesky(precomp.Kzz)
        precomp.Lzzinv = np.linalg.inv(precomp.Lzz)
        precomp.Kzzinv = precomp.Lzzinv.T @ precomp.Lzzinv
    except Exception as e:
        print('gamma:', gamma, 'alpha:', alpha, 'Kzz:', precomp.Kzz)
        raise

    tmin = 0
    exp = ((np.pi * alpha / 4) ** (1 / 2)) * (gamma ** 2) * np.exp(-wh.sqdist(z, None) / (4 * alpha))
    zy = z
    zbar = 0.5 * (z.reshape(z.shape[0], z.shape[1], 1) + zy.reshape(zy.shape[0], 1, zy.shape[1]))
    dmin_array = special.erf((zbar - tmin) / np.sqrt(alpha))
    dprod_sum = np.sum(
        np.array([np.prod(dmin_array - special.erf((zbar - (run_time - x[i])) / np.sqrt(alpha)), axis=0)
                  for i in range(len(x))]), axis=0)
    r = exp * dprod_sum
    r = 0.5 * (r + r.T)
    precomp.psi_sum = r + 2 * gamma * __nugget_scalar + np.eye(r.shape[0]) * __nugget_scalar ** 2
    precomp.Kzzinv_psi_sum = precomp.Kzzinv@precomp.psi_sum
    precomp.Kzzinv_psi_sum_Kzzinv = precomp.Kzzinv @ precomp.psi_sum @ precomp.Kzzinv
    precomp.Kxz = k(taus, z, gamma, alpha)
    precomp.Kzzinv_kzx = precomp.Kzzinv @ precomp.Kxz.T
    precomp.sigmas = kdiag(taus, gamma)
    return precomp


def precompute_predictive(x, z, gamma, alpha):
    """???"""
    precomp = wh.RaisingDotDict()
    precomp.Kxz = k(x, z, gamma, alpha)
    precomp.Kxx = k(x, None, gamma, alpha)
    precomp.Kxx_diag = kdiag(x, gamma)
    return precomp


def update_pij(paramslin, taus, z, gamma, alpha, pij, kzzinv):
    nx = len(taus)+1
    params = util.unlinearise_params(paramslin, verbose=0)
    kzzinv_m = kzzinv @ params.m
    expEmu = params.scale*np.exp(special.digamma(params.shape))

    for i in range(nx-1):
        tau = taus[i]
        Kxz = k(tau, z, gamma, alpha)
        mutilde = (Kxz @ kzzinv_m).flatten()
        sigmaa = kdiag(tau, gamma)
        kzzinv_kzx = kzzinv @ Kxz.T
        sigmab = np.sum(Kxz * kzzinv_kzx.T, axis=1)
        sigmac = np.sum((params.L.T @ kzzinv_kzx) ** 2, axis=0)
        sigmatilde = sigmaa - sigmab + sigmac

        eqn19a, eqn19b, eqn19c = expected_log_f2(mutilde, np.sqrt(sigmatilde))
        eqn19 = eqn19a + eqn19b + eqn19c
        expeqn19 = np.exp(eqn19)
        denom = expEmu + np.sum(expeqn19)
        pij[i+1][0] = expEmu / denom
        pij[i+1][1:tau.shape[1]+1] = expeqn19 / denom

    return pij


def eqn15sum_numerical(paramslin, x,z,run_time, gamma, alpha):
    params = util.unlinearise_params(paramslin, verbose=0,)
    precomp = precompute(z, gamma, alpha, [0, run_time])
    kzzinv = precomp.Lzzinv.T @ precomp.Lzzinv
    kzzinv_m = kzzinv @ params.m
    interp_x = np.linspace(0,run_time,4096).reshape((1,-1))
    delta = interp_x[0,1]-interp_x[0,0]
    kxz = k(interp_x,z, gamma, alpha)
    mutilde2 = (kxz@kzzinv_m).flatten()**2
    eqn15sum = 0
    for i in range(x.shape[1]):
        N = np.sum(interp_x[0] < (run_time - x[0, i]))
        eqn15sum += np.sum([np.sum(mutilde2[:N-1]+mutilde2[1:N])])*delta/2
    return eqn15sum


def eqn16sum_numerical(paramslin, x,z,run_time, gamma, alpha):
    params = util.unlinearise_params(paramslin, verbose=0)
    precomp = precompute(z, gamma, alpha, [0, run_time])
    kzzinv = precomp.Lzzinv.T @ precomp.Lzzinv
    interp_x = np.linspace(0, run_time, 5000).reshape((1, -1))
    kxx = kdiag(interp_x, gamma)
    delta = interp_x[0,1]-interp_x[0,0]
    kxz = k(interp_x, z, gamma, alpha)
    sigmatilde2 = kxx-np.sum(kxz*(kxz@kzzinv.T),axis=1) + np.sum((kxz@kzzinv@params.L)**2,axis=1)

    eqn16sum = 0
    for i in range(x.shape[1]):
        N = np.sum(interp_x[0] < (run_time - x[0, i]))
        eqn16sum += np.sum([np.sum(sigmatilde2[:N-1]+sigmatilde2[1:N])])*delta/2
    return eqn16sum


def eqn19sum_numerical(paramslin, x,z, pij, run_time, gamma, alpha):
    params = util.unlinearise_params(paramslin, verbose=0)
    precomp = precompute(z, gamma, alpha, [0, run_time])
    kzzinv = precomp.Lzzinv.T @ precomp.Lzzinv
    kzzinv_m = kzzinv @ params.m
    eqn19sum = 0
    kzzinv_S_kzzinv = kzzinv @ params.L @ params.L.T @ kzzinv
    mutilde_list = []
    for i in range(1, x.shape[1]):
        taus = np.array([x[0, i] - x[0, :i]])
        kxx = kdiag(taus, gamma)
        kxz = k(taus, z, gamma, alpha)
        mutilde = (kxz @ kzzinv_m).flatten()
        mutilde_list += mutilde.tolist()
        sigmatilde = np.sqrt(np.diag(kxx - kxz @ kzzinv @ kxz.T + kxz @ kzzinv_S_kzzinv @ kxz.T))
        assert mutilde.ndim == 1
        assert sigmatilde.ndim == 1
        for j in range(len(mutilde)):
            mui = mutilde[j]
            sigmai = sigmatilde[j]
            interp_f = np.linspace(mui - 6*sigmai, mui + 6 * sigmai, 4096)
            delta = interp_f[1] - interp_f[0]
            e_log_f2 = multivariate_normal(mui, sigmai**2).pdf(interp_f) * np.log(interp_f ** 2) * delta
            eqn19sum += pij[i, j + 1] * np.sum(e_log_f2)
    return eqn19sum


def objective(paramslin, x, z, pij_flatten, pij0sum, run_time, taus, gamma, alpha, g0_params,precomp):
    params = util.unlinearise_params(paramslin, verbose=0)
    d, nz = z.shape
    nx = x.shape[1]
    kzzinv_m = precomp.Kzzinv @ params.m
    s = params.L @ params.L.T+__nugget(params.L.shape[1])
    eqn15sum = (params.m.T @ precomp.Kzzinv_psi_sum_Kzzinv @params.m)[0,0]

    eqn16a = np.trace(precomp.Kzzinv_psi_sum)
    eqn16b = np.trace(precomp.Kzzinv_psi_sum_Kzzinv @ s)
    eqn16sum = gamma*np.sum((run_time-x[0])**d)-eqn16a + eqn16b

    mutilde = (precomp.Kzzinv_kzx.T @ params.m).flatten()
    sigmaa = precomp.sigmas
    sigmab = np.sum(precomp.Kxz * precomp.Kzzinv_kzx.T, axis=1)
    sigmac = np.sum((params.L.T @ precomp.Kzzinv_kzx) ** 2, axis=0)
    sigmatilde = sigmaa - sigmab + sigmac
    eqn19a, eqn19b, eqn19c = expected_log_f2(mutilde, np.sqrt(sigmatilde))
    eqn19sum = -(eqn19c + eqn19a + eqn19b)@pij_flatten

    kl_normal = kl_tril(params.L, params.m, precomp.Lzz, 0)
    kl_g = kl_gamma(params.scale,params.shape, g0_params['scale'],g0_params['shape'])

    total = kl_normal+kl_g+eqn15sum + eqn16sum + eqn19sum +run_time*params.shape*params.scale-\
            pij0sum*(special.digamma(params.shape)+np.log(params.scale))
    return total

def log_marginal_likelihood(paramslin, x, z, pij, pij_flatten, pij0sum, run_time,taus, gamma, alpha, precomp):
    params = util.unlinearise_params(paramslin, verbose=0)
    d, nz = z.shape
    nx = x.shape[1]
    s = params.L @ params.L.T+__nugget(params.L.shape[1])
    eqn15sum = (params.m.T @ precomp.Kzzinv_psi_sum_Kzzinv @params.m)[0,0]

    eqn16a = np.trace(precomp.Kzzinv_psi_sum)
    eqn16b = np.trace(precomp.Kzzinv_psi_sum_Kzzinv @ s)
    eqn16sum = gamma*np.sum((run_time-x[0])**d)-eqn16a + eqn16b

    mutilde = (precomp.Kzzinv_kzx.T @ params.m).flatten()
    sigmaa = precomp.sigmas
    sigmab = np.sum(precomp.Kxz * precomp.Kzzinv_kzx.T, axis=1)
    sigmac = np.sum((params.L.T @ precomp.Kzzinv_kzx) ** 2, axis=0)
    sigmatilde = sigmaa - sigmab + sigmac
    eqn19a, eqn19b, eqn19c = expected_log_f2(mutilde, np.sqrt(sigmatilde))
    eqn19sum = -(eqn19c + eqn19a + eqn19b)@pij_flatten

    ppij = pij[pij > 0]

    total = eqn15sum + eqn16sum + eqn19sum + run_time * params.shape * params.scale - \
            pij0sum*(special.digamma(params.shape) + np.log(params.scale)) + ppij @ np.log(ppij)

    return -total


def kl_gamma(a,b,c,d):
    # `b` and `d` are Gamma shape parameters and
    # `a` and `c` are scale parameters.
    # (All, therefore, must be positive.)
    # copy from: https://stats.stackexchange.com/questions/11646/kullback-leibler-divergence-between-two-gamma-distributions
    def I(a,b,c,d):
        return -c*d/a -b*np.log(a) - special.gammaln(b) + (b-1)*(special.digamma(d) + np.log(c))

    return I(c,d,c,d) - I(a,b,c,d)
