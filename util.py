import numpy as np
from autograd import primitive
import wh
from scipy.sparse import csr_matrix
from scipy.stats import gamma
from scipy import integrate


__nugget_scalar = 1e-7
__hyper_shrinkage = 1e-9
__sparse = True
__sparse_fmt = csr_matrix


def linearise_params(params):
    """ using a vector to save all parameters"""
    (M, N) = params.L.shape
    (P, Q) = params.m.shape
    assert M == N and M == P and Q == 1, (M, N, P, Q)
    return np.concatenate((tril2lin(params.L), params.m.flatten(),
                           [params.shape,params.scale]))

def unlinearise_params(x,verbose=False):
    """ using an object to save all parameters"""
    x = x.flatten()
    M = len(x)
    N = nparams2dim(M)
    NL = int(N * (N+1) / 2)
    L = lin2tril(x[:NL])
    m = x[NL:(NL+N)].reshape(-1,1)
    shape, scale = x[(NL + N):]
    return wh.RaisingDotDict(L=L, m=m, shape=shape,scale=scale)


def nparams2dim(M):
    # Solve[n (n + 1)/2 + 2 + n == x, n]
    # {{n -> 1/2 (-3 - Sqrt[-7 + 8 x])}, {n -> 1/2 (-3 + Sqrt[-7 + 8 x])}}
    N = int((-3 + np.sqrt(-7 + 8*M))/2)
    M2, _, _, _ = dim2nparams(N)
    assert M == M2, (M, M2, N)
    return N


def dim2nparams(N):
    total = int(N * (N+1) / 2 + N + 2)
    len_dict = dict()
    range_dict = dict()
    i0 = 0
    names = []
    name_len = [('L',N*(N+1)/2), ('m',N), ('shape',1),('scale',1)]
    for k, n in name_len:
        n = int(n)
        len_dict[k] = n
        range_dict[k] = range(i0,i0+n)
        names += [k] * n
        i0 = i0 + n
    assert sum(len_dict.values()) == total, (len_dict,total)
    return total, len_dict, range_dict, names


@primitive
def matmul_ad(a, b, c):
    """ for packing and unpacking parameters"""
    return b @ a


def make_grad_matmul_ad(ans, a, b, c):
    def gradient_product(g):
        return c @ g
    return gradient_product


matmul_ad.defgrad(make_grad_matmul_ad, 0)


if __sparse:
    matmul = matmul_ad
else:
    matmul = lambda a, b, c: b @ a


@wh.memodict
def lin2triltfm(m):
    """ tfm and tfm_t is a matrix recording the positions of elements below the main diagonal"""
    n = int(m * (m+1) / 2)
    tfm = np.zeros((m*m, n))
    i = np.where(np.tril(np.ones((m,m))).flatten())[0]
    tfm[i, range(len(i))] = 1
    tfm_t = tfm.T
    if __sparse:
        tfm = __sparse_fmt(tfm)
        tfm_t = __sparse_fmt(tfm_t)
    return tfm, tfm_t


@wh.memodict
def lin2triltfm(m):
    """ tfm and tfm_t is a matrix recording the positions of elements below the main diagonal"""
    n = int(m * (m+1) / 2)
    tfm = np.zeros((m*m, n))
    i = np.where(np.tril(np.ones((m,m))).flatten())[0]
    tfm[i, range(len(i))] = 1
    tfm_t = tfm.T
    if __sparse:
        tfm = __sparse_fmt(tfm)
        tfm_t = __sparse_fmt(tfm_t)
    return tfm, tfm_t


def lin2tril(x):
    """ a horizontal vector to a triangular matrix"""
    """ for unpacking parameters"""
    n = len(x.flatten())
    m = int(1/2 * (-1 + np.sqrt(1 + 8 * n)))
    assert m * (m+1) / 2 == n, (m, n, x.shape)
    tfm, tfm_t = lin2triltfm(m)
    return (matmul(x, tfm, tfm_t)).reshape(m,m)


def tril2lin(L):
    """ a triangular matrix to a horizontal vector"""
    """ packing parameters"""
    mm, m = L.shape
    assert m == mm
    tfm, tfm_t = lin2triltfm(m)
    return matmul(L.flatten(), tfm_t, tfm)


def squared_normal_shape_scale(mu, sigma):
    shape = (mu ** 2 + sigma ** 2) ** 2 / (2 * sigma ** 2 * (2 * mu ** 2 + sigma ** 2))
    scale = (2 * mu ** 2 * sigma ** 2 + sigma ** 4) / (mu ** 2 + sigma ** 2)
    return shape, scale


def squared_normal_quantiles(mu, sigma, probs, double=False):
    # quantiles of Y = 1/2 X^2 where X ~ N(mu, sigma^2)
    # if probs is none, return the mean
    assert len(mu) == len(sigma)
    factor = 2.0 if double else 1.0
    mu = mu.flatten()
    sigma = sigma.flatten()
    shape, scale = squared_normal_shape_scale(mu, sigma)
    if probs is None:
        return shape * scale * factor
    else:
        probs2 = 1-np.array(probs).flatten()
        rval = np.array([gamma(a=thisshape, scale=thisscale).isf(probs2) for thisshape, thisscale in zip(shape, scale)])
    return rval * factor



def l2_distance(f1,f2,a,b,n):
    interp_x = np.linspace(a,b,n)
    measure = interp_x[1]-interp_x[0]
    y = (f1(interp_x)-f2(interp_x))**2
    return np.sqrt(np.sum((y[:-1]+y[1:]))*measure*0.5)


def seq2diff(x):
    assert 1 == x.ndim
    return [ x[i]-x[:i] for i in range(1,len(x))]
