import autograd.numpy as np
import scipy.interpolate
from autograd import primitive
from scipy.sparse import csr_matrix
import wh

__euler_mascheroni = 0.57721566490153286060651209008240243104215933593992
__sparse_fmt = csr_matrix
__interp1d_kind = 'linear'
__gtilde_subsample = 1

__gtilde_pickle_fn = 'gtilde.pkl'
__gtilde_csv_fn = 'gtilde.csv'

_gtilde_table = wh.load(__gtilde_pickle_fn)

isub = list(range(0, _gtilde_table.shape[1]-1, __gtilde_subsample)) + [_gtilde_table.shape[1]-1]
_gtilde_table = _gtilde_table[:,isub]
_gtilde_neglogz, _gtilde_value, _grad_gtilde_value =_gtilde_table
assert not np.isinf(min(_gtilde_neglogz))
_gtilde_neglogz_0, _gtilde_value_0, _grad_gtilde_value_0 = -np.inf, 0.0, 2
_gtilde_neglogz_range = (min(_gtilde_neglogz),max(_gtilde_neglogz))
imin = np.argmin(_gtilde_neglogz)
assert imin == 0
assert np.allclose(_gtilde_value_0, _gtilde_value[imin])
assert np.allclose(_grad_gtilde_value_0, _grad_gtilde_value[imin])
_gtilde_interp = scipy.interpolate.interp1d(_gtilde_neglogz, _gtilde_value, fill_value=(_gtilde_value_0, np.nan), bounds_error=False, kind=__interp1d_kind)
_grad_gtilde_interp = scipy.interpolate.interp1d(_gtilde_neglogz, _grad_gtilde_value, fill_value=(_grad_gtilde_value_0, np.nan), bounds_error=False, kind=__interp1d_kind)



def gtilde(z):
    """get the value of gtilde at -z by intersection"""
    assert isinstance(z, np.ndarray)
    assert np.all(z <= 0.0)
    lognegz = np.log(-z)
    assert np.all(lognegz <= _gtilde_neglogz_range[1]), (min(lognegz), max(lognegz), _gtilde_neglogz_range)
    rval = _gtilde_interp(lognegz)
    rval[z==0] = _gtilde_value_0
    rval[lognegz < _gtilde_neglogz[0]] = 0.0
    assert np.all(~np.isnan(rval).flatten())
    return rval


def grad_gtilde(z):
    """get the value of grad of gtilde at -z by intersection"""
    assert np.all(z <= 0.0)
    lognegz = np.log(-z)
    assert np.all(lognegz <= _gtilde_neglogz_range[1]), (min(lognegz), max(lognegz), _gtilde_neglogz_range)
    rval = _grad_gtilde_interp(lognegz)
    rval[z==0] = _grad_gtilde_value_0
    assert not np.any(np.isnan(rval).flatten()), (np.min(z), np.max(z), np.min(lognegz), np.max(lognegz))
    return rval


@primitive
def gtilde_ad(z):
    return gtilde(z)


def make_grad_gtilde_ad(ans, z):
    def gradient_product(g):
        return g * grad_gtilde(z)
    return gradient_product


gtilde_ad.defgrad(make_grad_gtilde_ad)