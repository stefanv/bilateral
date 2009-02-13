'''
  A cython implementation of the plain (and slow) algorithm for bilateral
  filtering.

  Copyright 2008, Nadav Horesh
  nadavh at visionsense.com 
'''

from numpy.numarray.nd_image import generic_filter
import numpy as np


cdef extern from "arrayobject.h":
    ctypedef int intp
    ctypedef extern class numpy.ndarray [object PyArrayObject]:
        cdef char *data
        cdef int nd
        cdef intp *dimensions
        cdef intp *strides
        cdef int flags

cdef extern from "math.h":
    double exp(double x)

## gcc specific (?)
cdef extern:
    int __builtin_abs(int x)

cdef int GAUSS_SAMP = 32
cdef int GAUSS_IDX_MAX = GAUSS_SAMP - 1

class _Bilat_fcn(object):
    '''
    The class provides the bilaterl filter function to be called by
    generic_filter.
    initialization parameters:
      spat_sig:    The sigma of the spatial Gaussian filter
      inten_sig:   The sigma of the gray-levels Gaussian filter
      filter_size: (int) The size of the spatial convolution kernel. If
                   not set, it is set to ~ 4*spat_sig.
    '''
    def __init__(self, spat_sig, inten_sig, filter_size=None):
        if filter_size is not None and filter_size >= 2:
            self.xy_size = int(filter_size)
        else:
            self.xy_size = int(round(spat_sig*4))
            # Make filter size odd
            self.xy_size += 1-self.xy_size%2
        x = np.arange(self.xy_size, dtype=float)
        x = (x-x.mean())**2
        #xy_ker: Spatial convolution kernel
        self.xy_ker = np.exp(-np.add.outer(x,x)/(2*spat_sig**2)).ravel()
        self.xy_ker /= self.xy_ker.sum()
        self.inten_sig = 2 * inten_sig**2
        self.index = self.xy_size**2 // 2

        ## An initialization for LUT instead of a Gaussian function call
        ## (for the fc_filter method)

        x = np.linspace(0,3.0, GAUSS_SAMP)
        self.gauss_lut = np.exp(-x**2/2)
        self.gauss_lut[-1] = 0.0  # Nullify all distant deltas        
        self.x_quant = 3*inten_sig / GAUSS_IDX_MAX


    ##
    ## Filtering functions
    ##

    def __call__ (self, data):
        'An unoptimized (pure python) implementation'
        weight = np.exp(-(data-data[self.index])**2/self.inten_sig) * self.xy_ker
        return np.dot(data, weight) / weight.sum()

    def cfilter(self, ndarray data):
        'An optimized implementation'
        cdef ndarray kernel = self.xy_ker
        cdef double sigma   = self.inten_sig
        cdef double weight_i, weight, result, centre, dat_i
        cdef double *pdata=<double *>data.data, *pker=<double *>kernel.data
        cdef int i, dim = data.dimensions[0]
        centre = pdata[self.index]

        weight = 0.0
        result = 0.0

        for i from  0 <= i < dim:
            dat_i = pdata[i]
            weight_i = exp(-(dat_i-centre)**2 / sigma) * pker[i]
            weight += weight_i;
            result += dat_i * weight_i
        return result / weight


    def fc_filter(self, ndarray data):
        'Use further optimisation by replacing exp functions calls by a LUT'
        cdef ndarray kernel = self.xy_ker
        cdef ndarray gauss_lut_arr = self.gauss_lut

        cdef double sigma   = self.inten_sig
        cdef double weight_i, weight, result, centre, dat_i
        cdef double *pdata=<double *>data.data, *pker=<double *>kernel.data
        cdef int i, dim = data.dimensions[0]
        cdef int exp_i    # Entry index for the LUT
        cdef double x_quant = self.x_quant
        cdef double *gauss_lut = <double *>gauss_lut_arr.data

        centre = pdata[self.index]

        weight = 0.0
        result = 0.0

        for i from  0 <= i < dim:
            dat_i = pdata[i]
            exp_i = __builtin_abs(<int>((dat_i-centre) / x_quant))
            if exp_i > GAUSS_IDX_MAX:
                exp_i = GAUSS_IDX_MAX
            weight_i = gauss_lut[exp_i] * pker[i]
            weight += weight_i;
            result += dat_i * weight_i
        return result / weight


#
# Filtering functions to be called from outsize
#


def bilateral(mat, xy_sig, z_sig, filter_size=None, mode='reflect'):
    '''
    Bilateral filter a 2D array.
      mat:         A 2D array
      xy_sig:      The sigma of the spatial Gaussian filter.
      z_sig:       The sigma of the gray levels Gaussian filter.
      filter_size: Sise of the spatial filter kernel: None of values < 2 --- auto select.
      mode:        See numpy.numarray.nd_image.generic_filter documentation

    output: A 2D array of the same dimensions as mat and a float64 dtype

    Remarks:
      1. The spatial filter kernel size is ~4*xy_sig, unless specified otherwise
      2. The algorithm is slow but has a minimal memory footprint
    '''
    
    filter_fcn = _Bilat_fcn(xy_sig, z_sig, filter_size)
    size = filter_fcn.xy_size
    return generic_filter(mat, filter_fcn.cfilter, size =(size,size), mode=mode)

def bilateral_slow(mat, xy_sig, z_sig, filter_size=None, mode='reflect'):
    'A pure python implementation of the bilateral filter, for details see bilateral doc.'
    filter_fcn = _Bilat_fcn(xy_sig, z_sig, filter_size)
    size = filter_fcn.xy_size
    return generic_filter(mat, filter_fcn, size =(size,size), mode=mode)

def bilateral_fast(mat, xy_sig, z_sig, filter_size=None, mode='reflect'):
    'A fast implementation of bilateral filter, for details see bilateral doc.'
    filter_fcn = _Bilat_fcn(xy_sig, z_sig, filter_size)
    size = filter_fcn.xy_size
    return generic_filter(mat, filter_fcn.fc_filter, size =(size,size), mode=mode)
