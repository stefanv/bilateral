#   Copyright 2008, Nadav Horesh
#   nadavh at visionsense dot com
#
#   The software is licenced under BSD licence.
'''
 A front end functions for bilateral filtering.
 Provides the following functions:
 
     bilateral
     bilateral_slow
     bilateral_fast
 
  Thier output should be roughly the same, but there should be a
  considerable speed factor.
  Full documentation is found in the bilateral function doc string.
'''


from numpy.numarray.nd_image import generic_filter
import bilateral_base as _BB

def bilateral(mat, xy_sig, z_sig, filter_size=None, mode='reflect'):
    '''
    Bilateral filter a 2D array.
      mat:         A 2D array
      xy_sig:      The sigma of the spatial Gaussian filter.
      z_sig:       The sigma of the gray levels Gaussian filter.
      filter_size: Size of the spatial filter kernel: None or values < 2 --- auto select.
      mode:        See numpy.numarray.nd_image.generic_filter documentation

    output: A 2D array of the same dimensions as mat and a float64 dtype

    Remarks:
      1. The spatial filter kernel size is ~4*xy_sig, unless specified otherwise
      2. The algorithm is slow but has a minimal memory footprint
    '''
    
    filter_fcn = _BB.Bilat_fcn(xy_sig, z_sig, filter_size)
    size = filter_fcn.xy_size
    return generic_filter(mat, filter_fcn.cfilter, size=size, mode=mode)

def bilateral_slow(mat, xy_sig, z_sig, filter_size=None, mode='reflect'):
    'A pure python implementation of the bilateral filter, for details see bilateral doc.'
    filter_fcn = _BB.Bilat_fcn(xy_sig, z_sig, filter_size)
    size = filter_fcn.xy_size
    return generic_filter(mat, filter_fcn, size=size, mode=mode)

def bilateral_fast(mat, xy_sig, z_sig, filter_size=None, mode='reflect'):
    'A fast implementation of bilateral filter, for details see bilateral doc.'
    filter_fcn = _BB.Bilat_fcn(xy_sig, z_sig, filter_size)
    size = filter_fcn.xy_size
    return generic_filter(mat, filter_fcn.fc_filter, size =size, mode=mode)
