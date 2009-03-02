import bilateral
import numpy as np
from numpy.testing import assert_array_almost_equal

def test_basic():
    expected = np.array([[ 7,  7,  7,  8,  8],
                         [ 9,  9,  9, 10, 10],
                         [11, 11, 12, 12, 12],
                         [13, 13, 14, 14, 14],
                         [15, 15, 16, 16, 16]])
    out = bilateral.bilateral(np.arange(25).reshape((5,5)), 4, 10)
    print out
    print expected
    assert_array_almost_equal(out, expected)
    return out, expected
