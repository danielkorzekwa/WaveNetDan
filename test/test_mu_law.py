'''
Created on Sep 27, 2016

@author: daniel
'''
import unittest

from numpy.ma.testutils import assert_array_equal
from numpy.testing import *

import mu_law


class test_mu_law(unittest.TestCase):


    def test_encode_decode(self):
        
        encoded = mu_law.encode([-1, -0.5, 0, 0.5, 1], 256)
        
        assert_array_equal([0, 16, 128, 239, 255], encoded)
        
        assert_array_almost_equal(
            [ -1.000000e+00, -4.966767e-01, 8.621310e-05, 4.966767e-01, 1.000000e+00],
            mu_law.decode(encoded, 256),
            decimal=4)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
