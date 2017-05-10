#!/usr/bin/python2.7
import numpy as np
import LeastSquare

class PyLeastSquare:
    def __init__(self, func, numParameter, numDimX, dtype = 'real'):
	x = np.random.random([numDimX])
        p = np.random.random([numParameter])

        try:
            func(p, x)
        except:
            raise TypeError("Error! check your function!")

        if dtype == 'real':
            self.lm = LeastSquare.LeastSquareReal(func, numParameter, numDimX)
            self.dtype = 'float64'
        elif dtype == 'complex':
            self.lm = LeastSquare.LeastSquareComplex(func, numParameter, numDimX)
            self.dtype = 'complex128'
        else:
            raise TypeError("Error! check your argument, dtype.")

        self.numParameter = numParameter
        self.numDimX = numDimX


    def in_data(self, x, y):
        x = np.array(x, dtype = self.dtype); x = x.reshape([-1, self.numDimX])
        y = np.array(y, dtype = self.dtype)

        assert(len(x) == len(y))

        self.lm.in_data(x, y)


    def fitting(self, p, numIter = int(1e4), tol = 1e-5, printlog = False):
        assert(len(p) == self.numParameter)

        pIn = np.array(p, dtype = 'float64')

        self.lm.fitting(pIn, numIter, tol, printlog)

        return pIn



if __name__ == "__main__":

    numSample = 1000
    pDim = 4
    xDim = 1

    def model_sine(p, x):
        return p[0]*np.sin(p[1]*x[0] + p[2]) + p[3]

    lm = PyLeastSquare(model_sine, pDim, xDim, 'real')

    p = np.array([3, 2, 3, 4])
    x = np.random.random([numSample, xDim])
    y = np.array([model_sine(p, x[i]) for i in xrange(numSample)]) + np.random.random()*1e-1

    lm.in_data(x, y)
    pred = np.random.random([pDim])

    print lm.fitting(pred, 10000, 1e-5, True)
