# https://github.com/ryukau/filter_notes/blob/master/gibbs/demo/gibbs.py
from gibbs import *

from scipy.special import binom

def applyFilter_(source, filterFunc, numSeries, power=1):
    spec = rfft(source)
    filt = numpy.zeros(len(spec))
    m = numSeries + 1
    for i in range(m):
        eta = i / m
        filt[i] = filterFunc(eta)**power
    
    applied = irfft(spec * filt)
    filt = numpy.append(filt[m:0:-1], filt[:m+1]) # for plot
    
    return (applied, filt)

def plotGibbsSuppression_(length, filterFunc, numSeries=16, power=1):
    source = additiveSaw(length, numSeries)
    suppressed, filt = applyFilter_(source, filterFunc, numSeries, power)
    #source = numpy.append(source, numpy.zeros(length))
    plot(
        source,
        suppressed,
        filt,
        filterFunc.__name__ + ' Filter',
        'result2_' + filterFunc.__name__ + '.png',
    )

'''
def fejer(eta): # eta < 0 があるなら要abs
    return 1 - abs(eta)
'''

def exponential(eta, alpha=37, p=4):
    return numpy.exp(-alpha * eta**p)

'''
def daubechies4(eta): # eta < 0 があるなら要abs
    return 1 - 35 * eta**4 + 84 * eta**5 - 70 * eta**6 + 20 * eta**7
'''

def daubechies(eta, p=4):
    seq = numpy.arange(p)
    sgn = numpy.where(seq%2==0, 1, -1)
    coeffs = factorial(2*p-1) / factorial(p-1)**2 * sgn * binom(p-1, seq) / (seq + p)
    polynomial = numpy.append(numpy.zeros_like(seq), coeffs)
    return 1 - numpy.polynomial.polynomial.polyval(eta, polynomial)

if __name__ == "__main__":
    plotGibbsSuppression_(1024, fejer, 16, 1)
    plotGibbsSuppression_(1024, lanczos, 16, 1)
    plotGibbsSuppression_(1024, raisedCosine, 16, 1)
    plotGibbsSuppression_(1024, sharpendRaisedCosine, 16, 1)
    plotGibbsSuppression_(1024, exponential, 16, 1)
    plotGibbsSuppression_(1024, daubechies, 16, 1)
