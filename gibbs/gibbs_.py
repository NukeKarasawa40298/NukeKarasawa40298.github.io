# https://github.com/ryukau/filter_notes/blob/master/gibbs/demo/gibbs.py
from gibbs import *

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

def daubechies4(eta): # eta < 0 があるなら要abs
    return 1 - 35 * eta**4 + 84 * eta**5 - 70 * eta**6 + 20 * eta**7

if __name__ == "__main__":
    plotGibbsSuppression_(1024, fejer, 16, 1)
    plotGibbsSuppression_(1024, lanczos, 16, 1)
    plotGibbsSuppression_(1024, raisedCosine, 16, 1)
    plotGibbsSuppression_(1024, sharpendRaisedCosine, 16, 1)
    plotGibbsSuppression_(1024, exponential, 16, 1)
    plotGibbsSuppression_(1024, daubechies4, 16, 1)
