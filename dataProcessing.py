#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:59:26 2018

@author: yangx
"""

import pywt
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import leastsq
from scipy import signal
from scipy.ndimage import filters
from sklearn.ensemble import RandomForestRegressor

def waveletDecomposition(image, level=3):
    """
    Plot of 2D wavelet decompositions for given number of levels.

    image needs to be either a colour channel or greyscale image:
        rgb: self.I[:, :, n], where n = {0, 1, 2}
        greyscale: use rgb_to_grey(self.I)

    """
    coeffs = pywt.wavedec2(image, wavelet='bior1.3', level=level)   
#    coeffs[0] /= np.abs(coeffs[0]).max()
#    for iloop in range(level):
#        coeffs[iloop + 1] = [d/np.abs(d).max() for d in coeffs[iloop + 1]]
    arr, slices = pywt.coeffs_to_array(coeffs)
#    for i, (cH, cV, cD) in enumerate(coeffs[1:]):
#        imd = np.concatenate((imd, cH), axis=1)
#        cVD = np.concatenate((cV, cD), axis=1)
#        imd = np.concatenate((imd, cVD), axis=0)
#        imd = np.concatenate((imd, cH.flatten()))
#        imd = np.concatenate((imd, cV.flatten()))
#        imd = np.concatenate((imd, cD.flatten()))
    return(arr)

def dimensionReduction(mat,num):
    pca = PCA(n_components=num)
    matDR = pca.fit_transform(mat)
    return(matDR)
    
def autoCorrelation(x):
    if x.ndim==1:
        n = len(x)
        x = np.array(x)
        variance = x.var()
        x = x-x.mean()
        corr =  np.correlate(x,x,mode='full')
        autocorr = corr[-n:]/(n*variance)
    elif x.ndim==2:
        (n,k) = x.shape
        autocorr = np.zeros((n,k))
        for iloop in range(k):
            autocorr[:,iloop] = autoCorrelation(x[:,iloop])
    else:
        print('Error-Only 1-D array or 2-D mat can be processed')
        return(-1)
#    result = np.correlate(x, x, mode = 'full')[-n+1:-n+lags+1]/(variance*(np.arange(n-1,n-1-lags,-1)))
    return(autocorr)  
    
def expDecayFit(y,t):
    p0 = [1,0,100]
    plsq = leastsq(residuals,p0, args=(y,t))
    print("拟合参数", plsq[0])
    return plsq

def func(t, p): 
    """ 数据拟合所用的函数: """
    A, B, tau = p
    return A*np.exp(-t/tau)+B

def residuals(p, y, x): 
    """ 实验数据x, y和拟合函数之间的差，p为拟合需要找到的系数 """
    return y - func(x, p)

def PSD(x,fs):
    f, Pxx_spec = signal.welch(x, fs, 'hann', 1980,1932)
#    f, Pxx_spec = signal.welch(x, fs, 'flattop', 1024, scaling='spectrum')
    plt.figure()
    plt.semilogy(f, np.sqrt(Pxx_spec))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Linear spectrum [V RMS]')
    plt.title('Power spectrum density (scipy.signal.welch)')
    plt.show()

def randomProjection(imt,ndim):
    (h,w,l) = imt.shape
    n = w*h
    imt = imt.reshape((n,l))
    transMat = np.random.rand(ndim,n)
    projMat = np.dot(transMat,imt)
    return(projMat)
    
    
def filteredDerivative(x,k,h,sigma):
    dx = x[k:,:]-x[:-k,:] 
    xb = np.int0(dx>h)
    xb = xb.sum(axis=0)
    xf = filters.gaussian_filter1d(xb,sigma)
    return

def randomForest4up(up,down,num):
    (m,n,k) = up.shape
    trainInd1 = np.random.randint(0,k,num)
    trainInd2 = np.random.randint(0,k,num)
    X = np.concatenate((up[...,trainInd1],down[...,trainInd2]),axis=2)
    X = X.reshape([m*n,-1])
    X = X.T
    y1 = np.ones(num)
    y2 = np.zeros(num)
    y = np.concatenate((y1,y2))    
    regr = RandomForestRegressor(n_estimators=100)
    regr.fit(X, y)
#    print(regr.predict(X))
    return(regr)