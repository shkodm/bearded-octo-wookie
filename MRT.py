# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:00:07 2015

@author: michal
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:57:41 2015

@author: michal
"""

from bearded_octo_wookie.lbm import *

from sympy import *
from fractions import Fraction

W = np.array([Rational(_W) / Rational(36) for _W in W[:] * 36.])

U = [var('U_0'), var('U_1')]
rho = var('rho')


def genFeq(_e):

    cu =  3 * ( U[0] * _e[:,0] + U[1] * _e[:,1] ) 

    feq = W[:] * rho * (1 + cu[:] + Fraction(1./2.)*cu[:]*cu[:] - Fraction(3./2.) * ( U[0]**2 + U[1]**2 ) )

    feq = np.array( [ cancel(f) for f in feq] )

    return feq
    #pprint(feq)

def genM(_e):
    _M = np.zeros((9,9)).tolist()
    
    for i, v in enumerate(_e):

        _M[0][i] = 1.
        
        for k in range(1,3):
            _M[k][i] = v[k-1]

        _M[3][i] = -Rational(4) +  Rational(3) * (v[0] * v[0] + v[1] * v[1])
        _M[4][i] =   Rational(3) -   Rational(3) * (v[0] * v[0] + v[1] * v[1])
        
        for k in range(5,7):
            _M[k][i] = -v[k-5]        

        _M[7][i] = (v[0] * v[0] - v[1] * v[1]) 
        _M[8][i] = (v[0] * v[1])         
        
    return np.array(_M)


def getMRT(e0):

    feq0 = genFeq(e0)
    
    M0 = genM(e0)
    
    meq = list()
    for i, ms in enumerate( M0.dot(feq0)  ):
        meq.append(lambdify((rho, U[0], U[1]), ms))
    
    return lambda r,ux,uy: np.array([f(r,ux,uy) for f in meq]), M0
    #return  meq, M0