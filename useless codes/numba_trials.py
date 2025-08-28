# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 02:06:57 2021

@author: cjh93
"""

from numba import jit, jitclass, njit, generated_jit
from numba import int32, float32
import numpy as np
import time


#@jit(nopython=True)
#def monte_carlo(n):
#    acc = 0
#    for i in range(n):
#        x = random.random()
#        y = random.random()
#        if (x**2 + y**2 ) < 1:
#            acc += 1
#    return 4*acc/n

spec = [
    ('s', float32),
    ('x', float32), 
    ('t', float32),   
    ('sigma', float32),   
    ('rf', float32),   
    ('div', float32),   
    ('dt', float32),
    ('n', int32)
]

@jitclass(spec)
class TreeOption():
    def __init__(self, s, x, t, sigma, rf, div, n):
        self.s = s
        self.x = x
        self.t = t
        self.sigma = sigma
        self.rf = rf
        self.div = div
        self.n = n
        self.dt = self.t/self.n
        
        self.stock_price_tree = np.ndarray(shape=(self.n+1,self.n+1))

    def __repr__(self):        
        return ('The option, based on tree method, with starting S0 of %4.2f, strike of %4.2f, time to maturity of %2.2f years, ''sigma of %2.3f, risk-free rate of %2.3f, dividend rate of %2.2f,' 
                % (self.s, self.x, self.t, self.sigma, self.rf, self.div) )

    def get_binomial_factors(self):
        u = np.exp(self.sigma * ((self.dt)**(1/2)))
        d = 1/u
        p = (np.exp((self.rf - self.div)*self.dt) -d) / (u - d)
        return [u,d,p]
    
    def indicator(self, x, y):
        return x>=y
    
    def build_binomial_stock_price_tree(self):
        u = self.get_binomial_factors()[0]
        d = self.get_binomial_factors()[1]
        res = np.ndarray(shape=(self.n+1,self.n+1))
        
        for i in range(self.n+1):
            res[i] = np.array([self.indicator(j,i) * self.s * d**(2*i) * u**j for j in range(self.n+1)])
            
        return res

if __name__ == '__main__': 
#    monte_carlo(1)
#    nsamples = 10000000
#    t0= time.process_time()
#    monte_carlo(nsamples)
#    t1 = time.process_time() - t0
#    print("Time elapsed: ", t1)
    
    s = 217.71
    x = 202.5
    t = 0.5
    sigma = 0.2
    rf = 0.028
    div = 0
    n = 100
    mkt_price = 0.79
    
    t0= time.process_time()
    option_test = TreeOption(s = s, x = x, t = t, sigma = sigma, rf = rf, n = 2, div = div)
    option_test.build_binomial_stock_price_tree()
    t1 = time.process_time() - t0
    print("Time elapsed: ", t1)

    t0= time.process_time()
    option1 = TreeOption(s = s, x = x, t = t, sigma = sigma, rf = rf, n = n, div = div)
    option1.build_binomial_stock_price_tree()
    t1 = time.process_time() - t0
    print("Time elapsed: ", t1)