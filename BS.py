# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from scipy.stats import norm

"""
Created on Thu Oct 26 01:01:56 2017

@author: Jinghao Chen
"""

class BSOption:
    def __init__(self, s, x, t, sigma, rf, div, call_put):
        self.s = s
        self.x = x
        self.t = t
        self.sigma = sigma
        self.rf = rf
        self.div = div
        self.call_put = call_put
    
    def __repr__(self):        
        return ('The BS option with starting S0 of %4.2f, strike of %4.2f, time to maturity of %2.2f years, '
                'sigma of %2.3f, risk-free rate of %2.3f, dividend rate of %2.2f' 
                % (self.s, self.x, self.t, self.sigma, self.rf, self.div) )
        
    def d1(self):
        a = 1/(self.sigma * (self.t ** (1/2)))
        b = np.log(self.s / self.x)
        c = (self.rf - self.div + (self.sigma**2)/2)*self.t
        return a * (b + c)
    
    def d2(self):
        return self.d1() - (self.sigma * ((self.t) ** (1/2)))
        
    def nd1(self):
        return norm.cdf(self.d1())
    
    def nd2(self):
        return norm.cdf(self.d2())
    
    
    def value(self):
        if self.call_put == 'call':
            return (self.nd1() * self.s * np.exp(-self.div * self.t) - 
                    self.nd2() * self.x *np.exp(-self.rf * self.t) )
        elif self.call_put == 'put':
            return ( (1 - self.nd2()) * self.x * np.exp(-self.rf * self.t) - 
                    (1 - self.nd1()) * self.s *np.exp(-self.div * self.t) )
    
    def MCvalue(self,niter = 10000):
        if self.call_put == 'call':
            desire = 0
            for i in range(niter):
                cum_sum = self.s*np.exp( (self.rf -self.div -self.sigma**2/2) * self.t +self.sigma*np.sqrt(self.t)*
                                        np.random.normal(0,1))
                if cum_sum - self.x > 0:
                    desire += cum_sum-self.x
            return (np.exp(-self.rf*self.t)*desire/niter)
        
        elif self.call_put == 'put':
            desire = 0
            for i in range(niter):
                cum_sum = self.s*np.exp( (self.rf -self.div -self.sigma**2/2) * self.t +self.sigma*np.sqrt(self.t)*
                                        np.random.normal(0,1))
                if cum_sum - self.x < 0:
                    desire -=  (cum_sum-self.x)
            return (np.exp(-self.rf*self.t)*desire/niter)
            
            
    def delta(self): 
        if self.call_put == 'call':
            return  np.exp(-self.div*self.t)*self.nd1()
        elif self.call_put == 'put':       
            return np.exp(-self.div*self.t)*(self.nd1() -1)
        
    def put_call_parity_price(self):
        if self.call_put == 'call':
            return self.value() - self.s*np.exp(-self.div * self.t) + self.x*np.exp(-self.rf * self.t)
        elif self.call_put == 'put':
            return self.value() + self.s*np.exp(-self.div * self.t) - self.x*np.exp(-self.rf * self.t)
        
    def gamma(self):
        cons_part = 1/np.sqrt(2*np.pi)
        top = np.exp(-self.div*self.t)
        bottom = self.s * self.sigma * np.sqrt(self.t)
        return(cons_part*top*np.exp(-self.d1()**2/2) /bottom)

            
    def theta(self):
        if self.call_put == 'call':
            first = -(self.s*self.sigma*np.exp(-self.div*self.t)*np.exp(-self.d1()**2/2)/ 
                     (2*np.sqrt(self.t*2*np.pi) ) )
            second = -self.rf*self.x*np.exp(-self.rf*self.t) *self.nd2()       
            third = self.div+self.s*np.exp(-self.div*self.t)*self.nd1()
            return(  (first+second+third)/self.t )
        elif self.call_put == 'put':
            first = -(self.s*self.sigma*np.exp(-self.div*self.t)*np.exp(-self.d1()**2/2)/ 
                     (2*np.sqrt(self.t*2*np.pi) ) )
            second = self.rf*self.x*np.exp(-self.rf*self.t) *(1- self.nd2())     
            third = -self.div+self.s*np.exp(-self.div*self.t)*(1-self.nd1())
            return(  (first+second+third)/self.t )
            
    def vega(self):
        if self.call_put == 'call':
            return(self.s*np.exp(-self.div*self.t-self.d1()**2/2)*np.sqrt(self.t)/ 
                   (100*np.sqrt(2*np.pi)) )                 
        elif self.call_put == 'put':
            return(self.s*np.exp(-self.div*self.t-self.d1()**2/2)*np.sqrt(self.t)/ 
               (100*np.sqrt(2*np.pi)) )
            
    def rho(self):
        if self.call_put == 'call':
            return (self.x*self.t*np.exp(-self.rf*self.t)*self.nd2()*0.01)
        elif self.call_put == 'put':
            return (-self.x*self.t*np.exp(-self.rf*self.t)*(1-self.nd2())*0.01)
        
    def S(self):
        return self.s*np.exp( (self.rf -self.div -self.sigma**2/2) * self.t +self.sigma*np.sqrt(self.t)*
                                    np.random.normal(0,1))
    
    def S_path(self,tstep = 252):
        path = [self.s]
        dt = self.t / tstep
        for i in range(1,tstep):
            path.append(path[-1]+self.rf*path[-1]*dt + self.sigma*path[-1]*np.random.normal(0,np.sqrt(dt)))
        return path
    
    def implied_volatility(self, mkt_price, err_thsld = 0.01):
        lb = 0
        ub = 2
        
        self.sigma = ub
        if self.value() < mkt_price:
            return np.nan
        
        self.sigma = lb
        if self.value() > mkt_price:
            return np.nan
        
        test_rate = (lb+ub)/2
        self.sigma = test_rate
        error = abs(self.value() - mkt_price)
        
        while abs(error) >= err_thsld:
            if self.value() - mkt_price > 0:
                ub = ub - (ub-lb)/2
                self.sigma = (ub+lb) /2
                error = (self.value() - mkt_price)
            else:
                lb = lb + (ub-lb)/2
                self.sigma = (ub+lb)/2
                error = (self.value() - mkt_price)
                
            if self.sigma < 0.001:
                return np.nan
        return self.sigma

if __name__ == '__main__':
    s = 100
    x = 100
    t = 0.38
    sigma = 0.25
    rf = 0.01 
    div = 0
    call_put = 'call'
    
    option1 = BSOption(s = s, x = x, t = t, sigma = sigma, rf = rf, div = div, call_put = 'call')
    option2 = BSOption(s = s, x = x, t = t, sigma = sigma, rf = rf, div = div, call_put = 'put')
    
    print('euro call value is ', option1.value())
    print('euro put value is ', option2.value())