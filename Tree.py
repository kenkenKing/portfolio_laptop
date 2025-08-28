import numpy as np
import time


class TreeOption:
    def __init__(self, s, x, t, sigma, rf, div, n, call_put, style):
        self.s = s
        self.x = x
        self.t = t
        self.sigma = sigma
        self.rf = rf
        self.div = div
        self.n = n
        self.dt = self.t/self.n
        self.call_put = call_put
        self.style = style
        
        
    def __repr__(self):        
        return ('The option, based on tree method, with starting S0 of %4.2f, strike of %4.2f, time to maturity of %2.2f years, ''sigma of %2.3f, risk-free rate of %2.3f, dividend rate of %2.2f,' 
                % (self.s, self.x, self.t, self.sigma, self.rf, self.div) )

    def get_binomial_factors(self):
        u = np.exp(self.sigma * ((self.dt)**(1/2)))
        d = 1/u
        p = (np.exp((self.rf - self.div)*self.dt) -d) / (u - d)
        return [u,d,p]
    
    def build_binomial_stock_price_tree(self):
        u = self.get_binomial_factors()[0]
        d = self.get_binomial_factors()[1]
        res = np.ndarray(shape=(self.n+1,self.n+1))
        def indicator(x,y):
            return x>=y
        for i in range(self.n+1):
            res[i] = np.array([indicator(j,i) * self.s * d**(2*i) * u**j for j in range(self.n+1)],dtype = float)
        return res


    def build_euro_call_value_tree(self):
        a = self.build_binomial_stock_price_tree()
        
        def intrinsic(s, strike):
            return max(s-strike,0)
        np_intrinsic = np.vectorize(intrinsic,otypes=[float])
        a[:,self.n] = np_intrinsic(a[:,self.n], self.x)
        
        p = self.get_binomial_factors()[2]
        q = 1-p
        disfac = np.exp(-(self.rf-self.div)*self.dt)
        
        backward = self.n - 1
        while backward != -1:
            a[:self.n,backward] = disfac*(p*a[:self.n,backward+1] + q*a[1:,backward+1])
            backward -= 1
        return a
    
    def euro_call_value(self):
        return self.build_euro_call_value_tree()[0][0]

    def build_amer_call_value_tree(self):
        a = self.build_binomial_stock_price_tree()
        
        def intrinsic(s, strike):
            return max(s-strike,0)
        np_intrinsic = np.vectorize(intrinsic,otypes=[float])
        a[:,self.n] = np_intrinsic(a[:,self.n], self.x)
        
        p = self.get_binomial_factors()[2]
        q = 1-p
        disfac = np.exp(-(self.rf-self.div)*self.dt)

        def choose_large(x,y):
            if x>=y:
                return x
            else:
                return y
        np_choose_large = np.vectorize(choose_large)
        backward = self.n - 1
        while backward != -1:
            temp1 = disfac*(p*a[:self.n,backward+1] + q*a[1:,backward+1])
            temp2 = np_intrinsic(a[:self.n,backward], self.x)
            a[:self.n,backward] = np_choose_large(temp1,temp2)
            backward -= 1
        return a
    
    def amer_call_value(self):
        return self.build_amer_call_value_tree()[0][0]

    def build_euro_put_value_tree(self):
        a = self.build_binomial_stock_price_tree()
        
        def intrinsic(s, strike):
            return max(strike-s,0)
        np_intrinsic = np.vectorize(intrinsic,otypes=[float])
        a[:,self.n] = np_intrinsic(a[:,self.n], self.x)
        
        p = self.get_binomial_factors()[2]
        q = 1-p
        disfac = np.exp(-(self.rf-self.div)*self.dt)
        
        backward = self.n - 1
        while backward != -1:
            a[:self.n,backward] = disfac*(p*a[:self.n,backward+1] + q*a[1:,backward+1])
            backward -= 1
        return a
    
    def euro_put_value(self):
        return self.build_euro_put_value_tree()[0][0]

    def build_amer_put_value_tree(self):
        a = self.build_binomial_stock_price_tree()
        
        def intrinsic(s, strike):
            return max(strike-s,0)
        np_intrinsic = np.vectorize(intrinsic,otypes=[float])
        a[:,self.n] = np_intrinsic(a[:,self.n], self.x)
        
        p = self.get_binomial_factors()[2]
        q = 1-p
        disfac = np.exp(-(self.rf-self.div)*self.dt)

        def choose_large(x,y):
            if x>=y:
                return x
            else:
                return y
        np_choose_large = np.vectorize(choose_large)
        backward = self.n - 1
        while backward != -1:
            temp1 = disfac*(p*a[:self.n,backward+1] + q*a[1:,backward+1])
            temp2 = np_intrinsic(a[:self.n,backward], self.x)
            a[:self.n,backward] = np_choose_large(temp1,temp2)
            backward -= 1
        return a
    
    def amer_put_value(self):
        return self.build_amer_put_value_tree()[0][0]
    
    def value(self):
        if self.call_put == 'call' and self.style == 'american':
            return self.build_amer_call_value_tree()[0][0]
        
        elif self.call_put == 'put' and self.style == 'american':
            return self.build_amer_put_value_tree()[0][0]
        
        if self.call_put == 'call' and self.style == 'european':
            return self.build_euro_call_value_tree()[0][0]
        
        elif self.call_put == 'put' and self.style == 'european':
            return self.build_euro_put_value_tree()[0][0]
        
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
#            print(self.sigma)
        return self.sigma


if __name__ == '__main__':
    s = 217.71
    x = 202.5
    t = 0.5
    sigma = 0.2
    rf = 0.028
    div = 0
    n = 100
    call_put = 'put'
    style = 'american'
    mkt_price = 0.79
    
    option1 = TreeOption(s = s, x = x, t = t, sigma = sigma, rf = rf, n = n,div = div, call_put = 'call', style = 'american')
    option2 = TreeOption(s = s, x = x, t = t, sigma = sigma, rf = rf, n = n,div = div, call_put = 'put', style = 'american')
    option3 = TreeOption(s = s, x = x, t = t, sigma = sigma, rf = rf, n = n,div = div, call_put = 'call', style = 'european')
    option4 = TreeOption(s = s, x = x, t = t, sigma = sigma, rf = rf, n = n,div = div, call_put = 'put', style = 'european')

    print('amer call value is ', option1.value())
    print('amer put value is ', option2.value())
    print('euro call value is ', option3.value())
    print('euro put value is ', option4.value())
    
    t0= time.process_time()
    print(option1.value())
    t1 = time.process_time() - t0
    print("Time elapsed: ", t1)
    
    t0= time.process_time()
    print(option1.value())
    t1 = time.process_time() - t0
    print("Time elapsed: ", t1)
