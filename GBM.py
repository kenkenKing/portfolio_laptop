import numpy as np
from scipy.stats import norm

class GBM:
    def __init__(self, s, x, t, sigma, rf, div, call_put, position = 1, shares = 1):
        self.s = s
        self.x = x
        self.t = t
        self.sigma = sigma
        self.rf = rf
        self.div = div
        self.call_put = call_put
        self.position = position
        self.shares = shares
    
    def __repr__(self):     
        if self.call_put == 1: 
            call_or_put = 'call'  
        else:
            call_or_put = 'put'
            
        if self.position == 1:
            long_or_short = 'long'
        else:
            long_or_short = 'short'
        
        notional = self.x*self.shares
        string = f'The BS option with starting S0 of {self.s}, strike of {self.x}, time to maturity of {self.t} years, sigma of {self.sigma}, risk-free rate of {self.rf}, dividend rate of {self.div}, {call_or_put}, {long_or_short} position, {self.shares} shares, notional of {notional}.' 
        return string
        # return ('The BS option with starting S0 of %4.2f, strike of %4.2f, time to maturity of %2.2f years, '
        #         'sigma of %2.3f, risk-free rate of %2.3f, dividend rate of %2.2f' 
        #         % (self.s, self.x, self.t, self.sigma, self.rf, self.div) )
    
    def notional(self):
        return self.x * self.shares
    
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
            return self.position*self.shares*((self.nd1() * self.s * np.exp(-self.div * self.t) - 
                                                 self.nd2() * self.x *np.exp(-self.rf * self.t) ))
        elif self.call_put == 'put':
            return self.position*self.shares*(((1 - self.nd2()) * self.x * np.exp(-self.rf * self.t) - 
                                                 (1 - self.nd1()) * self.s *np.exp(-self.div * self.t)))
    
    def MCvalue(self,niter = 10000):
        z = np.random.normal(0, 1, size=niter)
        ST = self.s*np.exp((self.rf -self.div -self.sigma**2/2) * self.t +self.sigma * np.sqrt(self.t)*z)
        
        if self.call_put == 'call':
            payoff = np.maximum(ST - self.x, 0)
            return self.position * self.shares * np.exp(-self.rf*self.t) * np.mean(payoff)
        
        elif self.call_put == 'put':
            payoff = np.maximum(self.x - ST, 0)
            return self.position * self.shares * np.exp(-self.rf*self.t) * np.mean(payoff)
            
    def CVA_adjusted(self, hazard_rate, recovery_rate = 0.4, niter=50000, nsteps = 100):
        '''
        Credit Value Adjustment, Expected loss due to counterparty default
        '''
        dt = self.t/nsteps
        time_grid = np.linspace(0, self.t, nsteps + 1)
        
        # Simulate GBM paths
        S = self.S_path(niter, nsteps)
        
        # Option exposure
        if self.call_put == 'call':
            V = np.maximum(S - self.x, 0)
        
        elif self.call_put == 'put':
            V = np.maximum(self.x - S, 0)
  
        # Simulate default times for each path
        U = np.random.rand(niter)
        tau = -np.log(U)/hazard_rate
  
        # Exposure conditional on default
        Exposure = np.zeros_like(V)
        for i in range(niter):
            default_idx = np.searchsorted(time_grid, tau[i])
            Exposure[i, :default_idx+1] = V[i, :default_idx+1]
            Exposure[i, default_idx+1:] = 0
        
        # Expected Exposure at each time step
        EE = Exposure.mean(axis=0)
        
        # Incremental default probabilities
        PD = 1 - np.exp(-hazard_rate * time_grid)
        dPD = np.diff(np.insert(PD,0,0)) 
        
        # Discount factors
        DF = np.exp(-self.rf * time_grid)
        
        CVA = (1 - recovery_rate) * np.sum(EE * dPD * DF)
        return self.position * self.shares * CVA
    
    def DVA_adjusted(self, hazard_rate, recovery, hazard_cpt=0.02, recovery_cpt=0.4, niter=50000, nsteps=100, DVA_type = 'unilateral'):
        '''
        Debit Value Adjustment, Expected gain due to own default
        
        Parameters
        ----------
        hazard_rate : float
            our hazard rate.
        recovery : float
            our recovery rate.
        hazard_rate_cpt : float
            our hazard rate for counterparty, only needed if bilateral DVA.
        recovery_cpt : float
            our recovery rate for counterparty, only needed if bilateral DVA.
        niter : int
            simulation paths.
        nsteps : int
            time steps for each simulation path.
        DVA_type : string, unilateral or bilateral
            type of the DVA value, 
        Returns
        -------
        DVA value
        float.

        '''
        dt = self.t/nsteps
        time_grid = np.linspace(0, self.t, nsteps + 1)
        
        S = self.S_path(niter, nsteps)
        
        # Option exposure
        if self.call_put == 'call':
            V = np.maximum(S - self.x, 0)
        
        elif self.call_put == 'put':
            V = np.maximum(self.x - S, 0)
        
        # Position sign makes it a liability if you're short
        V = self.position * V
        
        # Positive/negative exposure profiles (unconditional, averaged over market paths)
        EE_pos = np.maximum(V,0).mean(axis=0) # used for CVA (not needed here)
        EE_neg = np.maximum(-V,0).mean(axis=0) # used for DVA

        LGD = 1 - recovery
        DF = np.exp(-self.rf * time_grid)
        
        survival = np.exp(-hazard_rate * time_grid) 
        PD = 1 - survival
        dPD = np.diff(PD) # positive increments
        
        # DVA formulas
        # (A) Unilateral DVA (ignores counterparty default)
        #     DVA_uni ≈ LGD_B * sum_t EE_neg(t_i) * dPD_B(t_i) * DF(t_i)
        if DVA_type == 'unilateral':
            DVA = LGD * np.sum(EE_neg[1:] * dPD * DF[1:])
        
        # (B) Bilateral DVA (your default must happen while counterparty is alive)
        #     Weight each time bucket by P(you default in bucket AND C survives to t_i)
        #     Under independence: weight_i = dPD_B(t_i) * S_C(t_i)
        elif DVA_type == 'bilateral': 
            survival_cpt = np.exp(-hazard_cpt * time_grid)
            weights_bilateral = dPD * survival_cpt[1:]
            DVA = LGD * np.sum(EE_neg[1:] * weights_bilateral * DF[1:])
        return self.shares*DVA

    def FVA_adjusted(self, funding_spread, include_benefit, niter=50000, nsteps = 100):
        '''

        Parameters
        ----------
        funding_spread : float
            funding spread, funding cost - collateral remuneration..
        include_benefit : boolean
            to condition if including the benefit or not.
        niter : int
            simulation paths. The default is 50000.
        nsteps : int
            time steps for each simulation path. The default is 100.

        Returns
        -------
        FVA.

        '''
        dt = self.t/nsteps
        time_grid = np.linspace(0, self.t, nsteps + 1)
        
        S = self.S_path(niter, nsteps)
        
        # Option exposure
        if self.call_put == 'call':
            V = np.maximum(S - self.x, 0)
        
        elif self.call_put == 'put':
            V = np.maximum(self.x - S, 0)
            
        # Exposure expectation (funding need)
        if include_benefit:
            EE = V.mean(axis=0)  # allow positive & negative
        else:
            EE = np.maximum(V,0).mean(axis=0)  # only funding cost
        
        DF = np.exp(-self.rf * time_grid)
        
        # FVA calculation
        FVA = funding_spread * np.sum(EE[1:] * DF[1:] * dt)
        return FVA

    def MVA_adjusted(self, funding_spread, alpha, niter=50000, nsteps=100):
        '''

        Parameters
        ----------
        funding_spread : float
            funding spread, funding cost - collateral remuneration.
        alpha : float
            IM as the alpha quantile of the exposure distribution (99% quantile of loss).
        niter : int
            simulation paths. The default is 50000.
        nsteps : int
            time steps for each simulation path. The default is 100.

        Returns
        -------
        MVA
        float.

        '''
        
        dt = self.t/nsteps
        time_grid = np.linspace(0, self.t, nsteps + 1)
        
        S = self.S_path(niter, nsteps)
       
        # Option exposure
        if self.call_put == 'call':
            V = np.maximum(S - self.x, 0)
        
        elif self.call_put == 'put':
            V = np.maximum(self.x - S, 0)
        
        # Initial Margin proxy = quantile of exposures at each time
        IM = np.quantile(V, alpha, axis=0)
        
        DF = np.exp(-self.rf * time_grid)
        
        MVA = funding_spread * np.sum(IM[1:] * DF[1:] * dt)
        return MVA
    
    def KVA_adjusted(self, capital_factor, hurdle_rate, niter=50000, nsteps=100):
        '''

        Parameters
        ----------
        capital_factor : float
            capital factor.
        hurdle_rate : float
            hurdle rate.
        niter : int, optional
            simulation paths. The default is 50000.
        nsteps : int, optional
            time steps for each simulation path. The default is 100.

        Returns
        -------
        KVA : float
            DESCRIPTION.

        '''
            
        dt = self.t/nsteps
        time_grid = np.linspace(0, self.t, nsteps + 1)
        
        S = self.S_path(niter, nsteps)
        
        # Option exposure
        if self.call_put == 'call':
            V = np.maximum(S - self.x, 0)
        
        elif self.call_put == 'put':
            V = np.maximum(self.x - S, 0)
            
        # Expected positive exposure
        EE = np.maximum(V,0).mean(axis=0)
        
        # Capital requirement proxy (Basel style: % of exposure)
        capital = capital_factor * EE
        
        # Discount factor
        DF = np.exp(-self.rf * time_grid)
    
        # KVA calculation
        KVA = hurdle_rate * np.sum(capital[1:] * DF[1:] * dt)
        return KVA
    
    def risk_adjusted(self, 
                      hazard_cpt, 
                      recovery_cpt, 
                      hazard, 
                      recovery, 
                      funding_spread_uncollateralized_exposure, 
                      include_benefit,
                      funding_spread_IM,
                      alpha,
                      capital_factor,
                      hurdle_rate,
                      DVA_type='unilateral',
                      niter=50000,
                      nsteps=100
                      ):
        
        dt = self.t/nsteps
        time_grid = np.linspace(0, self.t, nsteps + 1)
        
        # Simulate GBM paths
        S = self.S_path(niter, nsteps)
        
        # Option exposure
        if self.call_put == 'call':
            payoff = np.maximum(S - self.x, 0)
        
        elif self.call_put == 'put':
            payoff = np.maximum(self.x - S, 0)
        
        # CVA: 
        # Simulate default times for each path
        U = np.random.rand(niter)
        tau = -np.log(U)/hazard_cpt
  
        # Exposure conditional on default
        Exposure = np.zeros_like(payoff)
        for i in range(niter):
            default_idx = np.searchsorted(time_grid, tau[i])
            Exposure[i, :default_idx+1] = payoff[i, :default_idx+1]
            Exposure[i, default_idx+1:] = 0
        
        # Expected Exposure at each time step
        EE_cva = Exposure.mean(axis=0)
        
        # Incremental default probabilities
        PD_cpt = 1 - np.exp(-hazard_cpt * time_grid)
        dPD_cpt = np.diff(np.insert(PD_cpt,0,0)) 
        
        # Discount factors
        DF = np.exp(-self.rf * time_grid)
        
        CVA = (1 - recovery_cpt) * np.sum(EE_cva * dPD_cpt * DF)
        
        # DVA:
        V = self.position * payoff
        
        EE_neg = np.maximum(-V,0).mean(axis=0) # used for DVA

        LGD = 1 - recovery
        
        survival = np.exp(-hazard_rate * time_grid) 
        PD = 1 - survival
        dPD = np.diff(PD) # positive increments
        
        if DVA_type == 'unilateral':
            DVA = LGD * np.sum(EE_neg[1:] * dPD * DF[1:])
        
        elif DVA_type == 'bilateral': 
            survival_cpt = np.exp(-hazard_cpt * time_grid)
            weights_bilateral = dPD * survival_cpt[1:]
            DVA = LGD * np.sum(EE_neg[1:] * weights_bilateral * DF[1:])    
        
        # FVA:
        if include_benefit:
            EE_fva = payoff.mean(axis=0)  # allow positive & negative
        else:
            EE_fva = np.maximum(payoff,0).mean(axis=0)  # only funding cost
        
        FVA = funding_spread_uncollateralized_exposure * np.sum(EE_fva[1:] * DF[1:] * dt)
        
        # MVA:
        IM = np.quantile(payoff, alpha, axis=0)
        
        MVA = funding_spread_IM * np.sum(IM[1:] * DF[1:] * dt)    
            
        # KVA:
        EE_mva = np.maximum(payoff,0).mean(axis=0)

        capital = capital_factor * EE_mva

        KVA = hurdle_rate * np.sum(capital[1:] * DF[1:] * dt)    
        
        return self.position*(- CVA + DVA - FVA - MVA - KVA)
    
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
    
    def S_path(self,niter=50000, nsteps=252):
        dt = self.t/nsteps
        rng = np.random.default_rng()
        z = rng.standard_normal((niter, nsteps)) # does the same as z = np.random.randn(niter, nsteps)
        increments = np.exp((self.rf - self.div - self.sigma**2/2) * dt + self.sigma * np.sqrt(dt) * z)
        S = np.zeros((niter, nsteps+1))
        S[:,0] = self.s
        S[:,1:] = self.s * np.cumprod(increments, axis = 1)
        return S
    
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
    hazard_rate = 0.02
    recovery = 0.4
    funding_spread = 0.01
    include_benefit = True
    alpha = 0.99
    capital_factor = 0.08
    hurdle_rate = 0.1
    
    # niter = 
    # nsteps = 5
    
    
    option1 = GBM(s = s, x = x, t = t, sigma = sigma, rf = rf, div = div, call_put = 'call')
    option2 = GBM(s = s, x = x, t = t, sigma = sigma, rf = rf, div = div, call_put = 'put')
    option3 = GBM(s = s, x = x, t = t, sigma = sigma, rf = rf, div = div, call_put = 'call', position = -1)
    option4 = GBM(s = s, x = x, t = t, sigma = sigma, rf = rf, div = div, call_put = 'put', position = -1)
    
    print('euro call value is ', option1.value())
    print('euro put value is ', option2.value())
    print('euro call MC value is ', option1.MCvalue(50000))
    print('euro put MC value is ', option2.MCvalue(50000))
    
    print('euro call MC CVA value is ', option1.CVA_adjusted(hazard_rate))
    print('euro put MC CVA value is ', option2.CVA_adjusted(hazard_rate))
    
    print('long euro call MC DVA value is ', option1.DVA_adjusted(hazard_rate,recovery))
    print('long euro put MC DVA value is ', option2.DVA_adjusted(hazard_rate,recovery))
    print('short euro call MC DVA value is ', option3.DVA_adjusted(hazard_rate,recovery))
    print('short euro put MC DVA value is ', option4.DVA_adjusted(hazard_rate,recovery))
    
    print('euro call MC FVA value is ', option1.FVA_adjusted(funding_spread, include_benefit))
    print('euro put MC FVA value is ', option2.FVA_adjusted(funding_spread, include_benefit))
    
    print('euro call MC MVA value is ', option1.MVA_adjusted(funding_spread, alpha))
    print('euro put MC MVA value is ', option2.MVA_adjusted(funding_spread, alpha))
    
    print('euro call MC KVA value is ', option1.KVA_adjusted(capital_factor, hurdle_rate))
    print('euro put MC KVA value is ', option2.KVA_adjusted(capital_factor, hurdle_rate))
    
    print('euro call MC XVA value is ', option1.risk_adjusted(
                                                                hazard_rate, 
                                                                recovery, 
                                                                hazard_rate, 
                                                                recovery, 
                                                                funding_spread, 
                                                                include_benefit,
                                                                funding_spread,
                                                                alpha,
                                                                capital_factor,
                                                                hurdle_rate,
                                                               ))
    print('euro put MC XVA value is ', option2.risk_adjusted(
                                                                hazard_rate, 
                                                                recovery, 
                                                                hazard_rate, 
                                                                recovery, 
                                                                funding_spread, 
                                                                include_benefit,
                                                                funding_spread,
                                                                alpha,
                                                                capital_factor,
                                                                hurdle_rate,
                                                               ))