import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import timedelta


def dirac(n):
    y = [0 for i in range(len(n))]
    y[0] = 1
    return y


class HestonOption:
    def __init__(self, s, x, t, rf, nu0, volvol, kappa, theta, rho,div):
        self.s = s
        self.x = x
        self.t = t
        self.rf = rf
        self.nu0 = nu0
        self.volvol = volvol
        self.kappa = kappa
        self.theta = theta
        self.rho = rho
        self.div = div
    
    def __repr__(self):        
        return ('The Heston option with starting S0 of %4.2f, strike of %4.2f, time to maturity of %2.2f years, '
                'risk-free rate of %2.3f' 
                % (self.s, self.x, self.t, self.rf) )
    
    def characteristic_function(self,u):
        ima = np.complex(0,1)
        lamda = np.sqrt(self.volvol**2 *(u**2+ima*u)+(self.kappa-ima*self.volvol*self.rho*u)**2)
        
        omega = (np.exp(ima*u*np.log(self.s)+ima*u*(self.rf-self.div)*self.t + 
                       (self.kappa*self.theta*self.t*(self.kappa -ima*self.rho*self.volvol*u))/self.volvol**2 ) /
            (np.cosh(lamda*self.t/2) +(self.kappa-ima*self.rho*self.volvol*u)/lamda*np.sinh(lamda*self.t/2) )**(2*self.kappa
            *self.theta/ (self.volvol**2)))
            
        phi = omega*np.exp(-(u**2+ima*u)*self.nu0/(lamda/np.tanh(lamda*self.t/2)+self.kappa-ima*self.rho*self.volvol*u ))
        return (phi)            
    
    def MCvalue(self,tstep,niter):
        desire = 0
        dt = self.t/tstep
        for i in range(niter):
            cum_sum = self.s
            sig = self.nu0
            for j in range(tstep):
                
                rand1 = np.random.normal(0,np.sqrt(dt))
                rand2 = rand1*self.rho + np.sqrt(1 - self.rho**2)*np.random.normal(0,np.sqrt(dt))
                
                cum_sum += self.rf*cum_sum*dt+np.sqrt(sig)*cum_sum*rand2
                sig += self.kappa*(self.theta-sig)*dt + self.volvol*np.sqrt(sig)*rand1
                
                if sig < 0:
                    sig = 0

            if cum_sum - self.x > 0:
                desire += cum_sum-self.x
                
        return (np.exp(-self.rf*self.t)*desire/niter)        

    def value(self, alpha, n, B):
        N = 2**n
        eta = B/N
        lambda_eta = 2*np.pi/N
        lamda = lambda_eta/eta
        
        J = range(1,N+1)
        vj = [(i-1)*eta for i in J]
        m = range(1,N+1)
        beta = np.log(self.x) - lamda*N /2
        km = [beta + (i-1)*lamda for i in m]
        
        ima = np.complex(0,1)
        psi_vj = [0 for i in range(len(J))]
        for zz in range(1,N+1):
            u = vj[zz-1] - (alpha+1.0)*ima
            numer = self.characteristic_function(u)
            denom = ( (alpha+ima*vj[zz-1]   ) * (alpha+1.0+ima*vj[zz-1])  )
            psi_vj[zz-1] = numer/denom
        
        XX = [(eta/2)*psi_vj[i]*np.exp(-ima*beta*vj[i])*(2-(dirac(J)[i])) for i in range(len(J))]
        ZZ = np.fft.fft(XX)
        
        multiplier = [np.exp(-alpha * i )/np.pi for i in km]
        ZZ2 = [multiplier[i]*np.real(ZZ[i]) for i in range(len(ZZ))]
        
        fft_price = np.exp(-self.rf*self.t)*ZZ2[int(N/2)]
        return fft_price

    def S(self,tstep):
        cum_sum = self.s
        sig = self.nu0
        dt = self.t/tstep
        for j in range(tstep):
                
            rand1 = np.random.normal(0,np.sqrt(dt))
            rand2 = rand1*self.rho + np.sqrt(1 - self.rho**2)*np.random.normal(0,np.sqrt(dt))
                
            cum_sum += self.rf*cum_sum*dt+np.sqrt(sig)*cum_sum*rand2
            sig += self.kappa*(self.theta-sig)*dt + self.volvol*np.sqrt(sig)*rand1

            if sig < 0:
                sig = 0

        return cum_sum
    
    def simulate_S_path(self,tstep):
        path = [self.s]
        dt = self.t / tstep
        sig = self.nu0
        for j in range(1,tstep):
                
            rand1 = np.random.normal(0,np.sqrt(dt))
            rand2 = rand1*self.rho + np.sqrt(1 - self.rho**2)*np.random.normal(0,np.sqrt(dt))
                
            path.append(path[-1]+path[-1]*self.rf*dt+np.sqrt(sig)*path[-1]*rand2 )
            sig += self.kappa*(self.theta-sig)*dt + self.volvol*np.sqrt(sig)*rand1

            if sig < 0:
                sig = 0
        return path
    
    def simulate_sig_path(self,tstep):
        path = [self.nu0]
        dt = self.t / tstep
        sig = self.nu0
        for j in range(1,tstep):
            rand1 = np.random.normal(0,np.sqrt(dt))
            sig += self.kappa*(self.theta-sig)*dt + self.volvol*np.sqrt(sig)*rand1
            if sig < 0:
                sig = 0
            path.append(sig)
        return path
    
    def numerical_delta(self,alpha, n, B, ds):
        temp_Option_1 = HestonOption(self.s+ds, self.x, self.t, self.rf, self.nu0, self.volvol, 
                                   self.kappa, self.theta, self.rho,self.div)
        temp_Option_2 = HestonOption(self.s-ds, self.x, self.t, self.rf, self.nu0, self.volvol, 
                                   self.kappa, self.theta, self.rho,self.div)
        numerator = temp_Option_1.value(alpha,n,B) - temp_Option_2.value(alpha,n,B)
        denominator = 2*ds
        return numerator/denominator

if __name__ == '__main__':
    test_param = [100,100,0.2465753,0.25,0.045,0.02]
    test_param = [100,100,1, 0.025, 0.06 , 0.25 ,  0.8 , 0.09 ,  -0.25, 0]
    #              s   x  t  rf   nu0   volvol   kappa  theta   rho  div
    testOption = HestonOption(test_param[0],test_param[1],test_param[2],test_param[3],test_param[4],
    test_param[5],test_param[6],test_param[7],test_param[8],test_param[9])

    print(testOption)
    print('The option has theoretical value:',testOption.value(1,10,250))
    print('The option has simulated final underlying value:',testOption.S(100))
    print('The numerical delta for the option is',testOption.numerical_delta(3,10,250,0.0001))
    
    plt.plot(range(100),testOption.simulate_S_path(100))
    plt.show()
    plt.plot(range(10000),testOption.simulate_sig_path(10000))
    
    start_time = time.time()
    print('The option has MC value:',testOption.MCvalue(100,5000))
    elapsed_time_secs = time.time() - start_time
    msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
    print(msg) 
