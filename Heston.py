import numpy as np
from scipy.stats import norm, qmc
import matplotlib.pyplot as plt
import time
from datetime import timedelta
import warnings 
warnings.filterwarnings("ignore")


def dirac(n):
    y = [0 for i in range(len(n))]
    y[0] = 1
    return y


class HestonOption:
    def __init__(self, s, x, t, rf, nu0, volvol, kappa, theta, rho, div):
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
    
    def forward_price(self):
        
        return self.s * np.exp((self.rf - self.div) * self.t)
    
    def characteristic_function(self,u):
        ima = np.complex(0,1)
        lamda = np.sqrt(self.volvol**2 *(u**2+ima*u)+(self.kappa-ima*self.volvol*self.rho*u)**2)
        
        omega = (np.exp(ima*u*np.log(self.s)+ima*u*(self.rf-self.div)*self.t + 
                       (self.kappa*self.theta*self.t*(self.kappa -ima*self.rho*self.volvol*u))/self.volvol**2 ) /
            (np.cosh(lamda*self.t/2) +(self.kappa-ima*self.rho*self.volvol*u)/lamda*np.sinh(lamda*self.t/2) )**(2*self.kappa
            *self.theta/ (self.volvol**2)))
            
        phi = omega*np.exp(-(u**2+ima*u)*self.nu0/(lamda/np.tanh(lamda*self.t/2)+self.kappa-ima*self.rho*self.volvol*u ))
        return (phi)                   

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
    
    def S_path(self,niter, nsteps, method = 'sobol'):
        
        if method == 'authentic':
        
            dt = self.t / nsteps
            time_grid = np.linspace(0, self.t, nsteps+1)
            
            # Initialize paths
            S = np.zeros((niter, nsteps+1))
            v = np.zeros((niter, nsteps+1))
            
            S[:,0] = self.s
            v[:,0] = self.nu0
            
            rng = np.random.default_rng()
            for t in range(1, nsteps+1):
                Z_v = rng.standard_normal(niter)
                Z_perp = rng.standard_normal(niter)
                Z_s = self.rho * Z_v + np.sqrt(1 - self.rho**2) * Z_perp
                
                # Variance update
                v[:,t] = np.maximum(
                    v[:,t-1] + self.kappa*(self.theta - v[:,t-1])*dt 
                    + self.volvol*np.sqrt(np.maximum(v[:,t-1],0))*np.sqrt(dt)*Z_v,
                    0
                )
                # Asset update
                S[:,t] = S[:,t-1] * np.exp(
                    (self.rf - self.div - 0.5*v[:,t-1])*dt + np.sqrt(np.maximum(v[:,t-1],0)*dt)*Z_s
                )

        elif method == 'sobol':
            dt = self.t / nsteps
            time_grid = np.linspace(0.0, self.t, nsteps + 1)
            
            # Sobol: dimension = 2 * n_steps (Zv and Zperp per step)
            dim = 2 * nsteps
            sampler = qmc.Sobol(d=dim, scramble=True)
            # Request n_paths points (Sobol requires n_paths to be power-of-two for best behaviour,
            # but scrambled Sobol works for other sizes too)
            U = sampler.random(niter)  # shape (n_paths, dim)
            # map uniforms to standard normals via inverse cdf
            Z_all = norm.ppf(U)          # shape (n_paths, dim)
            
            # split into Zv and Zperp arrays with shape (n_paths, n_steps)
            Zv = Z_all[:, 0::2]          # variance drivers: columns 0,2,4,...
            Zp = Z_all[:, 1::2]          # independent drivers: columns 1,3,5,...
         
            # build correlated driver for S
            Zs = self.rho * Zv + np.sqrt(max(0.0, 1 - self.rho**2)) * Zp
            
            # Prepare storage
            S = np.empty((niter, nsteps + 1), dtype=float)
            v = np.empty((niter, nsteps + 1), dtype=float)
            S[:, 0] = self.s
            v[:, 0] = self.nu0
        
            # time stepping: full-truncation Euler
            sqrt_dt = np.sqrt(dt)
            for t in range(1, nsteps + 1):
                zv = Zv[:, t-1]
                zs = Zs[:, t-1]
        
                # previous variance, enforce non-negativity inside sqrt
                v_prev = np.maximum(v[:, t-1], 0.0)
                sqrt_v_prev = np.sqrt(v_prev)
        
                # variance update: full-truncation Euler
                v_next = v[:, t-1] + self.kappa * (self.theta - v_prev) * dt + self.volvol * sqrt_v_prev * sqrt_dt * zv
                v_next = np.maximum(v_next, 0.0)  # full truncation (no negative variances)
                v[:, t] = v_next
        
                # asset update using v_prev (you may also use (v_prev+v_next)/2 for better accuracy)
                S[:, t] = S[:, t-1] * np.exp(
                    (self.rf - self.div - 0.5 * v_prev) * dt + sqrt_v_prev * sqrt_dt * zs
                )

        return S, v, time_grid
        
    def MCvalue(self,niter=50000,nsteps=100, method='sobol'):

        S_ = self.S_path(niter,nsteps,method)[0]
        ST = S_[:,-1]
        payoffs = np.maximum(ST - self.x,0)
        DF = np.exp(-self.rf * self.t)
        price = DF * payoffs.mean()
        
        se = DF * payoffs.std(ddof=1) / np.sqrt(S_.shape[0])
        ci95 = (price - 1.96*se, price + 1.96*se)
        return price, se, ci95
    
    def S_path_forward_price(self, niter=50000, nsteps=10, method = 'sobol'):
        ST = self.S_path(niter, nsteps, method)[0][:,-1]
        return ST.mean()
    
    def numerical_delta(self,alpha, n, B, ds):
        temp_Option_1 = HestonOption(self.s+ds, self.x, self.t, self.rf, self.nu0, self.volvol, 
                                   self.kappa, self.theta, self.rho,self.div)
        temp_Option_2 = HestonOption(self.s-ds, self.x, self.t, self.rf, self.nu0, self.volvol, 
                                   self.kappa, self.theta, self.rho,self.div)
        numerator = temp_Option_1.value(alpha,n,B) - temp_Option_2.value(alpha,n,B)
        denominator = 2*ds
        return numerator/denominator

if __name__ == '__main__':
    # test_param = [100,100,0.2465753,0.25,0.045,0.02]
    test_param = [100,100,1, 0.025, 0.06 , 0.25 ,  0.8 , 0.09 ,  -0.25, 0]
    #              s   x  t  rf   nu0   volvol   kappa  theta   rho  div
    testOption = HestonOption(test_param[0],test_param[1],test_param[2],test_param[3],test_param[4],
    test_param[5],test_param[6],test_param[7],test_param[8],test_param[9])

    print(testOption)
    print('The option has theoretical value:',testOption.value(1,11,250))
    # print('The numerical delta for the option is',testOption.numerical_delta(3,10,250,0.0001))
    
    # plt.plot(range(100),testOption.simulate_S_path(100))
    # plt.show()
    # plt.plot(range(10000),testOption.simulate_sig_path(10000))
    
    niter = 50000
    nsteps = 10
    start_time = time.time()
    print('The option has authentic MC value:',testOption.MCvalue(niter, nsteps)[0])
    print('The option has sobol MC value:',testOption.MCvalue(niter, nsteps, method='sobol')[0])
    print('option forward price:', testOption.forward_price())
    print('option simulated authentic forward price:', testOption.S_path_forward_price(niter, nsteps, method = 'authentic'))
    print('option simulated sobol forward price:', testOption.S_path_forward_price(niter, nsteps, method = 'sobol'))
    elapsed_time_secs = time.time() - start_time
    msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
    print(msg) 
