
import mlmc

def euler_step(X,r,h,sig,dW):
    return X + r*X*h + sig*X*dW

def milstein_step(X,r,h,sig, dW): 
    return X + r*X*h + sig*X*dW + 1/2 * (sig**2)*(dW**2 - h) 

def rk_step(X,r,h,sig,dW):
    X_hat = X + r*X*h + sig*X*np.sqrt(h) 
    return X + r*X*h + sig*X*dW + 1/(2*math.sqrt(h))*(dW**2 - h)*(sig*(X_hat -X)) 

def opre_gbm (l,N, X0, r, sig, T, M, payOff, integration_method = 'E', ant_variates = False, randn = np.random.randn): 
    
    match integration_method:
        case 'E': 
            step = euler_step
        case 'M': 
            step = milstein_step
        case 'RK':
            step = rk_step 
        case _:
            raise ValueError("Please choose an appropiate integration method")
    
    
    nf = M**l 
    hf = T/nf 
    
    nc = max(nf/M, 1) 
    hc = T/nc 
    
    sums = np.zeros(6) 

    for N1 in range(1, N+1, 10000): 
        N2 = min(10000, N-N1 +1) 
        
        Xf = X0*np.ones(N2) 
        Xc = X0*np.ones(N2) 

        if ant_variates: 
            ant_Xf = X0*np.ones(N2) 
            ant_Xc = np.ones(N2) 
        
        
        if l== 0: 
            dWf = math.sqrt(hf)*randn(1,N2) 
            Xf[:] = step(Xf, r, hf, sig, dWf) 
            
            if ant_variates: 
                ant_Xf[:] = step(ant_Xf, r, hf, sig, -dWf)
            
        else: 
            for n in range(int(nc)): 
                dWc = np.zeros((1,N2)) 
                
                for m in range(M): 
                    dWf = math.sqrt(hf)*randn(1,N2) 
                    dWc[:] = dWc + dWf 
                    Xf[:]  = step(Xf, r, hf, sig, dWf)  

                    if ant_variates: 
                        ant_Xf[:] = step(ant_Xf, r, hf, sig, -dWf)
                
                Xc[:] = step(Xc, r, hc, sig, dWc)
                
                if ant_variates: 
                    ant_Xc[:] = step(ant_Xc, r, hc, sig, -dWc)
        
        Pf = np.vectorize(payOff) (Xf)
        Pc = np.vectorize(payOff) (Xf)

        dP = np.exp(-r*T)*(Pf -Pc)        
        Pf = np.exp(-r*T)*Pf


        if ant_variates: 
            ant_Pf = np.vectorize(payOff)(ant_Xf)
            ant_Pc = np.vectorize(payOff)(ant_Xc) 

            ant_dP = np.exp(-r*T)*(ant_Pf - ant_Pc) 
            ant_Pf = np.exp(-r*T)*Pf

        if l == 0: 
            Pc = 0 

        if ant_variates:
          sums += 1/2*np.array([(np.sum(dP) + np.sum(ant_dP)), 
                          np.sum(dP**2)+ np.sum(ant_dP**2), 
                          np.sum(dP**3)+ np.sum(ant_dP**3), 
                          np.sum(dP**4)+ np.sum(ant_dP**4) , 
                          np.sum(Pf) + np.sum(ant_Pf), 
                          np.sum(Pf**2) + np.sum(ant_Pf**2)])
        else:
          sums += np.array([np.sum(dP), 
                          np.sum(dP**2), 
                          np.sum(dP**3), 
                          np.sum(dP**4), 
                          np.sum(Pf), 
                          np.sum(Pf**2)])

    cost = N*nf
    return (np.array(sums), cost) 

def europeanEstimation(X0, K, r, sig, T, l, N, put_or_call, integration_method = 'E', ant_variates = False, randn = np.random.randn):

  if put_or_call == 'C': 
    def payOff(S, K):
      return np.maximum(S-K,0) 
      
  elif put_or_call == 'P':
    def payOff(S,K):
      return np.maximum(K-S, 0) 

return (l,N, X0, r, sig, T, M, payOff, integration_method = 'E', ant_variates = False, randn = np.random.randn)
