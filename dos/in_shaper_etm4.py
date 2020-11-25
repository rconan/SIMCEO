import numpy as np
import logging
import queue

#logging.basicConfig()

class in_shaper_ETM4:
    def __init__(self,n_r,res_freq,zeta,Ts,**kwargs):
 #       self.logger = logging.getLogger(name='4th order Equal shaping-time and magnitude (ETM4)')
 #       self.logger.setLevel(logging.INFO)
        
        self.r_f = np.zeros(n_r)

        try:
            m = kwargs['m']
        except:
            m = 0.97

        K_ = np.exp(zeta*np.pi/np.sqrt(1-zeta**2))
        I1 = ((1+m)*(K_**2))/((K_**2) + (1+m)*(K_**(4/3)+K_**(2/3)) + m)
        self.i1_A = [I1/(1+m), I1/(K_**(2/3)), I1/(K_**(4/3)), m*I1/((1+m)*K_**2)]
        i1_t = np.array([0, (1/3)/res_freq, (2/3)/res_freq, 1/res_freq])

        delays = [0]+[np.int(np.rint(x)) for x in np.diff(i1_t/Ts)]
        [print('A:%.3g,t:%d samples' %(x,y)) for x,y in zip(self.i1_A,delays)], print('.')

#        self.logger.info("ETM: %.3g / %.3g /%.3g", (res_freq,zeta,m))

        # Instantiate queues - The delays [1,2,3] should be equal, anyway ...
        self.I1queue1 = queue.Queue(delays[1])
        self.I1queue2 = queue.Queue(delays[2])
        self.I1queue3 = queue.Queue(delays[3])
        #Initialize TT queue
        for _ in range(delays[1]):
            self.I1queue1.put(np.zeros_like(self.r_f))
        for _ in range(delays[2]):    
            self.I1queue2.put(np.zeros_like(self.r_f))
        for _ in range(delays[3]):    
            self.I1queue3.put(np.zeros_like(self.r_f))

    def init(self):
        pass

    def update(self,r):
        x3 = self.I1queue3.get()
        x2 = self.I1queue2.get()
        x1 = self.I1queue1.get()
        self.r_f = self.i1_A[0]*r + self.i1_A[1]*x1 + self.i1_A[2]*x2 + self.i1_A[3]*x3 
        self.I1queue3.put(x2)
        self.I1queue2.put(x1)
        self.I1queue1.put(r)
        
 #       self.logger.debug(f"r_f: {self.r_f.shape}")
        

    def output(self):
        return np.atleast_2d(self.r_f.T.ravel())
