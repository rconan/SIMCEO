import numpy as np
import logging
import queue

logging.basicConfig()

class pseudoWFsensor:
    def __init__(self):
        self.logger = logging.getLogger(name='pseudoWFsensor')
        self.logger.setLevel(logging.INFO)
        # Load M1/M2 to WFS slopes transformation matrix
        self.RacoDwfs = np.load('Telescope/lom_aco.npz')['R_times_D']
        
        # Initialize averaging counter
        self.count = 0

        self.c_hat = np.zeros((self.RacoDwfs.shape[0],1))
        self.__yout = np.zeros_like(self.c_hat)

    def init(self):
        pass

    def update(self,u):
        self.logger.debug(f"u: {u.shape}")
        
        # Integrate tip-tilt measurement (no sensor delay)
        self.c_hat += self.RacoDwfs @ u.T
        
        #print(f"Residual c: {np.array_str(self.c_hat, precision=2, suppress_small=True)}")          
        self.count += 1
        

    def output(self):
        #print(np.array_str(self.c_hat/self.count, precision=1, suppress_small=True))
        self.__yout[...] = self.c_hat/self.count
        self.c_hat[...] = 0.0
        self.count = 0
        return np.atleast_2d(self.__yout.T.ravel())
