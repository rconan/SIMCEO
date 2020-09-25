import numpy as np
import logging
import queue

logging.basicConfig()

class pseudoWFsensor:
    def __init__(self):
        self.logger = logging.getLogger(name='pseudoWFsensor')
        self.logger.setLevel(logging.INFO)
        # Load M1/M2 to WFS slopes transformation matrix
        self.Dwfs = np.load('Telescope/lom_aco.npz')['Dwfs']
        
        # Initialize averaging counter
        self.count = 0

        self.wfs = np.zeros((self.Dwfs.shape[0],1))
        self.__yout = np.zeros_like(self.wfs)

    def init(self):
        pass

    def update(self,u):
        self.logger.debug(f"u: {u.shape}")
        
        # Integrate tip-tilt measurement (no sensor delay)
        self.wfs += self.Dwfs @ u.T
        
        #print(f"WFS: {np.array_str(self.wfs,precision=2,suppress_small=True)}")          
        self.count += 1
        

    def output(self):
        #print(np.array_str(self.seg_tt/self.count ,precision=1,suppress_small=True))
        self.__yout[...] = self.wfs/self.count
        self.wfs[...] = 0.0
        self.count = 0
        return np.atleast_2d(self.__yout.T.ravel())
