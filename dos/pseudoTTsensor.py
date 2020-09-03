import numpy as np
import logging
import queue

logging.basicConfig()

class pseudoTTsensor:
    def __init__(self):
        self.logger = logging.getLogger(name='pseudoTTsensor')
        self.logger.setLevel(logging.INFO)
        # Load M1/M2 to segment tip-tilt transformation matrix
        self.D_seg_tt = np.load('Telescope/linear_jitter.npz')['D_seg_tt']
        # Initialize averaging counter
        self.count = 0

        self.seg_tt = np.zeros((14,1))
        self.__yout = np.zeros_like(self.seg_tt)

        queueSize = 12
        #Instantiate TT queue
        self.TTqueue = queue.Queue(queueSize)
        #Initialize TT queue
        for _ in range(queueSize):
            self.TTqueue.put(np.zeros_like(self.seg_tt))

    def init(self):
        pass

    def update(self,u):
        self.logger.debug(f"u: {u.shape}")
        # Average tip-tilt delayed measurement
        self.seg_tt += self.TTqueue.get()
        #print(f"seg_tt: {np.array_str(self.seg_tt*1e6,precision=1,suppress_small=True)}")          
        self.count += 1
        # Save new TT sample into the 
        self.TTqueue.put(self.D_seg_tt @ u.T)

    def output(self):
        #print(np.array_str(self.seg_tt/self.count ,precision=1,suppress_small=True))
        self.__yout[...] = self.seg_tt/self.count
        self.seg_tt[...] = 0.0
        self.count = 0
        return np.atleast_2d(self.__yout.T.ravel())
