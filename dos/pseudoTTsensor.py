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

        try:
            self.queueSize = kwargs['TTdelay']
        except:
            self.queueSize = 12
        self.logger.info("TT sensor delay: %d", self.queueSize)

        try:
            self.reordered_in = kwargs['reordered_input']
        except:
            self.reordered_in = False
        if self.reordered_in:
            QT = np.kron(np.eye(7),np.vstack((np.eye(3), np.zeros((3,3)) )))
            QR = np.kron(np.eye(7),np.vstack((np.zeros((3,3)), np.eye(3) )))
            # Return from RT ordering to S1-RxyzTxyz....S7-RxyzTxyz
            self.D_seg_tt = self.D_seg_tt.dot(np.kron(np.eye(2), np.hstack((QT,QR))))
        self.logger.info("Pseudo TT sensor model M1/M2 reordering: %s", self.reordered_in)

        #Instantiate TT queue
        self.TTqueue = queue.Queue(self.queueSize)
        #Initialize TT queue
        for _ in range(self.queueSize):
            self.TTqueue.put(np.zeros_like(self.seg_tt))

    def init(self):
        pass

    def update(self,u):
        self.logger.debug(f"u: {u.shape}")
        if(self.queueSize > 0):
            # Integrate tip-tilt delayed measurement
            self.seg_tt += self.TTqueue.get()
            # Save new TT sample into the 
            self.TTqueue.put(self.D_seg_tt @ u.T)
        else:
            # Integrate tip-tilt measurement (no sensor delay)
            self.seg_tt += self.D_seg_tt @ u.T
        #print(f"seg_tt: {np.array_str(self.seg_tt*1e6,precision=1,suppress_small=True)}")          
        self.count += 1
        

    def output(self):
        #print(np.array_str(self.seg_tt/self.count ,precision=1,suppress_small=True))
        self.__yout[...] = self.seg_tt/self.count
        self.seg_tt[...] = 0.0
        self.count = 0
        return np.atleast_2d(self.__yout.T.ravel())
