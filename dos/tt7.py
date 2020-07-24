import numpy as np
import logging

logging.basicConfig()

class TT7:
    def __init__(self):
        self.logger = logging.getLogger(name='TT7')
        self.logger.setLevel(logging.INFO)
        self.P = np.vstack([-np.eye(3),np.eye(3)])
        self.D_seg_tt = np.load('linear_jitter.npz')['D_seg_tt']
        D = np.load('D_TzRxRy.npz')['D_TzRxRy']
        self.M = np.linalg.pinv(D)
        D_FSM = np.load('D_FSM.npz')['D_FSM']
        self.M_FSM = np.linalg.inv(D_FSM)
        self.seg_tt = np.zeros((14,1))
        self.__yout = np.zeros((6,7))
        self.count = 0

    def init(self):
        pass

    def update(self,u):
        self.logger.debug(f"u: {u.shape}")
        self.seg_tt += self.D_seg_tt@u.T
        #print(f"seg_tt: {np.array_str(self.seg_tt*1e6,precision=1,suppress_small=True)}")          
        self.count+=1

    def output(self):
        M2_RxRy = self.M_FSM@self.seg_tt/self.count
        #print(np.array_str(M2_RxRy*1e6,precision=1,suppress_small=True))
        TzRxRy = np.insert(M2_RxRy,np.arange(0,14,2),0,axis=0)
        ue = np.reshape(self.M@TzRxRy,(7,3)).T
        p = self.P@ue
        #print(np.array_str(p ,precision=1,suppress_small=True))
        self.__yout[...] = p
        self.seg_tt[...] = 0.0
        self.count = 0
        return np.atleast_2d(self.__yout.T.ravel())

