from scipy import signal
import pickle
from .mpc_driver import MPC
from .MountController import Mount
import numpy as np
class System:
    def __init__(self,**kwargs):
        if 'transfer function' in kwargs:
            self.system = signal.dlti(kwargs['transfer function']['num'],
                                      kwargs['transfer function']['denom'])
        elif 'zeros poles gain' in kwargs:
            self.system = signal.dlti(kwargs['transfer function']['zeros'],
                                      Kwargs['transfer function']['poles'],
                                      kwargs['transfer function']['gain'])
        else:
            raise Exception("System should be of the type "+\
                            "'transfer function' or 'zeros poles gains'")
        self.__xout = np.zeros(0)
        self.__yout = np.zeros(0)

    def init(self):
        self.system = self.system._as_ss()
        self.__xout = np.zeros((1,self.system.A.shape[0]))
        self.__yout = np.zeros((1, self.system.C.shape[0]))

    def update(self,u):
        self.__yout = np.dot(self.system.C, self.__xout) + np.dot(self.system.D, u)
        self.__xout = np.dot(self.system.A, self.__xout) + np.dot(self.system.B, u)

    def output(self):
        return self.__yout
