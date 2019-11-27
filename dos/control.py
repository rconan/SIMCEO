from scipy import signal
import pickle
from .mpc_driver import MPC
from .sh_aco_driver import SHAcO
from .sh_aco_debug_driver import SHAcO_debug
from .MountController import Mount
from .source import Source
import numpy as np
class System:
    def __init__(self,**kwargs):
        self.system = signal.dlti(*tuple(kwargs['parameters'].values()))
        self.__xout = np.zeros(0)
        self.__yout = np.zeros(0)

    def init(self):
        self.system = self.system._as_ss()
        self.__xout = np.zeros((self.system.A.shape[1],1))
        self.__yout = np.zeros((self.system.C.shape[0],1))

    def update(self,u):
        try:
            self.__yout = self.system.C @ self.__xout + self.system.D @ u
            self.__xout = self.system.A @ self.__xout + self.system.B @ u
        except ValueError:
            u = u.reshape(self.system.B.shape[1],1)
            self.__yout = self.system.C @ self.__xout + self.system.D @ u
            self.__xout = self.system.A @ self.__xout + self.system.B @ u

    def output(self):
        return np.atleast_2d(self.__yout.ravel())
