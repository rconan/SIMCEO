import numpy as np
import scipy.sparse as sparse
import osqp     # OSQP solver
import logging
import dos.tools as tools
from scipy.linalg import block_diag
logging.basicConfig()
#import yaml

class SHAcO_debug:
    def __init__(self,D,W2,n_bm,wfsMask,A,B,verbose=logging.INFO,**kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(verbose)

        self.M = tools.build_AcO_Rec(D,
                                  n_bm=n_bm, W2=W2, rec_alg='RLS',#TSVD
                                  wfsMask=wfsMask)
        self.D = block_diag(*[Dseg[:,:12+n_bm] for Dseg in D])
        print('Dimension of Dsh:',self.D.shape)
        self.W2, self.n_bm, self.wfsMask = W2, n_bm, wfsMask

        self.__u = np.zeros(0)
        self.__xpast = np.zeros(0)

        self.logger.debug('Initializing!')

        # Plant model dimensions
        self.nx = A.shape[0]
        self.nu = B.shape[1]

    def init(self):
        self.__u = np.zeros(self.nu)
        self.__xpast = np.zeros(self.nx)

    def update(self,y_sh):

        # AcO state reconstructor
        y_sh = y_sh.ravel()
        y = np.concatenate((y_sh,self.__u), axis=0)
        x = self.M.dot(y)

        y_valid = np.hstack([*[y_sh[MaskSeg] for MaskSeg in self.wfsMask]])
        # xwoS7Rz = np.delete(x,[42,84])
        J1 = np.linalg.norm(y_valid - self.D.dot(x))**2
        delta = x-self.__u
        J3 = delta.T.dot(np.kron(np.eye(7),self.W2)).dot(delta)
        if(J3):
            print('-> J1:%0.8f, J3:%0.8f, ratio:%0.8f' %(J1,J3,J1/J3))

        # Update controller output
        self.__u = self.__u -0.15*x
        # Update past state
        self.__xpast = x 
        self.logger.debug('x: %s',self.__xpast)
        self.logger.debug('u: %s',self.__u)

    def output(self):
        return np.atleast_2d(self.__xpast)
        

# Function used to test empty constraint vectors
def empty(value):
    try:
        value = np.array(value)
    except ValueError:
        pass
    if value.size:
        return False
    else:
        return True
