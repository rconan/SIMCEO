import numpy as np
import scipy.sparse as sparse
import dos.tools as tools
from scipy.linalg import block_diag
import osqp     # OSQP solver
import logging
logging.basicConfig()
import os.path
#import yaml

class SHAcO:
    def __init__(self,D,n_bm,W2,W3,K,rho_2,rho_3,wfsMask,A,B,Q,R,npred,umin,umax,**kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)

        #self.M = tools.build_AcO_Rec(D,n_bm=n_bm, W3=W3, rec_alg='RLS', wfsMask=wfsMask)
        #self.Dpart = D                          
        
        self.D = D
        self.DwS7Rz = np.insert(D,[((12+n_bm)*6)+5,((12+n_bm)*6)+10],0,axis=1)
        self.M = tools.build_RLS_RecM(D, W2, W3, K, rho_2, rho_3, insM1M2S7Rz=True)
        #self.M = tools.build_TSVD_RecM(D, n_r=12, insM1M2S7Rz=True)

        self.W2, self.W3, self.k = W2, W3, K
        self.rho_2, self.rho_3 = rho_2, rho_3 
        self.n_bm, self.wfsMask = n_bm, wfsMask

        self.npred = npred
        self.umin = umin
        self.umax = umax

        try:
            self._Tu = kwargs['_Tu']
        except:
            self._Tu = np.eye(B.shape[1])
            
        if 'J1_J3_ratio' in kwargs.keys():
            self.J1_J3_ratio = kwargs['J1_J3_ratio']
        else:
            self.J1_J3_ratio = 10

        self.__u = np.zeros(0)
        self.__xpast = np.zeros(0)
        
        self.logger.debug('Initializing!')

        # Plant model dimensions
        C = np.eye(A.shape[0])
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        self.ny = C.shape[0]

        # Incremental (augmented) model matrices (Aa,Ba,Ca)
        Aa = np.concatenate((
            np.concatenate((A,np.zeros((self.nx,self.ny))),axis=1),
            np.concatenate((C,np.eye(self.ny)),axis=1)),axis=0)
        Ba = np.concatenate((B,np.zeros((self.ny,self.nu))),axis=0)
        Ca = np.concatenate((C,np.eye(self.ny)),axis=1)
        # Augmented state dimension
        nxa = Aa.shape[0]

        # MPC model matrices
        # Compute Psi matrix: free response effect on the future outputs
        Phi = np.zeros((self.ny*self.npred,nxa)) #,Do try Float to speed up?
        for k in range(self.npred):
            krowInit, krowEnd = k * self.ny, (k+1)*self.ny
            # row indices
            Phi[krowInit:krowEnd, :] = np.dot(Ca,np.linalg.matrix_power(Aa,k+1))

        # Compute Gamma matrix: input effects on future outputs
        Gamma = np.zeros((self.ny*self.npred,self.nu*self.npred))
        # Fill 1st column block of Gamma
        for k in range(self.npred):
            krowInit, krowEnd = k * self.ny, (k+1)*self.ny
            Gamma[krowInit:krowEnd, 0:self.nu] = np.dot(Ca,np.dot(np.linalg.matrix_power(Aa,k),Ba))
        # Remaining column blocks    
        for k in range(1,self.npred):
            kGammarowInit, kcolInit, kcolEnd = k*self.ny, k*self.nu, (k+1)*self.nu
            Gamma[:, kcolInit:kcolEnd] = np.concatenate(
                (np.zeros((kGammarowInit,self.nu)), Gamma[kGammarowInit:,0:self.nu]), axis=0)

        # QP problem quadratic term
        Qkron = sparse.kron(sparse.eye(self.npred),Q)
        Rkron = sparse.kron(sparse.eye(self.npred),R)
        Gamma = sparse.csc_matrix(Gamma)
        P = sparse.csc_matrix((Gamma.T.dot(Qkron).dot(Gamma)) + Rkron)

        # Inequality constraint matrices

        # Test if there are absolute bound constraints
        if(empty(umin) and not empty (umax)):
            umin = -np.inf*np.ones(self.nu)
        elif (not empty(umin) and empty (umax)):
            umax = np.inf*np.ones(self.nu)
        Umin = np.kron(np.ones(self.npred),umin)
        Umax = np.kron(np.ones(self.npred),umax)

        if (empty(umin) and empty(umax)):
            mpcAbsConstr = False
            c = np.array([])
            Ain = sparse.csc_matrix(np.eye(self.nu*self.npred))
            print('No constraints introduced to the MPC')
        else:
            mpcAbsConstr = True
            Ain = sparse.csc_matrix(np.kron(np.tril(np.ones(self.npred)),self._Tu))
            c = np.kron(np.ones((npred,1)), self._Tu)
            # Create inequality constraint matrix Ain: lb <= Ain*u <= ub
            print('Absolute constraints introduced to the MPC')
             
        # Create an OSQP object as global
        self.qpp = osqp.OSQP()
        # Setup QP problem
        self.qpp.setup(P=P, q=np.zeros(self.npred*self.nu), A=Ain,
            l=-np.inf*np.ones(Ain.shape[0]),
            u=np.inf*np.ones(Ain.shape[0]),
            eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 50*self.nu,
            verbose=False, warm_start=True)

        # MPC-QP data matrices
        self.mpc_qpdt = {
            'Phi':Phi, 'GammaT':Gamma.T, 'Qkron':Qkron,
            'Umin':Umin, 'Umax':Umax, 'c':c, 'AbsConstr':mpcAbsConstr,
        }

    def init(self):
        self.__u = np.zeros(self.nu)
        self.__xpast = np.zeros(self.nx)
        self.GammaTQkron = self.mpc_qpdt['GammaT'].dot(self.mpc_qpdt['Qkron']).toarray()
        self.MPC_flag = False
        if not (self.MPC_flag):
            self.invTu = np.linalg.pinv(self._Tu)

    def update(self,y_sh):

        # AcO state reconstructor
        y_sh = y_sh.ravel()
        y_valid = np.hstack([*[y_sh[MaskSeg.ravel()] for MaskSeg in self.wfsMask]])
        
        if(self.M.shape[1] == self.D.shape[0]):
            y = y_valid
            c_hat = self.M.dot(y)
            J3 = 0
        else:
            y = np.concatenate((y_valid,self.__u), axis=0)
            c_hat = self.M.dot(y)
            
            epsilon = y_valid - self.DwS7Rz.dot(c_hat)
            J1 = epsilon.T.dot(epsilon)
            delta = self.k*c_hat - self.__u
            delta = np.delete(delta,list(map(lambda x:x+(12+self.n_bm)*6, [6,12])),0)
            J3 = delta.T.dot(self.W3).dot(delta)

        # J3 is zero if delta is also zero...
        if(J3):
            if(self.rho_3):
                norm_s = np.linalg.norm(y_valid)
                print('-> J1:%0.3g, J3:%0.3g, ratio:%0.3g, ||s||:%0.3g' %(J1,J3,J1/(self.rho_3*J3),norm_s**2))
            else:
                print('-> J1:%0.3g, J3:%0.3g, ratio:%0.3g, rho_3:-X-' %(J1,J3,J1/J3))

            # Update J3 weight
            self.rho_3 = (J1/(self.J1_J3_ratio*J3))             
            
            self.M = tools.build_RLS_RecM(self.D, self.W2, self.W3, self.k,
                        self.rho_2, self.rho_3, insM1M2S7Rz=True)
            c_hat = self.M.dot(y)
        
            epsilon = y_valid - self.DwS7Rz.dot(c_hat)
            J1 = epsilon.T.dot(epsilon)
            #c_hatwoRz = np.delete(c_hat, list(map(lambda x:x+(12+self.n_bm)*6, [6,12])),0)
            delta = self.k*c_hat - self.__u
            delta = np.delete(delta, list(map(lambda x:x+(12+self.n_bm)*6, [6,12])),0)
            J3 = delta.T.dot(self.W3).dot(delta)
            print('+> J1:%0.3g J1/(rho_3*J3):%0.3g\n' %(J1,J1/(self.rho_3*J3)))

        if(self.MPC_flag):
            # AcO control using MPC algorithm

            # State feedback - update MPC internal variables         
            xa = np.hstack([c_hat-self.__xpast, self.__xpast])
            # QP linear term - demands state feedback    
            q = np.dot(self.GammaTQkron, np.dot(self.mpc_qpdt['Phi'],xa))

            # Update QP variables
            if(self.mpc_qpdt['AbsConstr']):
                # QP input constraints
                lb = self.mpc_qpdt['Umin']-self.mpc_qpdt['c'].dot(self.__u)
                ub = self.mpc_qpdt['Umax']-self.mpc_qpdt['c'].dot(self.__u)
                self.qpp.update(q=q, l=lb, u=ub)
            else:
                self.qpp.update(q=q)
                
            # Solve QP
            U = self.qpp.solve() 
            if self.mpc_qpdt['AbsConstr']:
                if(any(U.x[:self.nu]<0.98*lb[:self.nu]) or any(U.x[:self.nu]>1.02*ub[:self.nu])):
                    self.logger.warning('Constraint violation, check tolerance settings!')

            # Check solver status
            if U.info.status != 'solved':
                print(U.info.status)
                self.logger.warning('Infeasible QP problem!!!')    
            # Update controller output
            self.__u = self.__u + U.x[:self.nu]
        else:
            # Integral controller
            self.__u = self.__u -self.k*c_hat

            if not (empty(self.umin) and empty(self.umax)):
                clip_iter, clip_tol = 0, 1.02
                while (clip_iter<30) and (
                            any(self._Tu.dot(self.__u) > clip_tol*self.umax) or 
                            any(self._Tu.dot(self.__u) < clip_tol*self.umin)):
                    clip_iter = clip_iter + 1        
                    self.__u = self.invTu.dot(np.clip(self._Tu.dot(self.__u), self.umin, self.umax))
                if(clip_iter):    
                    print('Number of clipping iterations:%d'%clip_iter)

        # Update past state
        self.__xpast = c_hat 
        self.logger.debug('u: %s',self.__u)

    def output(self):
        return np.atleast_2d(self.__u)

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
