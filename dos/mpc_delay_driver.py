import numpy as np
import scipy.sparse as sparse
import osqp     # OSQP solver
import logging
logging.basicConfig()
#import yaml

class MPC_1delay:
    def __init__(self,A,B,Q,R,npred,dumin,dumax,umin,umax,verbose=logging.INFO,**kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(verbose)

        self.__u = np.zeros(0)
        self._ypast = np.zeros(0)
        self.__ypast = np.zeros(0)
        if(npred < 2):
            print('For plants subject to input delay n_pred > 1.',
                'Setting prediction horizon to 2!')
            self.npred = 2
        else:    
            self.npred = npred

        self.logger.debug('Initializing!')
        if self.logger.getEffectiveLevel() > 20:
            verboseFlag = True #Verbose mode just from INFO level (or beyond)
        else:
            verboseFlag = False

        # Plant model dimensions
        C = np.hstack([np.zeros((A.shape[0],A.shape[0])),np.eye(A.shape[0])])
        A = np.hstack([ np.vstack([A,np.eye(A.shape[0])]),
            np.zeros((C.shape[1],A.shape[0])) ])
        B = np.vstack([B,np.zeros_like(B)])        
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

        # Constraint matrices
        # Test if there are incremental bound constraints
        if(empty(dumin) and not empty (dumax)):
            dumin = -np.inf*np.ones(self.nu)
        elif (not empty(dumin) and empty (dumax)):
            dumax = np.inf*np.ones(self.nu)
        Dumin = np.kron(np.ones(self.npred),dumin)
        Dumax = np.kron(np.ones(self.npred),dumax)
        if (empty(dumin) and empty (dumax)):
            mpcIncConstr = False
        else:
            mpcIncConstr = True

        # Test if there are absolute bound constraints
        if(empty(umin) and not empty (umax)):
            umin = -np.inf*np.ones(self.nu)
        elif (not empty(umin) and empty (umax)):
            umax = np.inf*np.ones(self.nu)
        Umin = np.kron(np.ones(self.npred),umin)
        Umax = np.kron(np.ones(self.npred),umax)
        if (empty(umin) and empty (umax)):
            mpcAbsConstr = False
            c = np.array([])
        else:
            mpcAbsConstr = True
            S = np.kron(np.tril(np.ones(self.npred)),np.eye(self.nu))
            c = np.kron(np.ones((self.npred,1)), np.eye(self.nu))
        # Create inequality constraint matrix Ain: lb <= Ain*u <= ub
        if mpcIncConstr and mpcAbsConstr:
            Ain = sparse.csc_matrix(np.concatenate(
                (np.eye(self.nu*self.npred),S), axis=0))
            print('Incremental and absolute constraints introduced to MPC')
        elif not mpcIncConstr and mpcAbsConstr:
            Ain = sparse.csc_matrix(S)
            print('Absolute constraints introduced to MPC')
        elif mpcIncConstr and not mpcAbsConstr:
            Ain = sparse.csc_matrix(np.eye(self.nu*self.npred))
            print('Incremental constraints introduced to MPC')
        
        # Create an OSQP object as global
        self.qpp = osqp.OSQP()
        # Setup QP problem
        if mpcIncConstr or mpcAbsConstr:
            self.qpp.setup(P=P, q=np.zeros(self.npred*self.nu), A=Ain,
                l=np.zeros(Ain.shape[0]), u=np.zeros(Ain.shape[0]), 
                eps_abs = 1.0e-10, eps_rel = 1.0e-10,
                verbose=verboseFlag, warm_start=True)
        else:
            self.qpp.setup(P=P, q=np.zeros(self.npred*self.nu),
                A=sparse.csc_matrix(np.eye(self.nu*self.npred)),
                l=-np.inf*np.ones(self.nu*self.npred),
                u=np.inf*np.ones(self.nu*self.npred),
                eps_abs = 1.0e-10, eps_rel = 1.0e-10,
                verbose=verboseFlag, warm_start=True)

        # MPC-QP data matrices
        self.mpc_qpdt = {
            'Phi':Phi, 'GammaT':Gamma.T, 'Qkron':Qkron,
            'Dumin':Dumin, 'Dumax':Dumax, 'Umin':Umin, 'Umax':Umax,
            'c':c, 'IncConstr':mpcIncConstr, 'AbsConstr':mpcAbsConstr,
        }


    def init(self):
        self.__u = np.zeros(self.nu)
        self._ypast = np.zeros(self.ny)
        self.__ypast = np.zeros_like(self._ypast)
        self.GammaTQkron = self.mpc_qpdt['GammaT'].dot(self.mpc_qpdt['Qkron']).toarray()


    def update(self,x):
        c_hat = x.ravel()
        # State feedback - update MPC internal variables 
        xa = np.hstack([self._ypast - self.__ypast, c_hat - self._ypast, self._ypast])
        # QP linear term - demands state feedback    
        q = np.dot(self.GammaTQkron, np.dot(self.mpc_qpdt['Phi'],xa))

        # QP input constraints
        if(self.mpc_qpdt['IncConstr'] and self.mpc_qpdt['AbsConstr']):
            lb = np.concatenate((self.mpc_qpdt['Dumin'],
                self.mpc_qpdt['Umin']-self.mpc_qpdt['c'].dot(self.__u)), axis=0)
            ub = np.concatenate((self.mpc_qpdt['Dumax'],
                self.mpc_qpdt['Umax']-self.mpc_qpdt['c'].dot(self.__u)), axis=0)
        elif(not self.mpc_qpdt['IncConstr'] and self.mpc_qpdt['AbsConstr']):
            lb = self.mpc_qpdt['Umin']-self.mpc_qpdt['c'].dot(self.__u)
            ub = self.mpc_qpdt['Umax']-self.mpc_qpdt['c'].dot(self.__u)
        elif(self.mpc_qpdt['IncConstr'] and not self.mpc_qpdt['AbsConstr']):
            lb = self.mpc_qpdt['Dumin']
            ub = self.mpc_qpdt['Dumax']
        else:
            pass
        # Update QP variables
        if self.mpc_qpdt['IncConstr'] or self.mpc_qpdt['AbsConstr']:
            self.qpp.update(q=q, l=lb, u=ub)
        else:
            self.qpp.update(q=q)
        # Solve QP
        U = self.qpp.solve() 
        if any((U.x[:self.nu]<lb[:self.nu])|(U.x[:self.nu]>ub[:self.nu])):
            print('Constraint violation, check tolerance settings!')

        # Check solver status
        if U.info.status != 'solved':
            print(U.info.status)
            self.logger.error('Infeasible QP problem!!!')    
        # Update controller output
        self.__u = self.__u + U.x[:self.nu]
    
        # Integral controller (For debug only)
        #self.__u = self.__u -0.15*x
        #self.__u = np.clip(self.__u, a_min = self.umin, a_max = self.umax)

        # Update past state
        self.__ypast = self._ypast
        self._ypast = c_hat 
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
