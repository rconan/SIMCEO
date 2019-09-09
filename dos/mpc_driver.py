import numpy as np
import scipy.sparse as sparse
import osqp     # OSQP solver
import logging
logging.basicConfig()
#import yaml

class MPC:
    def __init__(self,A,B,Q,R,npred,dumin,dumax,umin,umax,verbose=logging.INFO,**kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(verbose)
        self.__u = np.zeros(0)
        self.__xpast = np.zeros(0)
        self.npred = npred

        self.logger.debug('Initializing!')

        # Plant model dimensions
        self.nx = A.shape[0]
        self.nu = B.shape[1]

        # Incremental (augmented) model matrices (Aa,Ba,Ca)
        Aa = np.concatenate((
            np.concatenate((A,np.zeros((self.nx,self.nx))),axis=1),
            np.concatenate((np.eye(self.nx),np.eye(self.nx)),axis=1)),axis=0)
        Ba = np.concatenate((B,np.zeros((self.nx,self.nu))),axis=0)
        Ca = np.concatenate((np.eye(self.nx),np.eye(self.nx)),axis=1)
        # Augmented state dimension
        nxa = Aa.shape[0]

        # MPC model matrices
        # Compute Psi matrix: free response effect on the future outputs
        Phi = np.zeros((self.nx*self.npred,nxa)) #,Do try Float to speed up?
        for k in range(self.npred):
            krowInit, krowEnd = k * self.nx, (k+1)*self.nx
            # row indices
            Phi[krowInit:krowEnd, :] = np.dot(Ca,np.linalg.matrix_power(Aa,k+1))

        # Compute Gamma matrix: input effects on future states
        Gamma = np.zeros((self.nx*self.npred,self.nu*self.npred))
        # Fill 1st column block of Gamma
        for k in range(self.npred):
            krowInit, krowEnd = k * self.nx, (k+1)*self.nx
            Gamma[krowInit:krowEnd, 0:self.nu] = np.dot(Ca,np.dot(np.linalg.matrix_power(Aa,k),Ba))
        # Remaining column blocks    
        for k in range(1,self.npred):
            kGammarowInit, kcolInit, kcolEnd = k*self.nx, k*self.nu, (k+1)*self.nu
            Gamma[:, kcolInit:kcolEnd] = np.concatenate(
                (np.zeros((kGammarowInit,self.nu)), Gamma[kGammarowInit:,0:self.nu]), axis=0)

        # QP problem quadratic term
        Qkron = sparse.kron(sparse.eye(self.npred),Q)
        Rkron = sparse.kron(sparse.eye(self.npred),R)
        P = sparse.csc_matrix((Gamma.T@Qkron@Gamma) + Rkron)

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
        elif not mpcIncConstr and mpcAbsConstr:
            Ain = sparse.csc_matrix(S)
        elif mpcIncConstr and not mpcAbsConstr:
            Ain = sparse.csc_matrix(np.eye(self.nu*self.npred))
        
        # Create an OSQP object as global
        self.qpp = osqp.OSQP()
        # Setup QP problem
        if self.logger.getEffectiveLevel() > 20:
            verboseFlag = True #Verbose mode just from INFO level (or beyond)
        else:
            verboseFlag = False
        if mpcIncConstr or mpcAbsConstr:
            self.qpp.setup(P, np.zeros(self.npred*self.nx), Ain,
                np.zeros(Ain.shape[0]), np.zeros(Ain.shape[0]),
                verbose=verboseFlag, warm_start=True)
        else:
            self.qpp.setup(P, np.zeros(self.npred*self.nx),
                sparse.csc_matrix(np.eye(self.nu*self.npred)),
                -np.inf*np.ones(self.nu*self.npred),
                np.inf*np.ones(self.nu*self.npred),
                verbose=verboseFlag, warm_start=True)

        # MPC-QP data matrices
        self.mpc_qpdt = {
            'Phi':Phi, 'GammaT':Gamma.T, 'Qkron':Qkron.toarray(),
            'Dumin':Dumin, 'Dumax':Dumax, 'Umin':Umin, 'Umax':Umax,
            'c':c, 'IncConstr':mpcIncConstr, 'AbsConstr':mpcAbsConstr
        }

    def init(self):
        self.__u = np.zeros(self.nu)
        self.__xpast = np.zeros(self.nx)

    def update(self,x):

        x = x.ravel()
        
        # State feedback - update MPC internal variables
        xa = np.concatenate((x-self.__xpast,self.__xpast), axis=0)

        # QP linear term - demands state feedback
        q = np.dot(np.dot(self.mpc_qpdt['GammaT'],self.mpc_qpdt['Qkron']),
                (np.dot(self.mpc_qpdt['Phi'],xa)))
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
        if self.mpc_qpdt['IncConstr'] or self.mpc_qpdt['IncConstr']:
            self.qpp.update(q, lb, ub)
        else:
            self.qpp.update(q=q) 
        # Solve QP
        U = self.qpp.solve()
        # Check solver status
        if U.info.status != 'solved':
            self.logger.error('Infeasible QP problem!!!')
        # Upsate controller output
        self.__u = self.__u + U.x[:self.nu]
        # Update past state
        self.__xpast = x 
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
