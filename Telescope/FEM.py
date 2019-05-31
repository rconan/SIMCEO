import numpy as np
from scipy import sparse
from scipy.linalg import solve_lyapunov
import h5py

def readStateSpace(filename=None,ABC=None):
    if filename is not None:
        f = h5py.File(filename, 'r')
        AG=f['A']
        A = sparse.csr_matrix((np.array(AG['data']),AG['ir'],AG['jc']))
        B,C = [sparse.csr_matrix(np.array(f[x]).T) for x in list('BC')]
    if ABC is not None:
        A = sparse.csr_matrix(ABC[0])
        B = sparse.csr_matrix(ABC[1])
        C = sparse.csr_matrix(ABC[2])
    h = int(A.shape[0]/2)
    print('State space synopsis:')
    print(' * A shape: {0} ; Block diagonal test (0,{5:d},1,1): {1} {2} {3} {4}'.format(A.shape,
          A[:h,:h].diagonal().sum(),
          A[:h,h:].diagonal().sum(),
          A[h:,:h].diagonal().sum()/A[h:,:h].sum(),
          A[h:,h:].diagonal().sum()/A[h:,h:].sum(),int(A.shape[0]/2)))
    print(' * B shape:',B.shape)
    print(' * C shape:',C.shape)
    return (A,B,C,None)

def ss2fem(*args):
    A,B,C = args[:3]
    h = int(A.shape[0]/2)
    O = np.sqrt(-A[h:,:h].diagonal())
    Z= -0.5*A[h:,h:].diagonal()/O
    Phi = C[:,:-h]
    Phim = B[h:,:].T
    print('FEM synopsis:')
    print(' * # of modes: {}'.format(O.size))
    print(' * eigen frequencies range: [{0:.3f},{1:.3f}]Hz'.format(O[0]/2/np.pi,O[-1]/2/np.pi))
    print(' * damping ratio min-max: [{0:.1f},{1:.1f}]%'.format(Z.min()*1e2,Z.max()*1e2))
    print(' * $\Phi_m$ shape:',Phim.shape)
    print(' * $\Phi$ shape:',Phi.shape)
    return O,Z,Phim.toarray(),Phi.toarray()

def freqrep(nu,Phi,Phim,O,Z,n_mode_max=None):
    s = 2*1j*np.pi*nu
    if n_mode_max is None:
        q = s**2 + 2*s*Z*O + O**2
        iQ = sparse.diags(1/q)
        return Phi@iQ@Phim.T
    else:
        q = s**2 + 2*s*Z[:n_mode_max]*O[:n_mode_max] + O[:n_mode_max]**2
        iQ = sparse.diags(1/q)
        return Phi[:,:n_mode_max]@iQ@Phim[:,:n_mode_max].T

class FEM:

    def __init__(self,state_space_filename=None,
                 state_space_ABC=None,
                 fem_inputs=None,fem_outputs=None):
        if state_space_filename is not None:
            self.O,self.Z,self.Phim,self.Phi = ss2fem(*readStateSpace(filename=state_space_filename))
        if state_space_ABC is not None:
            self.O,self.Z,self.Phim,self.Phi = ss2fem(*readStateSpace(ABC=state_space_ABC))
        self.N = self.O.size
        self.INPUTS = fem_inputs
        self.OUTPUTS = fem_outputs
        self.ins_idx = np.cumsum([x[1] for x in self.INPUTS])[:-1]
        self.outs_idx = np.cumsum([x[1] for x in self.OUTPUTS])[:-1]
        self.__Phim__ = {x:y for y,x in zip(np.split(self.Phim,self.ins_idx),[x[0] for x in self.INPUTS])}
        self.__Phi__ = {x:y for y,x in zip(np.split(self.Phi,self.outs_idx),[x[0] for x in self.OUTPUTS])}

    def state_space(self):
        s = self.N,self.N
        A = sparse.bmat( [[sparse.coo_matrix(s,dtype=np.float),sparse.eye(self.N,dtype=np.float)],
                          [sparse.diags(-self.O**2),sparse.diags(-2*self.O*self.Z)]],
                         format='dia')
        B = sparse.bmat( [[sparse.coo_matrix((self.N,self.Phim.shape[0]),dtype=np.float)],
                                [self.Phim.T]],format='bsr')
        C = sparse.bmat( [[self.Phi,sparse.coo_matrix((self.Phi.shape[0],self.N),dtype=np.float)]],format='bsr')
        return A,B,C

    def __call__(self,inputs,outputs):
        u = np.vstack(inputs.values())
        _Phim_ = np.vstack([self.__Phim__[x] for x in inputs])
        _Phi_ = np.vstack([self.__Phi__[x] for x in outputs])
        iO2 = sparse.diags(1/self.O**2)
        y = _Phi_@iO2@_Phim_.T@u
        idx = np.cumsum([self.__Phi__[x].shape[0] for x in outputs])[:-1]
        return {x:y for x,y in zip(outputs,np.vsplit(y,idx))}

    def G(self,nu,inputs,outputs):
        _Phim_ = np.vstack([self.__Phim__[x] for x in inputs])
        _Phi_ = np.vstack([self.__Phi__[x] for x in outputs])
        G = np.zeros((nu.size,_Phi_.shape[0],_Phim_.shape[0]),dtype=np.complex)
        for k in range(nu.size):
            G[k,...] = freqrep(nu[k],_Phi_,_Phim_,self.O,self.Z,n_mode_max=None)
        return G

    def grammian(self):
        X2 = np.zeros((self.N,self.N))
        X3 = X2
        c = np.atleast_2d(0.25/(self.O*self.Z)).T
        print(c.shape)
        X4 = c*(self.Phim.T@self.Phim)
        c[:,0] = 1/self.O**2
        X1 = c*X4
        W = np.block([[X1,X2],[X3,X4]])
        return W

    def grammian2(self):
        A,B,C = self.state_space()
        Q = - B@B.T
        W = solve_lyapunov(A.toarray(),Q.toarray())
        return W

