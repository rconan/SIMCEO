import os
import sys
import time

import numpy as np
from scipy import sparse
from scipy.linalg import solve_lyapunov
from scipy.sparse import linalg as slinalg

from multiprocessing import Manager, Process

import scipy.linalg as lalg
import scipy.io as spio
import h5py

import logging

import matplotlib.pyplot as plt

try:
    import cupy as cp
    import cupyx.scipy.sparse as cps
    CUDA_LIBRARY = True
except:
    print('No cupy GPU library available for CUDA processing.')
    CUDA_LIBRARY = False

logging.basicConfig()

def readStateSpace(filename=None,ABC=None):
    """
    Load the A, B and C matrix of a state space model.
    The matrix are either in a mat-file with A as a sparse matrix
    or in a list

    Parameters
    ----------
    filename : string
        The name of the mat-file
    ABC : list
        The list containing the A, B and C matrices

    Returns
    -------
    tuple of numpy array: the A, B,  C and D=None matrices
    """
    if filename is not None:
        try:
            f = h5py.File(filename, 'r')
            AG=f['A']
            A = sparse.csr_matrix((np.array(AG['data']),AG['ir'],AG['jc']))
            A = A.transpose()
            B,C = [sparse.csr_matrix(np.array(f[x]).T) for x in list('BC')]
        except:
            f = spio.loadmat(filename, variable_names=('A','B','C'))
            A = sparse.csr_matrix(f['A'])
            B = sparse.csr_matrix(f['B'])
            C = sparse.csr_matrix(f['C'])
            
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

def loadFEM2ndOrder(filename):
    """
    Load a second order model from a mat-file
    The mat-file contains the following variables:
     . eigenfrequencies
     . proportionalDampingVec
     . Phim
     . Phi
     . FEM_IO
    FEM_IO is a dictionary with the following keys:
     . 'inputs_name'
     . 'inputs_size'
     . 'outputs_name'
     . 'outputs_size'

    Parameters
    ----------
    filename : string
        The name of the mat-file

    Returns
    -------
    eigenfrequencies, proportionalDampingVec, Phim, Phim, fem_inputs and fem_outputs
    """
    print('LOADING {}'.format(filename))
    data = spio.loadmat(filename)
    fem_inputs=[(x[0][0],y[0]) for x,y in zip(data['FEM_IO']['inputs_name'][0,0],data['FEM_IO']['inputs_size'][0][0])]
    fem_outputs=[(x[0][0],y[0]) for x,y in zip(data['FEM_IO']['outputs_name'][0,0],data['FEM_IO']['outputs_size'][0][0])]
    var = ['eigenfrequencies','proportionalDampingVec','Phim','Phi']
    return tuple(data[x] for x in var),fem_inputs,fem_outputs

def ss2fem(*args):
    """
    Convert a FEM continuous state space model into second order form

    Parameters
    ----------
    A : ndarray
       The state space A matrix
    B : ndarray
       The state space B matrix
    C : ndarray
       The state space C matrix

    Returns
    -------
    O, Z, Phim, Phi: the eigen frequencies, the damping coefficients, 
    the inputs to modes and modes to outputs matrix transforms
    """
    A,B,C = args[:3]
    h = int(A.shape[0]/2)
    O = np.sqrt(-A[h:,:h].diagonal())
    Z = -0.5*A[h:,h:].diagonal()/O
    Phi = C[:,:-h]
    Phim = B[h:,:].T
    return O,Z,Phim.toarray(),Phi.toarray()

def freqrep(nu,Phi,Phim,O,Z,n_mode_max=None):
    """
    Computes the FEM transfer function

    Parameters
    ----------
    nu: ndarray
       The frequency vector in Hz
    Phi: ndarray
       The  modes to outputs matrix transform
    Phi: ndarray
       The inputs to modes matrix transform
    O: ndarray
        The eigen frequencies
    Z: ndarray
        The damping coefficients
    n_mode_max: int
        The number of modes, default: None (all the modes)
    """
    s = 2*1j*np.pi*nu
    if n_mode_max is None:
        q = s**2 + 2*s*Z*O + O**2
        iQ = sparse.diags(1/q)
        return Phi@iQ@Phim.T
    else:
        q = s**2 + 2*s*Z[:n_mode_max]*O[:n_mode_max] + O[:n_mode_max]**2
        iQ = sparse.diags(1/q)
        return Phi[:,:n_mode_max]@iQ@Phim[:,:n_mode_max].T


def save_variable_on_disk(info):
    '''
        Function that is parallelized to save process variables on disk,
        on simulation time.
    '''
    
    # Create variable directory
    if not os.path.exists(info['savepath']):
        os.makedirs(info['savepath'])
    var_name = info['savepath'] + info['varname'] + '.dat'
    # Create the memory map
    variable = np.memmap(var_name, dtype=np.float32, mode='w+', shape=info['shape'])
    print(' * Variable created...' )
    print('    |- keeping log of: ', info['shape'])
    print('    |- in:', var_name)
    # Create queue
    q = info['queue']
    while True: # start Deamon
        # remove one task from Queue
        ninfo = q.get()
        # check if is alive
        if ninfo['alive']:
            variable[:,ninfo['index']] = ninfo['data'][:]
        else:
            del variable  # flush and save output memory map
            break         # kill Deamon



class FEM:

    def __init__(self,verbose=logging.INFO,**kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(verbose)
        self.logger.info('Instantiate')
        if kwargs:
            self.Start(**kwargs)

    def Start(self, state_space_filename=None,
            second_order_filename=None,
            state_space_ABC=None,
            second_order=None,
            fem_inputs=None, fem_outputs=None,
            reorder_RBM=False,
            damping_ratio=None,
            output_support=None,
            log_states=False, log_path='./variables/', log_type=np.float32,
            fem_io_filename=None):
        self.logger.info('Start')
        
        if fem_io_filename is not None:
            data = spio.loadmat(fem_io_filename)
            fem_inputs=[(x[0][0],y[0]) for x,y in zip(data['FEM_IO']['inputs_name'][0,0],data['FEM_IO']['inputs_size'][0][0])]
            fem_outputs=[(x[0][0],y[0]) for x,y in zip(data['FEM_IO']['outputs_name'][0,0],data['FEM_IO']['outputs_size'][0][0])]
        if state_space_filename is not None:
            self.O,self.Z,self.Phim,self.Phi = ss2fem(*readStateSpace(filename=state_space_filename))
            self.model_file_name = state_space_filename
        if second_order_filename is not None:
            second_order,fem_inputs,fem_outputs = loadFEM2ndOrder(second_order_filename)
        if state_space_ABC is not None:
            self.O,self.Z,self.Phim,self.Phi = ss2fem(*readStateSpace(ABC=state_space_ABC))
        if second_order is not None:
            self.O,self.Z,self.Phim,self.Phi = second_order
            self.O *= 2*np.pi
            self.O = np.ravel(self.O)
            self.Z = np.ravel(self.Z)
        self.INPUTS = fem_inputs
        self.OUTPUTS = fem_outputs
        self.state = {'u':None,'y':None,'A':None,'B':None,'C':None,'D':None, 'x': None, 'step':0}
        
        self.__setprop__()
        
        if reorder_RBM:
            self.reorder_rbm()

        if damping_ratio is not None:
            try:
                self.Z = damping_ratio*np.ones_like(np.ravel(self.Z))
            except:
                self.logger.warning('Unable to change the model damping!!!')

        if output_support is not None:
            self.output_support = output_support

        self.log_states = log_states
        if self.log_states:
            self.stateLogInfo = {'path': log_path, 'vartype': log_type}

        self.info()
        return "FEM"


    def __setprop__(self):
        self.N = self.O.size
        c = np.cumsum([x[1] for x in self.INPUTS])
        self.N_INPUTS = c[-1]
        self.ins_idx = c[:-1]
        c = np.cumsum([x[1] for x in self.OUTPUTS])
        self.N_OUTPUTS = c[-1]
        self.outs_idx = c[:-1]
        self.__Phim__ = {x:y for y,x in zip(np.split(self.Phim,self.ins_idx),[x[0] for x in self.INPUTS])}
        self.__Phi__ = {x:y for y,x in zip(np.split(self.Phi,self.outs_idx),[x[0] for x in self.OUTPUTS])}
        m = self.O==0
        if np.any(m):
            self.hsv     = np.ones_like(self.O)*np.Inf
            self.H2_norm = np.zeros_like(self.O)
            m=~m
            self.hsv[m]     = 0.25 * np.sqrt(np.sum(self.Phim[:,m]**2,0)) * np.sqrt(np.sum(self.Phi[:,m]**2,0)) / self.O[m] / self.Z[m]
            self.H2_norm[m] = 2*self.hsv[m]*np.sqrt(self.O[m]*self.Z[m]/2/np.pi)
        else:
            self.hsv     = 0.25 * np.sqrt(np.sum(self.Phim**2,0)) * np.sqrt(np.sum(self.Phi**2,0)) / self.O / self.Z
            self.H2_norm = 2*self.hsv*np.sqrt(self.O*self.Z/2/np.pi)

    def reorder_rbm(self):
        p = np.zeros((3,42))
        p[:3,:3] =  np.eye(3)
        PT = np.vstack([np.roll(p,k,axis=1) for k in range(0,42,6)])
        PR = np.roll(PT,3,axis=1)
        Q = np.vstack([PT,PR])
        self.P = {'OSS_M1_lcl':Q,'MC_M2_lcl_6D':Q}
        for k in self.P:
            self.__Phi__[k] = self.P[k]@self.__Phi__[k]
            self.logger.info('Reordering %s'%k)
        # Update Phi after reordering -- __setprop__() overwrites __Phi__ !
        self.Phi = np.vstack([self.__Phi__[x[0]] for x in self.OUTPUTS])

    def info(self):
        self.logger.info('FEM synopsis:')
        self.logger.info(' * # of inputs: {}'.format(self.N_INPUTS))
        self.logger.info(' * # of outputs: {}'.format(self.N_OUTPUTS))
        self.logger.info(' * # of modes: {}'.format(self.O.size))
        self.logger.info(' * $\Phi_m$ shape: %s',self.Phim.shape)
        self.logger.info(' * $\Phi$ shape: %s',self.Phi.shape)
        self.logger.info(' * eigen frequencies range: [{0:.3f},{1:.3f}]Hz'.format(self.O.min()/2/np.pi,self.O.max()/2/np.pi))
        self.logger.info(' * damping ratio min-max: [{0:.1f},{1:.1f}]%'.format(self.Z.min()*1e2,self.Z.max()*1e2))
        self.logger.info(' * Hankel singular values min-max: [{0:g},{1:g}]'.format(self.hsv.min(),self.hsv.max()))
        self.logger.info(' * mode H2 norm min-max: [{0:g},{1:g}]'.format(self.H2_norm.min(),self.H2_norm.max()))
        self.logger.info(' * system H2 norm: {0:g}'.format(np.sqrt(np.sum(self.H2_norm**2))))

    def hsv_sort(self,start_idx=0):
        """
        Sort the eigen modes according the Hankel singular values
        """
        idx = np.argsort(self.hsv[start_idx:])[::-1]
        idx = np.hstack([np.arange(start_idx),idx+start_idx])
        self.hsv_idx = idx
        self.O = self.O[idx]
        self.Z = self.Z[idx]
        self.Phim = self.Phim[:,idx]
        self.Phi = self.Phi[:,idx]
        self.__setprop__()

    def O_sort(self,start_idx=0):
        """
        Sort the eigen modes according the eigen frequencies
        """
        idx = np.argsort(self.O[start_idx:])
        idx = np.hstack([np.arange(start_idx),idx+start_idx])
        self.O_idx = idx
        self.O = self.O[idx]
        self.Z = self.Z[idx]
        self.Phim = self.Phim[:,idx]
        self.Phi = self.Phi[:,idx]
        self.__setprop__()

    def c2s(self,a,b,dt):
        """
        Convert a continuous state space model in a discrete one

        Parameters:
        -----------
        a : ndarray
            The continuous state space A matrix
        b : ndarray
            The continuous state space B matrix
        dt: float
            The discrete model sampling time

        Returns:
        --------
        ad, bd : the discrete state space A and B matrices
        """
        em_upper = sparse.hstack((a, b))
        em_lower = sparse.hstack((sparse.csr_matrix((b.shape[1], a.shape[0])),
                                  sparse.csr_matrix((b.shape[1], b.shape[1]))))
        em = sparse.vstack((em_upper, em_lower)).tocsc()
        ms = slinalg.expm(dt * em)
        ms = ms[:a.shape[0], :]
        ad = ms[:, 0:a.shape[1]]
        bd = ms[:, a.shape[1]:]
        return ad.tocsr(),bd.toarray()

    def gpu_c2s(self,a,b,dt,ftype=np.float64):
        """
            Convert a continuous state space model in a discrete one
        using the second order modules iterativelly.

        Parameters:
        -----------
        a : ndarray
            The continuous state space A matrix
        b : ndarray
            The continuous state space B matrix
        dt: float
            The discrete model sampling time

        Returns:
        --------
        ad, bd : the discrete state space A and B matrices
        """

        ss = a.shape[0] // 2

        I_2 = a[ss:,:ss].diagonal()
        I_3 = a[ss:,ss:].diagonal()

        Ad = np.zeros(a.shape, dtype=ftype)
        Bd = np.zeros(b.shape, dtype=ftype)

        A_ = np.zeros((3,3))
        B_ = np.zeros((2,b.shape[1]))
        A_[0,1], A_[1,2] = 1, 1

        phiGamma = np.zeros((3,3))
        for k in range(ss):
            ki = ss + k
            A_[1,0], A_[1,1] = I_2[k], I_3[k]
            B_[0,:]  = b[k, :].toarray()
            B_[1,:]  = b[ki,:].toarray()
            phiGamma = lalg.expm(A_ * dt)

            Ad[k,k],  Ad[k,ki]  = phiGamma[0,0], phiGamma[0,1]
            Ad[ki,k], Ad[ki,ki] = phiGamma[1,0], phiGamma[1,1]
            Bd[k,:]  = phiGamma[0,2] * b[ki,:].toarray()
            Bd[ki,:] = phiGamma[1,2] * b[ki,:].toarray()
        
        return sparse.csr_matrix(Ad), Bd


    def state_space(self,dt=None,ftype=np.float64):
        """
        Build the state space model from the second order model, if dt is given returns the discrete state space
        otherwise returns return the continuous state space

        Parameters
        ----------
        dt: float
            The discrete model sampling time

        Returns:
        --------
        A,B,C : the state space A, B, and C matrices
        """

        s = self.N,self.N
        OZ = self.O*self.Z
        OZ[np.isnan(OZ)] = 0
        A = sparse.bmat( [[sparse.coo_matrix(s,dtype=ftype),sparse.eye(self.N,dtype=ftype)],
                          [sparse.diags(-self.O**2),sparse.diags(-2*OZ)]],
                         format='csr')
        B = sparse.bmat( [[sparse.coo_matrix((self.N,self.Phim.shape[0]),dtype=ftype)],
                                [self.Phim.T]],format='csr')
        C = sparse.bmat( [[self.Phi,sparse.coo_matrix((self.Phi.shape[0],self.N),dtype=ftype)]],format='csr')
        h = int(A.shape[0]/2)
        print('State space synopsis:')
        print(' * A shape: {0} ; Block diagonal test (0,{5:d},1,1): {1} {2} {3} {4}'.format(A.shape,
            A[:h,:h].diagonal().sum(),
            A[:h,h:].diagonal().sum(),
            A[h:,:h].diagonal().sum()/A[h:,:h].sum(),
            A[h:,h:].diagonal().sum()/A[h:,h:].sum(),int(A.shape[0]/2)))
        print(' * B shape:',B.shape)
        print(' * C shape:',C.shape)

        if dt is not None:
            if CUDA_LIBRARY:
                Ad,Bd = self.gpu_c2s(A,B,dt)
            else:
                Ad,Bd = self.c2s(A,B,dt)
            return Ad,Bd,C
        else:
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
        """
        Computes the model transfer function between inputs and outputs

        Parameters:
        -----------
        nu: ndarray
            The frequency vector in Hz
        inputs: list
            The list of model inputs
        outputs: list
            The list of model outputs

        Returns:
        --------
        G : the transfer function matrix
        """
        _Phim_ = np.vstack([self.__Phim__[x] for x in inputs])
        _Phi_ = np.vstack([self.__Phi__[x] for x in outputs])
        G = np.zeros((nu.size,_Phi_.shape[0],_Phim_.shape[0]),dtype=np.complex)
        for k in range(nu.size):
            G[k,...] = freqrep(nu[k],_Phi_,_Phim_,self.O,self.Z,n_mode_max=None)
        return G

    def mountTransferFunction(self,nu,axis='both',
                              ElDrive_F = 'OSS_ElDrive_F',
                              ElDrive_D = 'OSS_ElDrive_D',
                              AzDrive_F = 'OSS_AzDrive_F',
                              AzDrive_D = 'OSS_AzDrive_D'):
        """
        Computes the mount transfer function
        """
        P = np.atleast_2d([-1]*4+[1]*4).T
        ElTF = ()
        AzTF = ()
        if axis in ['elevation','both']:
            G0 = self.G(nu,[ElDrive_F],[ElDrive_D])
            ElTF = (np.squeeze(P.T@G0@P),)
        if axis in ['azimuth','both']:
            G0 = self.G(nu,[AzDrive_F],[AzDrive_D])
            AzTF = (np.squeeze(P.T@G0@P),)
        TF = ElTF+AzTF
        if len(TF)>1:
            return TF
        else:
            return TF[0]

    def bodeMag(self,nu,TF,_filter_=None,legend=None,ax=None):
        if not isinstance(TF,list):
            TF = [TF]
        if ax is None:
            fig,ax = plt.subplots()
            fig.set_size_inches(10,5)
        for _TF_ in TF:
            if _filter_ is not None:
                _TF_ *= _filter_
            ax.semilogx(nu,20*np.log10(np.abs(_TF_)))
        ax.grid(which='both')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Magnitude [dB]')
        if legend is not None:
            ax.legend(legend)
        return ax

    def reduce(self,inputs=None,outputs=None,hsv_rel_threshold=None,n_mode_max=None,
               outputs_mapping=None):
        """
        Reduces the models according to the given
         . inputs
         . outputs
         . Hankel singular value threshold
         . number of modes

        Parameters:
        -----------
        inputs: list
            The list of model inputs
        outputs: list
            The list of model outputs
        hsv_rel_threshold: float
            Hankel singular value threshold
        n_mode_max:
            Number of modes
        """
        self.logger.info(' => Model reduction')
        x_size, in_size, out_size = self.O.size, self.Phim.shape[0], self.Phi.shape[0]
        total_complexity = x_size * (x_size+in_size+out_size)
        if inputs is not None:
            self.INPUTS = [(x,self.__Phim__[x].shape[0]) for x in inputs]
            self.Phim = np.vstack([self.__Phim__[x] for x in inputs])
        if outputs is not None:
            outputs = self._verify_bendingmodes(outputs)
            if len(outputs) > 0:
                self.OUTPUTS = [(x,self.__Phi__[x].shape[0]) for x in outputs]
                self.Phi = np.vstack([self.__Phi__[x] for x in outputs])
            self._build_bendingmodes()
        if outputs_mapping is not None:
            (old_outputs,new_output,T) = outputs_mapping
            O = np.vstack([self.__Phi__[x] for x in old_outputs])
            Q = T@O
            outputs_name = [x[0] for x in self.OUTPUTS]
            for k in old_outputs:
                self.OUTPUTS.pop(outputs_name.index(k))
                self.__Phi__.pop(k)
            self.__Phi__[new_output] = Q
            self.OUTPUTS += (new_output,T.shape[0])
            self.Phi = np.vstack([self.__Phi__[x] for x in self.OUTPUTS])
        if hsv_rel_threshold is not None:
            hsv_max = np.max(self.hsv[~np.isinf(self.hsv)])
            hsvn = self.hsv/hsv_max
            n_mode_max = np.sum(hsvn>hsv_rel_threshold)
        if n_mode_max is not None:
            self.Phim = self.Phim[:,:n_mode_max]
            self.Phi  = self.Phi[:,:n_mode_max]
            self.O = self.O[:n_mode_max]
            self.Z = self.Z[:n_mode_max]
        self.__setprop__()
        x_size, in_size, out_size = self.O.size, self.Phim.shape[0], self.Phi.shape[0]
        final_complexity = x_size * (x_size+in_size+out_size)
        factor = round(100*( 1-(final_complexity/total_complexity)),2)
        self.logger.info('   |- Model reduced in: ' + str(factor) + '% of complexity.')
        self.logger.info(' <= Reduction procedure finished!')
        self.info()
    
    def _verify_bendingmodes(self, outputs):
        self.build_bendingmodes = False
        if hasattr(self, 'output_support'):
            self.build_bendingmodes = True
            outputs.remove('bending_modes')
        return outputs
    
    def _build_bendingmodes(self):
        self.logger.info('  => Setting up the bending modes:')
        if self.build_bendingmodes:
            bm_data = self.output_support['bending_modes']
            Q = self._read_mat_file(bm_data['path'],'Q_incell')
            U = self._read_mat_file(bm_data['path'],'U_')
            Phi = self.__Phi__[bm_data['variable']['name']]
            if 'segment' in bm_data['variable']:
                segId = bm_data['variable']['segment']
                self.logger.info('    |- segment: ' + str(segId))
                Q, U = Q[segId,0], U[segId,0]
            else:
                segId = 0
                self.logger.info('   |- segment: all segments')
                Q = self._create_diag_from_mat(mat=Q) #np.vstack([Q[k,0] for k in range(Q.shape[0])])
                U = self._create_diag_from_mat(mat=U) #np.vstack([U[k,0] for k in range(U.shape[0])])
            _shape = U.shape 
            _init = _shape[0] * segId + 2
            _final = _init + 3*_shape[0]
            _ind = range(_init, _final, 3)
            self.Phi = np.vstack([self.Phi, Phi[_ind,:]])
            self.OUTPUTS.append(('zdisplacements', _shape[0]))
            self.OUTPUTS.append(('bending_modes', _shape[1]))

            _current = 0
            for item in self.OUTPUTS:
                if item[0] == 'zdisplacements':
                    _indexes = [_current + k for k in range(_shape[0])]
                else:
                    _current += item[1]
            
            if CUDA_LIBRARY:
                self.bm_states = {
                    'ind': _indexes,
                    'Q' : cp.array(Q, dtype=np.float32),
                    'U.T' : cp.array(U.T, dtype=np.float32),
                    'Qinv' : cp.array(np.linalg.pinv(Q), dtype=np.float32),
                    'bm_magnitude' : cp.zeros((U.shape[1],), dtype=np.float32),
                    'pistontiptilt': cp.zeros((U.shape[1],), dtype=np.float32),
                    'displacements': cp.zeros((U.shape[0],), dtype=np.float32),
                    'on_cuda': True
                }
            else:
                self.bm_states = {
                    'Q' : Q, 'U.T' : U.T, 'ind': _indexes,
                    'bm_magnitude' : np.zeros((U.shape[1],), dtype=np.float32),
                    'pistontiptilt': np.zeros((U.shape[1],), dtype=np.float32),
                    'displacements': np.zeros((U.shape[1],), dtype=np.float32),
                    'Qinv' : np.linalg.pinv(Q),
                    'on_cuda': False
                }
            self._info_bendingmodes()

    def _create_diag_from_mat(self, mat=None):
        _size = mat.shape[0]
        _lin, _col = [], []
        for k in range(_size):
            _lin.append(mat[k,0].shape[0])
            _col.append(mat[k,0].shape[1])
        
        ki, kii = 0, 0
        _shape = (sum(_lin),sum(_col))
        _result = np.zeros(_shape)
        for k in range(_size):
            xi, xii = ki+_lin[k], kii+_col[k]
            _result[ki:xi,kii:xii] = mat[k,0][:,:]
            ki, kii = xi, xii
        
        return _result

    def _info_bendingmodes(self):
        self.logger.info('    |- bending modes size: ' + str(self.bm_states['bm_magnitude'].shape[0]))
        self.logger.info('    |- displacements size: ' + str(self.bm_states['displacements'].shape[0]))
        self.logger.info('    |- This module includes the last 3 unused bending modes!!!')
        self.logger.info('    |- running on GPU: ' + str(self.bm_states['on_cuda']))
        self.logger.info('  <= Bending modes output setup completed!')

    
    def _read_mat_file(self, filename, varname=None):
        if varname is not None:
            try:
                data = spio.loadmat(filename)[varname]
            except:
                data = h5py.File(filename,'r')[varname]
        else:
            try:
                data = spio.loadmat(filename)
            except:
                data = h5py.File(filename,'r')
        return data


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
        A,B,_C = self.state_space()
        Q = - B@B.T
        W = solve_lyapunov(A.toarray(),Q.toarray())
        return W


    def Init(self,dt=0.5e-3,
              inputs=None,outputs=None,
              hsv_rel_threshold=None,n_mode_max=None,
              start_idx=0):
        
        self.logger.info('Init')
        self.hsv_sort(start_idx=start_idx)
        self.reduce(inputs=inputs,outputs=outputs,
                    hsv_rel_threshold=hsv_rel_threshold,
                    n_mode_max=n_mode_max)
        A,B,C = self.state_space(dt=dt)
        self.state.update({'u':np.zeros(self.N_INPUTS),
                           'y':np.zeros(self.N_OUTPUTS),
                           'A':A,'B':B,'C':C,'D':None,
                           'x':np.zeros(A.shape[1]),
                           'step':0})
        # Initializing GPU state space
        if CUDA_LIBRARY:
            self._initialize_gpu_env()
      
        # Create the state logging process
        if self.log_states:
            _shape_ = (self.state['A'].shape[0], 20000)
            self._start_state_logging(self.stateLogInfo['path'], _shape_, self.stateLogInfo['vartype'])
            time.sleep(.500)
        
        self.logger.info('      RUNNING SIMULATION...')

    def reset_state(self):
        self.state.update({'u':np.zeros(self.N_INPUTS),
                           'y':np.zeros(self.N_OUTPUTS),
                           'x':np.zeros(2*self.O.size),
                           'step':0})

    def _initialize_gpu_env(self, var_type=np.float64):

        '''
        Initialize the GPU environment and the proper variables.

        Parameters:
        -----------
        vartype : numpy._type_
            The type of the variable saved, it can be numpy.float32 or numpy.float64.
        '''
        
        self.logger.info(" => Setting up GPU environment...")
        self.logger.info(" GPU environment synopsis:")
        self.gpu = dict()
        for element in self.state:
            if sparse.issparse(self.state[element]):
                auxiliar = cp.asarray(self.state[element].todense(), dtype=var_type)
                __size = auxiliar.shape
                self.gpu[element] = cps.csr_matrix(auxiliar, shape=__size, dtype=var_type)
            else:
                self.gpu[element] = cp.asarray(self.state[element], dtype=var_type)
            self.logger.info("    |- {0} shape : {1}".format(element, self.gpu[element].shape))
            
        self.logger.info(" <= GPU environment built successfully!")

    
    def _start_state_logging(self, 
                             savepath, shape,
                             vartype=np.float32):
        
        '''
        Starts a local CPU parallel process for logging state variables.

        Parameters:
        ----------- 
        savepath : String
            Where ones wants to save the variable.
        shape : Tuple or list 
            The shape of the variable saved.
        vartype : numpy._type_
            The type of the variable saved, it can be numpy.float32 or numpy.float64.
        '''

        self.logger.info(" => CPU PARALLEL logging process...")
        self.logger.info("  * Building parallel state logging *")
        # Create the information
        self.stateQueue = Manager().Queue()
        info = {'savepath':savepath, 'shape':shape, 'queue':self.stateQueue, 'varname': 'states'}
        self.logger.info('  * Queue created...')
        # Create the save pack
        self.save_info = {'alive': True, 'index': 0, 'data':np.zeros((shape[0],1), dtype=vartype)}
        # Create the parallel process
        self.saveStateProcess = Process(target = save_variable_on_disk, args = (info,), daemon = True)
        self.logger.info('  * Process created...')
        self.saveStateProcess.start()
        self.logger.info('  * Process started...')
        self.logger.info(' <= Parallel process created!')

    def Update(self, **kwargs):
        _u = self.state['u']
        a, b = 0, 0
        for t, s in self.INPUTS:
            b += s
            if t in kwargs:
                _u[a:b] = kwargs[t]
            a += s

        if CUDA_LIBRARY:
            self.gpu['u'] = cp.array(_u, dtype = np.float64)
            self.gpu['x'] = self.gpu['A'].dot(self.gpu['x']) + self.gpu['B']@self.gpu['u']
            self.gpu['y'] = self.gpu['C'].dot(self.gpu['x']) 
            self.state['y'] = self.gpu['y'].get().ravel()
            if self.log_states:
                self.save_info['data'] = self.gpu['x'][:].get()
                self.stateQueue.put(self.save_info)
                self.save_info['index'] += 1
            if self.build_bendingmodes:
                self.bm_states['displacements'] = self.gpu['y'][self.bm_states['ind']]
                self.bm_states['pistontiptilt'] = cp.dot(self.bm_states['Qinv'],self.bm_states['displacements'])
                self.bm_states['pistontiptilt'] = cp.dot(self.bm_states['Q'],self.bm_states['pistontiptilt'])
                self.bm_states['displacements'] = self.bm_states['displacements'] #- self.bm_states['pistontiptilt']
                self.bm_states['bm_magnitude']  = cp.dot(self.bm_states['U.T'], self.bm_states['displacements'])
                self.state['y'] = np.concatenate((self.state['y'], self.bm_states['bm_magnitude'].get()), axis=0)
        else:
            _x = self.state['x']
            x_next = self.state['A']@_x + self.state['B']@_u
            _y = self.state['C']@_x
            self.state['x']    = x_next.flatten()
            self.state['y'][:] = _y.ravel()
        
        self.state['step']+=1

    def Outputs(self,**kwargs):
        d = {}
        if kwargs:
            outputs = kwargs['outputs']
            a = 0
            b = 0
            for t,s in self.OUTPUTS:
                b += s
                if t in outputs:
                    d[t] = self.state['y'][a:b]
                    """
                    if t in self.P:
                        #print('HERE')
                        d[t] = self.P[t]@d[t]
                    """
                a += s
        return d

    def Terminate(self,**kwargs):
        return "FEM deleted"



# update_states = cp.ElementwiseKernel(
#     'raw X A, Y Bu, raw T st',
#     'T next',
#     '''
#         int aux = (int)i / 2;
#         int index = 2 * aux;

#         next = A[2*i] * st[index] + A[2*i + 1] * st[index + 1] + Bu 
#     ''',
#     'update_states'
# )


# update_states_diag = cp.ElementwiseKernel(
#     'X A, Y Bu, T st',
#     'T next',
#     '''
#         next = A * st  + Bu 
#     ''',
#     'update_states_diag'
# )
