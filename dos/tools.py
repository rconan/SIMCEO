import pickle
import  numpy as np
from scipy import sparse
from scipy.linalg import block_diag


def linear_estimator_2_dos(estimator,dos_path, B_or_D='D'):
    """ Convert a linear estimator to a SIMCEO parameter file
    """
    n = estimator.shape[0]
    if B_or_D=='B':
        A = np.zeros((n,n))
        B = estimator
        C = np.eye(n)
        D = np.zeros_like(B)
    if B_or_D=='D':
        A = np.zeros((n,n))
        B = np.zeros_like(estimator)
        C = np.zeros((n,n))
        D = estimator
    sys = {'System': { 'parameters': {'A':A ,'B':B,'C':C,'D':D} } }
    with open(dos_path+'.pickle', 'wb') as f:
        pickle.dump(sys,f)


def state_space_2_dos(A,B,C,D,dos_path):
    """ Convert a state space model to a SIMCEO parameter file
    """
    sys = {'System': { 'parameters': {'A':A ,'B':B,'C':C,'D':D} } }
    with open(dos_path+'.pickle', 'wb') as f:
        pickle.dump(sys,f)


def build_RLS_RecM(Dsh, W2, W3, K, rho_2, rho_3, insM1M2S7Rz=True):
    """ The function builds the active optics reconstructor from the
    interaction matrices Dsh (Shack-Hartmann WFS).
    Th recontructor matrix is achieved using a regularized least-squares approach.
    """

    # Just implemented for K := float (LTI integral controller) for now!!!
    left_sym_inv = np.linalg.pinv(Dsh.T.dot(Dsh) + rho_2*W2 + rho_3*K*W3*K)
    M = left_sym_inv.dot( np.hstack([Dsh.T, rho_3*K*W3]) )

    if(insM1M2S7Rz):
        # Get the number of bending modes in Dsh
        n_bm = ((Dsh.shape[1]+2)//7) - 12
        n_c_oa, n_s = (12+n_bm)*6, Dsh.shape[0]
        print('%d BMs used in the reconstructor computation.'%n_bm)
        # Insert M1/2S7-Rz rows into M
        M = np.insert(M,[n_c_oa+5,n_c_oa+10],0,axis=0)
        # Insert M1/2S7-Rz columns related to W3 into M
        M = np.insert(M,[n_s+n_c_oa+5,n_s+n_c_oa+10],0,axis=1)

    return M
    

def build_TSVD_RecM(Dsh, n_r=0, insM1M2S7Rz=True):
    """ Build a reconstructor matrix (M) as the truncated singular value 
    decomposion (TSVD) a sort of the interaction matrix (D). The procedure 
    filters out the contribution of the n_r weakest singular values.
    """
    U,sigma,V = np.linalg.svd(Dsh, full_matrices=False)
    if(n_r):
        # Truncated SVD
        i_sigma = np.diag(1/sigma[:-n_r])
        M = np.transpose(V[:-n_r,:]).dot(i_sigma).dot(np.transpose(U[:,:-n_r]))
    else:
        i_sigma = np.diag(1/sigma)
        M = np.transpose(V).dot(i_sigma).dot(np.transpose(U))

    if(insM1M2S7Rz):
        n_bm = ((Dsh.shape[1]+2)//7) - 12
        n_c_oa = (12+n_bm)*6
        print('%d BMs used in the reconstructor computation.'%n_bm)
        # Insert M1/2S7-Rz rows into M
        M = np.insert(M,[n_c_oa+5,n_c_oa+10],0,axis=0)

    return M


def build_AcO_Rec(fullD, **kwargs):
    """Given segment-wise interaction matrices the function returns the AcO 
    reconstructor. If the optional argument wfsMask is used, the output is compatible
    with the 'data' output of wfs48 driver. Otherwise, the valid slopes are the inputs
    of the computed reconstruction matrix."""

    if not (len(fullD) == 7):
        print('First argument must be a list of interaction matrix from each segment!')  
    
    # Number of bending modes to be considered for the reconstruction
    if 'n_bm' in kwargs.keys():
        n_bm = kwargs['n_bm']
        
    remBM = (fullD[0].shape[1]-12) - n_bm
    if(remBM > 0):
        D = [fullDseg[:,:-remBM] for fullDseg in fullD]
        print('%d BMs used in the reconstructor computation.'%n_bm)
    else:
        D = fullD
        n_bm = fullD[0].shape[1]-12
        print('All %d calibrated BMs are considered in the reconstructor matrix.'%n_bm)
    
    # Get reconstruction algorithm
    if 'rec_alg' in kwargs.keys():
        rec_alg = kwargs['rec_alg']
    else:
        rec_alg = 'TSVD'

    # Zeros to be inserted into the reconstruction matrix for M1/M2 S7Rz
    zeroIdx = [None]*6 + [[5,10]]
    # Number of modes to be filtered
    _n_threshold_ = [2,2,2,2,2,2,0]

    if(rec_alg == 'TSVD'):
        UsVT = [np.linalg.svd(Dseg,full_matrices=False) for Dseg in D]            
        M = block_diag(*[ aco_tsvd(X,Y,Z) for X,Y,Z in zip(UsVT,_n_threshold_,zeroIdx) ])

    elif(rec_alg == 'RLS'):
        try:
            W3 = kwargs['W3']
        except:
            W3 = 0.0*np.eye(12+n_bm)
            print('No command norm regularization! W3 = 0.')

        M_mixed = [ aco_rls(X,W3,Y,Z) for X,Y,Z in zip(D,_n_threshold_,zeroIdx) ]
        Msh = block_diag(*[Mseg_mixed[:,:Dseg.shape[0]] for (Mseg_mixed,Dseg) in zip(M_mixed,D)])
        Mu = block_diag(*[Mseg_mixed[:,Dseg.shape[0]:] for (Mseg_mixed,Dseg) in zip(M_mixed,D)]) 
        M = np.hstack([Msh, Mu])

    if 'wfsMask' in kwargs.keys():
        return gen_recM_4_SIMCEO(M, kwargs['wfsMask'], reorder2CEO=False)
    else:
        return M

def aco_tsvd(_UsVT_, _n_threshold_, zeroIdx):
    iS = 1./_UsVT_[1]
    if _n_threshold_>0:
        iS[-_n_threshold_:] = 0        

    _M_ = np.dot(_UsVT_[2].T,np.dot(np.diag(iS),_UsVT_[0].T))
    if zeroIdx is not None:
        _M_ =  np.insert(_M_,zeroIdx,0,axis=0)
    return _M_

def aco_rls(Dseg, W3, n_r, zeroIdx):
    _U,sigma,VT = np.linalg.svd(Dseg,full_matrices=False)
    
    if(n_r):
        # Regularize poorly sensed modes (clocking)
        w1 = 0.0*np.zeros_like(sigma)
        w1[-n_r:] = 1.0
        sqrtW1 = np.diag(w1).dot(VT)
    else:
        sqrtW1 = np.eye(Dseg.shape[1])

    if zeroIdx is not None:
        # Remove weights relative to M1 and M2 S7-Rz
        W3_ = np.delete(W3,[6,12],0)
        W3_ = np.delete(W3_,[6,12],1)
    else:
        W3_ = W3
    
    # Solve RLS problem
    pinvA = np.linalg.pinv(Dseg.T.dot(Dseg) + np.dot(sqrtW1.T,sqrtW1) + W3_)
    Mseg = pinvA.dot( np.hstack([Dseg.T, W3_]) )
    
    if zeroIdx is not None:
        Mseg = np.insert(Mseg, zeroIdx, 0, axis=0)     
        Mseg = np.insert(Mseg, [Dseg.shape[0]+5,Dseg.shape[0]+10], 0, axis=1)

    return Mseg


def gen_recM_4_SIMCEO(M, wfsMask, reorder2CEO=True):
    """ The function performs an input transformation on the recontructor 
    matrix (M) based on a mask (wfsMask), so that the output Msimceo is 
    compatible with raw output from the wfs48 SIMCEO driver. The flux threshold
    determines the valid WFS measurement in wfsMask.
    """

    if len(wfsMask) != 7:
        raise Exception('SH-WFS mask is not partitioned segment-wise as expected!')

    # Adjust wfs mask data dimension
    for k in range (7):
        wfsMask[k] = wfsMask[k].ravel()#np.reshape(wfsMask[k],wfsMask[k].shape[0]*wfsMask[k].shape[1],1)

    # SH-WFS valid meas dimension
    n_sh = sum(np.count_nonzero(segMask) for segMask in wfsMask)
    # Initialize mask
    Mmask = sparse.lil_matrix(np.zeros((n_sh, len(wfsMask[0]))))
    # Matrix row pointer
    init_row = 0

    for k in range( len(wfsMask) ):
        iv = np.flatnonzero(wfsMask[k])
        Mmask[np.arange(init_row, len(iv)+init_row), iv] = 1
        init_row = init_row + len(iv)

    RecM = sparse.hstack([sparse.lil_matrix(M[:,:n_sh]).dot(Mmask),
            sparse.lil_matrix(M[:,n_sh:])]).toarray()

    if(reorder2CEO):
        Msimceo = np.empty_like(RecM)
        print('Reordering SIMCEO reconstructor matrix')
        n_bm = (RecM.shape[0] - 84)//7
        if((RecM.shape[0] - 84) % 7):
            print('The number of bending modes retrieved from Dsh is not an integer.\n'
            'Check the dimension of the reconstructor matrix!')   

        for k in range (7):
            row_sel = np.hstack([np.arange(6)+6*k,
                                np.arange(42,48)+6*k,
                                np.arange(84,84+n_bm)+n_bm*k])
            Msimceo[len(row_sel)*k:len(row_sel)*(k+1),:] = RecM[row_sel,:]
        return Msimceo
    else:
        print('No reconstructor matrix row reordering!')
        return RecM