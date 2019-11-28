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

def get_SHWFS_D(Din, **kwargs):
    """ Compute a consolidated Shack-Hartmann interaction matrix from 
        segment-wise data. The output format is compatible with the
        edge sensor pattern. The number of bending modes N-bm is an optional
        input argument.
    """
    if len(Din) != 7:
        raise Exception('Interaction matrix is not partitioned segment-wise as expected!')
    
    # Number of bending modes
    if 'n_bm' in kwargs.keys():
        n_bm = kwargs['n_bm']
    else:
        n_bm = Din[0].shape[1]-12
    
    # Number of M1/2-RBM DoF (M2 S7-Rz is not considered)
    n_M1RBM, n_M2RBM = 41, 41
    
    
    # Interaction matrix dimension: SH-WFS valid measurements x DoF
    n_sh, n_x = sum(Dseg.shape[0] for Dseg in Din), n_M1RBM + n_M2RBM + 7*n_bm

    # RBM D columns - format compatible to edge sensor interaction matrix
    Dcolsel = np.hstack([np.arange(6), n_M1RBM+np.arange(6)])
    # Initialize variables
    D, init_row = np.zeros((n_sh,n_x)), 0
    print('Consolidated WFS-SH Interaction matrix is',D.shape[0],'x',D.shape[1])
    # Fill interaction matrix for outer segments
    for k in range(6):
        coli = np.hstack(( Dcolsel+(6*k), n_M1RBM+n_M2RBM+(n_bm*k)+np.arange(n_bm) ))
        D[init_row : Din[k].shape[0]+init_row, coli] = Din[k][:,:12+n_bm]
        init_row = init_row + Din[k].shape[0]
    # Fill interaction matrix for center segments
    k = 6
    D[-Din[k].shape[0]:, np.hstack( (Dcolsel[:5]+(6*k),Dcolsel[6:11]+(6*k),
        np.arange(n_bm)+(n_M1RBM+n_M2RBM+n_bm*k) ))] = Din[k][:,:10+n_bm]

    return D

def merge_SH_ES_D(Dsh, De, alphaBM=1, alphaEs=1):
    """ Merge SH-WFS (Dsh) and M1 edge sensor (De) interaction matrices. The merged
    matrix is used to build a reconstructor using both sources of information. To merge
    the matrices, a column relative to M1S7-Rz is introduced.
    """
    
    # Compute consolidated interation matrix if it is segment-wise
    if len(Dsh) == 7:
        Dsh = get_SHWFS_D(Dsh)
    # Number of M1&2 RBM (M1&M2 S7-Rz are not considered)
    n_M1_RBM, n_M2_RBM = 41, 41
    if(De.shape[1] == 42):
        # If ES interaction matrix is full, introduce a column for M1S7Rz into Dsh
        Dsh = np.insert(Dsh,n_M1_RBM,0.0, axis=1)
        n_M1_RBM = n_M1_RBM+1
    
    # Merged interaction matrix
    Da = np.vstack([
            # SH-WFS block
            np.hstack([Dsh[:,:n_M1_RBM],           # M1 RBM (wo S7 Rz)
                Dsh[:,n_M1_RBM:(n_M1_RBM+n_M2_RBM)],           # M2 RBM
                alphaBM*Dsh[:,(n_M1_RBM+n_M2_RBM):]]),      # M1 bending modes
            # Edge sensor block    
            np.hstack([alphaEs*De, 
                np.zeros((De.shape[0], Dsh.shape[1]-De.shape[1]))])
        ])

    return Da

def build_TSVD_RecM(D, n_r=0, insM1M2S7Rz=True):
    """ Build a reconstructor matrix (M) as the truncated singular value 
    decomposion (TSVD) a sort of the interaction matrix (D). The procedure 
    filters out the contribution of the n_r weakest singular values.
    """
    U,sigma,V = np.linalg.svd(D, full_matrices=False)
    if(n_r):
        # Truncated SVD
        i_sigma = np.diag(1/sigma[:-n_r])
        M = np.transpose(V[:-n_r,:]).dot(i_sigma).dot(np.transpose(U[:,:-n_r]))
    else:
        i_sigma = np.diag(1/sigma)
        M = np.transpose(V).dot(i_sigma).dot(np.transpose(U))
    # Psuedo-inverse from truncated SVD

    if(insM1M2S7Rz):
        M = insert_M1M2_S7_Rz(M)

    return M


def build_RLS_RecM(Dsh, De, n_r, insM1M2S7Rz=True):
    """ The function builds the active optics reconstructor from the
    interaction matrices Dsh (Shack-Hartmann WFS) and De (M1 Edge sensors).
    Th recontructor matrix is achieved using a regularized least-squares approach.
    """

    Da = merge_SH_ES_D(Dsh, De, alphaBM=1, alphaEs=1)
    
    _U,sigma,V = np.linalg.svd(Da, full_matrices=False)
    q = 0.0*np.zeros_like(sigma)
    q[-n_r:] = sigma[-1]/sigma[-n_r:] #  np.ones(n_r) #
    print('Regularization term coefficients:\n',q[-n_r:])
    Q = np.diag(q).dot(V)

    A = Da.T.dot(Da) + np.dot(Q.T,Q)
#    UA,sigmaA,VA = np.linalg.svd(A, full_matrices=False)
#    left_inv_A = np.transpose(VA).dot(np.diag(1/sigmaA)).dot(np.transpose(UA))
    left_inv_A = np.linalg.pinv(A) #,rcond=1e-7)
    M = left_inv_A.dot(Da.T)

    if(insM1M2S7Rz) and (De.shape[1] < 42):
        M = insert_M1M2_S7_Rz(M)
    else:
        M = np.vstack([ M[:83,:], np.zeros((1,M.shape[1])),
                        M[83:,:]])

    return M


def insert_M1M2_S7_Rz(M):
    # Introduce zero rows into M for M1M2-S7Rz for compatibility
    return np.vstack([M[:41,:], np.zeros((1,M.shape[1])),
                        M[41:82,:], np.zeros((1,M.shape[1])),
                        M[82:,:]])


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
        if 'W2' not in kwargs.keys():
            # Weighting term based on DoF range
            w_M1TxyzRxyz = np.diag(np.array([1,1,1,4,4,4]))
            w_M2TxyzRxyz = np.diag(np.array([50,40,40,8,8,8]))          
            w_M1BM = 0.01*np.eye(n_bm)
            W2 = block_diag(w_M1TxyzRxyz, w_M2TxyzRxyz, w_M1BM)

            W2_coeff = 1/np.trace(W2)
            W2 = W2_coeff*W2
            print('Weighting factor of W2:',W2_coeff)
        else:
            W2 = kwargs['W2']
        M_mixed = [ aco_rls(X,W2,Y,Z) for X,Y,Z in zip(D,_n_threshold_,zeroIdx) ]
        M1 = block_diag(*[Mseg_mixed[:,:Dseg.shape[0]] for (Mseg_mixed,Dseg) in zip(M_mixed,D)])
        M2 = block_diag(*[Mseg_mixed[:,Dseg.shape[0]:] for (Mseg_mixed,Dseg) in zip(M_mixed,D)]) 
        M = np.hstack([M1, M2])

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

def aco_rls(Dseg, W2, n_r, zeroIdx):
    _U,sigma,V = np.linalg.svd(Dseg,full_matrices=False)
    
    if(n_r):
        # Regularize poorly sensed modes (clocking)
        w1 = 0.0*np.zeros_like(sigma)
        w1[-n_r:] = 1.0
        sqrtW1 = np.diag(w1).dot(V)
    else:
        sqrtW1 = np.eye(Dseg.shape[1])

    if zeroIdx is not None:
        # Remove weights relative to M1 and M2 S7-Rz
        W2_ = np.delete(W2,[6,12],0)
        W2_ = np.delete(W2_,[6,12],1)
    else:
        W2_ = W2
    
    # Solve RLS problem
    pinvA = np.linalg.pinv(Dseg.T.dot(Dseg) + np.dot(sqrtW1.T,sqrtW1) + W2_)
    Mseg = pinvA.dot( np.hstack([Dseg.T, W2_]) )
    
    if zeroIdx is not None:
        Mseg = np.insert(Mseg, zeroIdx, 0, axis=0)     
        Mseg = np.insert(Mseg, Dseg.shape[0]+6, 0, axis=1)
        Mseg = np.insert(Mseg, Dseg.shape[0]+12, 0, axis=1)

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




# Algorithm conceived by R.Conan Nov 1st, 2019
def build_CLS_RecM(Dsh, De, Pm2):
    n_x = Dsh.shape[1]
    n_constr = De.shape[0]+Pm2.shape[0]
    DshTDsh = Dsh.T.dot(Dsh)
    Gamma = np.hstack([ block_diag(De,Pm2), 
                        np.zeros((n_constr,n_x-(De.shape[1]+Pm2.shape[1])))])
    F = np.block([np.eye(n_x), np.zeros((n_x,n_constr))])
    L = np.block([[DshTDsh, Gamma.T],
                [Gamma, np.zeros((n_constr,n_constr))]])
    iL = np.linalg.pinv(L) #,rcond=1e-8)
    H = np.vstack([block_diag(Dsh.T,np.eye(De.shape[0])),
            np.zeros((Pm2.shape[0],Dsh.shape[0]+De.shape[0]))])
    M = F.dot(iL).dot(H)

    return M
