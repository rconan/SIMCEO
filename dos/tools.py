import pickle
import  numpy as np
from scipy import sparse

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

def get_SHWFS_D(Din, includeM1S7Rz_col=True):
    """ Compute a consolidated Shack-Hartmann interaction matrix from 
        segment-wise data. The output format is compatible with the
        edge sensor pattern. If includeM1S7Rz_col is True (a column for 
        M1-S7Rz is introduced) the number of M1&2 RBM is 83, otherwise
        it is 82.
    """
    if len(Din) != 7:
        raise Exception('Interaction matrix is not partitioned segment-wise as expected!')
    
    # Number of bending modes
    n_bm = Din[0].shape[1]-12
    # Number of M1/2-RBM DoF (M2 S7-Rz is not considered)
    n_M1RBM, n_M2RBM = 41, 41
    if(includeM1S7Rz_col):
        n_M1RBM = 42

    # Interaction matrix dimension: SH-WFS valid measurements x DoF
    n_sh, n_x = sum(Dseg.shape[0] for Dseg in Din), n_M1RBM + n_M2RBM + 7*n_bm

    # RBM D columns - format compatible to edge sensor interaction matrix
    Dcolsel = np.hstack([np.arange(6), n_M1RBM+np.arange(6)])
    # Initialize variables
    D, init_row = np.zeros((n_sh,n_x)), 0
    print('WFS-SH Interaction matrix is',D.shape[0],'x',D.shape[1])
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
    """ Merge SH-WFS (Dsh) and edge sensor (De) interaction matrices. The merged
    matrix is used to build a reconstructor using both sources of information
    """
    
    # Compute consolidated interation matrix if it is segment-wise
    if len(Dsh) == 7:
        Dsh = get_SHWFS_D(Dsh, includeM1S7Rz_col=True)
    # Number of M1&2 RBM (M2-S7Rz is not considered)
    n_M12RBM = 83    
    # Merged interaction matrix
    Da = np.vstack([
            np.hstack([Dsh[:,:n_M12RBM], alphaBM*Dsh[:,n_M12RBM:]]),
            np.hstack([alphaEs*De, np.zeros((De.shape[0],Dsh.shape[1]-De.shape[1]))])
        ])

    return Da

def build_TSVD_RecM(D, n_r=0):
    """ Build a reconstructor matrix (M_TSVD) as a sort of pseudo-inverse 
    of the interaction matrix (D). The procedure filters out the contribution
    of the n_r weakest singular values.
    """
    
    U,sigma,V = np.linalg.svd(D, full_matrices=False)

    # Truncated SVD
    i_sigma = np.diag(1/sigma[:-n_r])
    M_TSVD = np.transpose(V[:-n_r,:]).dot(i_sigma).dot(np.transpose(U[:,:-n_r]))
    # Introduce a row for M2-S7Rz for compatibility
    M_TSVD = np.vstack([M_TSVD[:83,:], np.zeros((1,M_TSVD.shape[1])), M_TSVD[83:,:]])

    # Psuedo-inverse from truncated SVD
    return M_TSVD


def build_RLS_RecM(D, mu = 1):
    """ Function comments ...
    """
    
    _U,sigma,V = np.linalg.svd(D, full_matrices=False)

    # Weighting matrices
    # q = (0.0)*np.ones_like(sigma)
    # q[-n_r:] = sigma[-1:]/sigma[-n_r:] # 1*np.ones(nr) #

    q = mu*sigma[-1]/sigma # 1*np.ones(nr) #
    print('Regularization term coefficients:\n',q)
    Q = np.diag(q).dot(V)

    # Damped (Regularized) least-squares
    M_RLS = np.dot(np.linalg.inv(np.dot(D.T,D) + np.dot(Q.T,Q)), D.T)
    # Introduce a row for M2-S7Rz for compatibility
    M_RLS = np.vstack([M_RLS[:83,:], np.zeros((1,M_RLS.shape[1])), M_RLS[83:,:]])
    return M_RLS


def gen_recM_4_SIMCEO(M, wfsMask):
    """ The function performs an input transformation on the recontructor 
    matrix (M) based on a mask (wfsMask), so that the output Msimceo is 
    compatible with raw output from the wfs48 SIMCEO driver. The flux threshold
    determines the valid WFS measurement in wfsMask.
    """

    if len(wfsMask) != 7:
        raise Exception('SH-WFS mask is not partitioned segment-wise as expected!')

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
        
    Msimceo = np.empty_like(RecM) 
    n_bm = (RecM.shape[0] - 84)//7

    for k in range (7):
        row_sel = np.hstack([np.arange(6)+6*k,
                             np.arange(42,48)+6*k,
                             np.arange(84,84+n_bm)+n_bm*k])
        Msimceo[len(row_sel)*k:len(row_sel)*(k+1),:] = RecM[row_sel,:]

    return Msimceo
