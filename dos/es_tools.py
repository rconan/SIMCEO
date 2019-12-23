#
# This file provides a set of functions to evaluate 
# the effectiveness of using M1 edge sensors mesurements in 
# the active optics (AcO) reconstruction problem
#
# Sep-Nov / 2019
# Author: Rodrigo A. Romano

import pickle
import  numpy as np
from scipy import sparse
from scipy.linalg import block_diag


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
    left_inv_A = np.linalg.pinv(A)
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
