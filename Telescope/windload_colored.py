import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import welch
import logging
import s3fs
from scipy.io import loadmat, savemat
import logging
import itertools,datetime

logging.basicConfig()

inputs_name = {'top-end':['OSS_TopEnd_6F','MC_M2_TE_6F','MC_M2_TE_6F'],
               'truss':['OSS_Truss_6F']*3,
               'GIR':['OSS_GIR_6F']*3,
               'C-Ring':['OSS_CRING_6F']*3,
               'M1':{'cell':['OSS_Cell_lcl_6F']*3,
                     'segment':['OSS_M1_lcl_6F']*3},
               'M2':{'cell':['MC_M2_MacroCell_F_6d','MC_M2_MacroCell_6F','MC_M2_RB_6F'],
                     'segment':['MC_M2_lcl_force_6F']*3}}


def Rx(theta):
    theta *= np.pi/180
    return np.asarray([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
def Ry(theta):
    theta *= np.pi/180
    return np.asarray([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
def Rz(theta):
    theta *= np.pi/180
    return np.asarray([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])

FEM_nodes = {'top-end':[0, 0, 25.2],
             'truss':
             {'bottom':[[-4.22936,7.316662,7.710727],
                        [8.431499,0.00042829,7.630185],
                        [-4.21956,-7.30688,7.63721]],
              'top':[[-2.01533,3.667605,18.39511],
                     [4.194657,-0.090631,18.37078],
                     [-2.17225,-3.59239,18.37593]]},
             'GIR':[0.007,0.021,-4.49],
             'C-Ring':[[0,0,0],
                       [5.76,0,0],
                       [-5.76,0,0]],
             'M1':[[0, 8.7100, 4.953694],
                   [7.543081, 4.3550, 4.953694],
                   [7.543081,-4.3550, 4.953694],
                   [0,-8.7100, 4.953694],
                   [-7.543081,-4.3550, 4.953694],
                   [-7.543081, 4.3550, 4.953694],
                   [0,0, 3.9000]],
             'M2 cell':[0, 0, 24.86],
             'M2':[[0,-1.08774,24.0197081],
                   [-0.9420105,-0.54387,24.0197081],
                   [-0.9420105, 0.54387,24.0197081],
                   [         0, 1.08774,24.0197081],
                   [0.9420105, 0.54387,24.0197081],
                   [0.9420105,-0.54387,24.0197081],
                   [0,0,24.1624761]]}

def cfd2fem(F,M,R,rot_mat=np.eye(3)):
    """
    Convert CFD forces and moments into FEM forces and moments
    F : ndarray
        The CFD forces
    M : ndarray
        The CFD moments
    R : list
        The FEM nodes (x,y,z) coordinates where forces and moments are applied
    """
    FM_IM = np.zeros((F.shape[0],6))
    FM_IM[:,:3] = (rot_mat@F.T).T

    if R:
        R = np.asarray(R)
        #M = R x F
        #Mx = Ry Fz - Rz Fy
        #My = Rz Fx - Rx Fz
        #Mz = Rx Fy - Ry Fx
        crossR = np.asarray([ [0, -R[2], R[1]], [R[2], 0, -R[0]], [-R[1], R[0], 0]])
        FM_IM[:,3:] = (rot_mat@(M.T-crossR@F.T)).T
    else:
        FM_IM[:,3:] = (rot_mat@(M.T)).T

    return FM_IM

def add_colored_noise(FM_IM,fs,windload_fs,psd_exp=-13/3):

    if fs==windload_fs:
        return FM_IM

    s0n = []

    for k in range(FM_IM.shape[1]):

        FM = FM_IM[:,k]

        # sub-sampling from windload_fs to fs
        fs_ratio = fs/windload_fs
        x  = np.arange(FM.shape[0])*fs_ratio
        xi = np.arange(x[-1])
        if FM.sum()==0:
            s0n += [np.zeros(int(xi.size))]
            continue
        yi = interp1d(x,FM.T,kind='linear',bounds_error=True)(xi)

        # PSD of sub-sampled signal 
        nu0i,W0i = welch(yi,fs=fs,nperseg=1e3*fs_ratio)
        fny = windload_fs/2
        idx = np.logical_and(nu0i>(fny-2),nu0i<(fny+2))
        yc = np.median(W0i[idx])

        # Synthetic signal above Nyquist frequency
        nSample = xi.size
        nu = np.fft.fftfreq(nSample,1/fs)
        A = yc*fny**(13/3)
        psd = np.zeros_like(nu)
        idx = np.abs(nu)>(fny-windload_fs/5)
        psd[idx] = A*np.abs(nu[idx])**psd_exp

        phi = np.exp(2*1j*np.pi*np.random.randn(int(nSample/2)))
        if nSample%2:
            w = np.sqrt(psd)*np.hstack([1,phi[:],np.flipud(np.conj(phi))])
        else:
            w = np.sqrt(psd)*np.hstack([1,phi[:-1],np.flipud(np.conj(phi))])
        s = np.fft.ifft(w)*w.size*np.sqrt(0.5*nu[1]).real

        # Blending both sub-sampled and synthetic signals
        s0i, n = yi.copy(), s.real.copy()
        s0if = np.fft.fft(s0i)
        nf = np.fft.fft(n)
        s0nf = np.zeros_like(s0if)
        f = np.fft.fftfreq(n.size,1/fs)
        af = np.abs(f)
        idx = af<=fny
        s0nf[idx] = s0if[idx]
        idx = af>fny
        s0nf[idx] = nf[idx]
        s0n += [np.fft.ifft(s0nf).real]

    return np.vstack(s0n).T

def subsampleCFD(cfd_case, fs=2e3,windload_fs=20,
                 s3path='s3://gmto.starccm/StandardYear/20Hz',
                 ForM='FORCES',
                 psd_exp=-13/3,do_nothing=False):
    time_switch = 1800
    case_id = cfd_case.split('_')[1]
    if case_id in list('123')+['17','18','19']:
        time_switch = 4000
    elif case_id in ['13','14']:
        time_switch = 1400
    elif case_id in ['16']:
        time_switch = 2200
    df = pd.read_csv('{0}/{1}/{2}.txt'.format(s3path,cfd_case,ForM))
    df.rename(index=str,
              columns={'Force Truss Top  2  y Monitor: Force (N)':
                       'Force Truss Top 2  y Monitor: Force (N)'},
              inplace=True)
    forces = df[df['Physical Time: Physical Time (s)']>=time_switch]

    subsampled_forces = {}
    for key in forces:
        if not key.startswith('Physical Time'):
            F = forces[key][:,None]
            if do_nothing:
                subsampled_forces[key.replace(' ','_')] = F
            else:
                subsampled_forces[key.replace(' ','_')] = add_colored_noise(F,fs,
                                                                            windload_fs,
                                                                            psd_exp=psd_exp)
    return subsampled_forces

class WindLoad:

    def __init__(self,verbose=logging.INFO,**kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(verbose)
        self.logger.info('Instantiate')
        self.state = {'fs':None,'Time':None, 'Groups':{},'step':0}
        if kwargs:
            self.Start(**kwargs)

    def Start(self,cfd_case, fs=2e3,windload_fs=20,
              s3path='s3://gmto.starccm/Baseline2020',
              groups=['top-end','truss',
                      'GIR','C-Ring',
                      'M1','M2'],
              inputs_version=0,
              M2type='PDR',
              time_range=[],
              last_seconds=None):
        self.logger.info('Start')

        self.state['fs'] = fs
        self.case = cfd_case

        if not (len(groups)==1 and ('M1TransientWind' in groups or 'truss_distributed_forces' in groups)):

            self.logger.info('Loading CFD forces and moments from {0} @ {1}'.format(cfd_case,s3path))

            if time_range:
                df = pd.read_csv('{0}/{1}/FORCES.txt'.format(s3path,cfd_case))
                df.rename(index=str,
                              columns={'Force Truss Top  2  y Monitor: Force (N)':
                                       'Force Truss Top 2  y Monitor: Force (N)'},
                              inplace=True)
                forces = df[df['Physical Time: Physical Time (s)'].between(*time_range)]
                df = pd.read_csv('{0}/{1}/MOMENTS.txt'.format(s3path,cfd_case))
                moments = df[df['Physical Time: Physical Time (s)'].between(*time_range)]
            elif last_seconds is not None:
                df = pd.read_csv('{0}/{1}/FORCES.txt'.format(s3path,cfd_case))
                df.rename(index=str,
                              columns={'Force Truss Top  2  y Monitor: Force (N)':
                                       'Force Truss Top 2  y Monitor: Force (N)'},
                              inplace=True)
                data_time = df['Physical Time: Physical Time (s)']
                time_range = [data_time[-1]-last_seconds,data_time[-1]]
                forces = df[data_time.between(*time_range)]
                df = pd.read_csv('{0}/{1}/MOMENTS.txt'.format(s3path,cfd_case))
                data_time = df['Physical Time: Physical Time (s)']
                l = data_time.values[-1]
                time_range = [l-last_seconds,l]
                moments = df[data_time.between(*time_range)]
            else:
                time_switch = 0#1800
                """
                case_id = cfd_case.split('_')[1]
                if case_id in list('123')+['17','18','19']:
                    time_switch = 4000
                elif case_id in ['13','14']:
                    time_switch = 1400
                elif case_id in ['16']:
                    time_switch = 2200
                """
                df = pd.read_csv('{0}/{1}/FORCES.txt'.format(s3path,cfd_case))
                df.rename(index=str,
                              columns={'Force Truss Top  2  y Monitor: Force (N)':
                                       'Force Truss Top 2  y Monitor: Force (N)'},
                              inplace=True)
                forces = df[df['Physical Time: Physical Time (s)']>=time_switch]
                df = pd.read_csv('{0}/{1}/MOMENTS.txt'.format(s3path,cfd_case))
                moments = df[df['Physical Time: Physical Time (s)']>=time_switch]

            self.logger.info(f'Forces and moments sample # {forces.shape[0]}')

        interp_method = 'linear'

        if groups:
            self.logger.info('wind loading the FEM:')

            if 'top-end' in groups:
                self.logger.info(' . top-end loading: ')
                input = inputs_name['top-end'][inputs_version]
                F = np.vstack([forces['Force Top End {0} Monitor: Force (N)'.format(x)] for x in list('xyz')]).T
                M = np.vstack([moments['Moment Top End {0} Monitor: Moment (N-m)'.format(x)] for x in list('xyz')]).T
                FM_IM = cfd2fem(F,M,FEM_nodes['top-end'])
                self.state['Groups'][input] = {'u':None,'y':None}
                self.state['Groups'][input]['u'] = add_colored_noise(FM_IM,fs,windload_fs)
                self.logger.info("['{}']".format(input))

            if 'mount.top-end' in groups:
                self.logger.info(' . top-end loading: ')
                F = np.vstack([forces['Force Top End {0} Monitor: Force (N)'.format(x)] for x in list('xyz')]).T
                M = np.vstack([moments['Moment Top End {0} Monitor: Moment (N-m)'.format(x)] for x in list('xyz')]).T
                FM_IM = cfd2fem(F,M,FEM_nodes['top-end'])

                self.logger.info(' . M2 cell loading: ')
                F = np.vstack([forces['Force M2_cell {0} Monitor: Force (N)'.format(x)] for x in list('xyz')]).T
                M = np.vstack([moments['Moment M2_cell {0} Monitor: Moment (N-m)'.format(x)] for x in list('xyz')]).T
                FM_IM += cfd2fem(F,M,FEM_nodes['top-end'])

                self.logger.info(' . M2 mirror loading: ')
                for k in range(7):
                    l = (k+1)%7
                    F = np.vstack([forces['Force M2_{1:d} {0} Monitor: Force (N)'.format(x,l)] for x in list('xyz')]).T
                    M = np.vstack([moments['Moment M2_{1:d} {0} Monitor: Moment (N-m)'.format(x,l)] for x in list('xyz')]).T
                    FM_IM += cfd2fem(F,M,FEM_nodes['top-end'])

                input = 'mount.top-end'
                self.state['Groups'][input] = {'u':None,'y':None}
                self.state['Groups'][input]['u'] = add_colored_noise(FM_IM,fs,windload_fs)
                self.logger.info("['{}']".format(input))

            if 'truss' in groups:
                self.logger.info(' . truss loading: ')
                input = inputs_name['truss'][inputs_version]
                F_Truss_top = [np.vstack([forces['Force Truss Top {1:d}  {0} Monitor: Force (N)'.format(x,k)]\
                                  for x in list('xyz')]).T for k in [3,1,2]]
                M_Truss_top = [np.vstack([moments['Moment Truss  Top {1:d} {0} Monitor: Moment (N-m)'.format(x,k)]\
                                  for x in list('xyz')]).T for k in [3,1,2]]
                FM_IM_TRUSS_top = [add_colored_noise(cfd2fem(F,M,R),fs,windload_fs) for F,M,R in zip(F_Truss_top,M_Truss_top,FEM_nodes['truss']['top'])]

                F_Truss_bot = [np.vstack([forces['Force Truss  Bottom {1:d}  {0} Monitor: Force (N)'.format(x,k)]\
                                          for x in list('xyz')]).T for k in [3,1,2]]
                M_Truss_bot = [np.vstack([moments['Moment Truss  Bottom {1:d} {0} Monitor: Moment (N-m)'.format(x,k)]\
                                          for x in list('xyz')]).T for k in [3,1,2]]
                FM_IM_TRUSS_bot = [add_colored_noise(cfd2fem(F,M,R),fs,windload_fs) for F,M,R in zip(F_Truss_bot,M_Truss_bot,FEM_nodes['truss']['bottom'])]

                FM_IM = np.dstack([np.dstack(FM_IM_TRUSS_bot),np.dstack(FM_IM_TRUSS_top)])
                self.state['Groups'][input] = {'u':None,'y':None}
                self.state['Groups'][input]['u'] = FM_IM
                self.logger.info("['{}']".format(input))

            if 'truss_distributed_forces' in groups:
                self.logger.info(' . distributed truss loading: ')
                input = "Truss_distributed_windF"
                s3  = s3fs.S3FileSystem(anon=False)
                key = '{0}/{1}/trussinputs.npz'.format(s3path,cfd_case)
                with s3.open(key,'rb') as f:
                    data = np.load(f)
                    F = data["input"].copy().T
                    t = data["time"].copy()
                if time_range:
                    idx = np.logical_and(t>=time_range[0],t<=time_range[1]);
                    FM_IM =  add_colored_noise(F[idx,:],fs,5)
                else:
                    FM_IM =  add_colored_noise(F,fs,5)
                self.state['Groups'][input] = {'u':None,'y':None}
                self.state['Groups'][input]['u'] = FM_IM
                self.logger.info("['{}']".format(input))

            if 'truss+top-end' in groups:
                self.logger.info(' . truss loading: ')
                input = inputs_name['truss'][inputs_version]
                F_Truss_top = [np.vstack([forces['Force Truss Top {1:d}  {0} Monitor: Force (N)'.format(x,k)]\
                                  for x in list('xyz')]).T for k in [3,1,2]]
                M_Truss_top = [np.vstack([moments['Moment Truss  Top {1:d} {0} Monitor: Moment (N-m)'.format(x,k)]\
                                  for x in list('xyz')]).T for k in [3,1,2]]
                FM_IM_TRUSS_top = [add_colored_noise(cfd2fem(F,M,R),fs,windload_fs) for F,M,R in zip(F_Truss_top,M_Truss_top,FEM_nodes['truss']['top'])]

                F_Truss_bot = [np.vstack([forces['Force Truss  Bottom {1:d}  {0} Monitor: Force (N)'.format(x,k)]\
                                          for x in list('xyz')]).T for k in [3,1,2]]
                M_Truss_bot = [np.vstack([moments['Moment Truss  Bottom {1:d} {0} Monitor: Moment (N-m)'.format(x,k)]\
                                          for x in list('xyz')]).T for k in [3,1,2]]
                FM_IM_TRUSS_bot = [add_colored_noise(cfd2fem(F,M,R),fs,windload_fs) for F,M,R in zip(F_Truss_bot,M_Truss_bot,FEM_nodes['truss']['bottom'])]

                FM_IM_truss = np.dstack([np.dstack(FM_IM_TRUSS_bot),np.dstack(FM_IM_TRUSS_top)])

                self.logger.info(' . top-end loading: ')
                input = inputs_name['top-end'][inputs_version]
                F = np.vstack([forces['Force Top End {0} Monitor: Force (N)'.format(x)] for x in list('xyz')]).T
                M = np.vstack([moments['Moment Top End {0} Monitor: Moment (N-m)'.format(x)] for x in list('xyz')]).T
                FM_IM_topend = add_colored_noise(cfd2fem(F,M,FEM_nodes['top-end']),fs,windload_fs)

                FM_IM = FM_IM_truss.sum(2)+FM_IM_topend
                self.state['Groups'][input] = {'u':None,'y':None}
                self.state['Groups'][input]['u'] = FM_IM
                self.logger.info("['{}']".format(input))

            if 'GIR' in groups:
                self.logger.info(' . GIR loading: ')
                input = inputs_name['GIR'][inputs_version]
                F = np.vstack([forces['Force GIR {0} Monitor: Force (N)'.format(x)] for x in list('xyz')]).T
                M = np.vstack([moments['Moment GIR 1 {0} Monitor: Moment (N-m)'.format(x)] for x in list('xyz')]).T
                FM_IM = cfd2fem(F,M,FEM_nodes['GIR'])
                self.state['Groups'][input] = {'u':None,'y':None}
                self.state['Groups'][input]['u'] = add_colored_noise(FM_IM,fs,windload_fs)
                self.logger.info("['{}']".format(input))

            if 'C-Ring' in groups:
                self.logger.info(' . C-Ring loading: ')
                input = inputs_name['C-Ring'][inputs_version]
                FM_IM = []
                for k,e in enumerate(['','+','-']):
                    F = np.vstack([forces['Force C Ring{1} {0} Monitor: Force (N)'.format(x,e)] for x in list('xyz')]).T
                    M = np.vstack([moments['Moment C Ring{1}  {0} Monitor: Moment (N-m)'.format(x,e)] for x in list('xyz')]).T
                    FM_IM += [add_colored_noise(cfd2fem(F,M,FEM_nodes['C-Ring'][k]),fs,windload_fs)]
                self.state['Groups'][input] = {'u':None,'y':None}
                self.state['Groups'][input]['u'] = np.dstack(FM_IM)
                self.logger.info("['{}']".format(input))

            if 'M1' in groups:
                self.logger.info(' . M1 cell loading: ')
                input = inputs_name['M1']['cell'][inputs_version]
                FM_IM = []
                R1 = Rx(-13.601685)
                for k in range(6):
                    if k<6:
                        R = R1@Rz(k*60)
                    else:
                        R = np.eye(3)
                    F = np.vstack([forces['Force M1_cell {0} Monitor: Force (N)'.format(x)] for x in list('xyz')]).T
                    M = np.vstack([moments['Moment M1_cell {0} Monitor: Moment (N-m)'.format(x)] for x in list('xyz')]).T
                    FM_IM += [add_colored_noise(cfd2fem(F,M,FEM_nodes['M1'][k],R),fs,windload_fs)]
                self.state['Groups'][input] = {'u':None,'y':None}
                self.state['Groups'][input]['u'] = np.dstack(FM_IM+[np.zeros(FM_IM[0].shape)])/6
                self.logger.info("['{}']".format(input))

                self.logger.info(' . M1 mirror loading: ')
                input = inputs_name['M1']['segment'][inputs_version]
                FM_IM = []
                for k in range(7):
                    l = (k+1)%7
                    if k<6:
                        R = R1@Rz(k*60)
                    else:
                        R = np.eye(3)
                    F = np.vstack([forces['Force M1_{1:d} {0} Monitor: Force (N)'.format(x,l)] for x in list('xyz')]).T
                    M = np.vstack([moments['Moment M1_{1:d} {0} Monitor: Moment (N-m)'.format(x,l)] for x in list('xyz')]).T
                    FM_IM += [add_colored_noise(cfd2fem(F,M,FEM_nodes['M1'][k],R),fs,windload_fs)]
                self.state['Groups'][input] = {'u':None,'y':None}
                self.state['Groups'][input]['u'] = np.dstack(FM_IM)
                self.logger.info("['{}']".format(input))

            if 'mount.M1' in groups:
                self.logger.info(' . GIR loading: ')
                F = np.vstack([forces['Force GIR {0} Monitor: Force (N)'.format(x)] for x in list('xyz')]).T
                M = np.vstack([moments['Moment GIR 1 {0} Monitor: Moment (N-m)'.format(x)] for x in list('xyz')]).T
                FM_IM_x = cfd2fem(F,M,[])

                self.logger.info(' . C-Ring loading: ')
                for k,e in enumerate(['','+','-']):
                    F = np.vstack([forces['Force C Ring{1} {0} Monitor: Force (N)'.format(x,e)] for x in list('xyz')]).T
                    M = np.vstack([moments['Moment C Ring{1}  {0} Monitor: Moment (N-m)'.format(x,e)] for x in list('xyz')]).T
                    FM_IM_x += cfd2fem(F,M,[])

                self.logger.info(' . M1 cell loading: ')
                F = np.vstack([forces['Force M1_cell {0} Monitor: Force (N)'.format(x)] for x in list('xyz')]).T
                M = np.vstack([moments['Moment M1_cell {0} Monitor: Moment (N-m)'.format(x)] for x in list('xyz')]).T
                FM_IM_x += cfd2fem(F,M,[])

                FM_IM_x/=7
                F_IM_x = FM_IM_x[:,:3]
                M_IM_x = FM_IM_x[:,3:]

                self.logger.info(' . M1 mirror loading: ')
                FM_IM = []
                R1 = Rx(-13.601685)
                for k in range(7):
                    l = (k+1)%7
                    if k<6:
                        R = R1@Rz(k*60)
                    else:
                        R = np.eye(3)
                    F = np.vstack([forces['Force M1_{1:d} {0} Monitor: Force (N)'.format(x,l)] for x in list('xyz')]).T
                    M = np.vstack([moments['Moment M1_{1:d} {0} Monitor: Moment (N-m)'.format(x,l)] for x in list('xyz')]).T
                    FM_IM += [add_colored_noise(cfd2fem(F+F_IM_x,M+M_IM_x,FEM_nodes['M1'][k],R),fs,windload_fs)]

                input = 'mount.M1'
                self.state['Groups'][input] = {'u':None,'y':None}
                self.state['Groups'][input]['u'] = np.dstack(FM_IM)
                self.logger.info("['{}']".format(input))

            if 'M2' in groups:
                if M2type=='PDR+':
                    self.logger.info(' . M2 cell loading: ')
                    input = inputs_name['M2']['cell'][inputs_version]
                    F = np.vstack([forces['Force M2_cell {0} Monitor: Force (N)'.format(x)] for x in list('xyz')]).T
                    M = np.vstack([moments['Moment M2_cell {0} Monitor: Moment (N-m)'.format(x)] for x in list('xyz')]).T
                    FM_IM = add_colored_noise(cfd2fem(F,M,FEM_nodes['M2 cell']),fs,windload_fs)
                    if inputs_version==2:
                        FM_IM = np.tile(FM_IM,(1,7))/7
                    self.state['Groups'][input] = {'u':None,'y':None}
                    self.state['Groups'][input]['u'] = FM_IM
                    self.logger.info("['{}']".format(input))

                self.logger.info(' . M2 mirror loading: ')
                input = inputs_name['M2']['segment'][inputs_version]
                FM_IM = []
                for k in range(7):
                    l = (k+1)%7
                    if k==0:
                        R = Rx(180-14.777462)
                    elif k==6:
                        R = Rx(180)
                    else:
                        R = Rx(-14.777462)@Rz(-60*k)@Rx(180)
                    F = np.vstack([forces['Force M2_{1:d} {0} Monitor: Force (N)'.format(x,l)] for x in list('xyz')]).T
                    M = np.vstack([moments['Moment M2_{1:d} {0} Monitor: Moment (N-m)'.format(x,l)] for x in list('xyz')]).T
                    FM_IM += [add_colored_noise(cfd2fem(F,M,FEM_nodes['M2'][k],R),fs,windload_fs)]
                self.state['Groups'][input] = {'u':None,'y':None}
                self.state['Groups'][input]['u'] = np.dstack(FM_IM)
                self.logger.info("['{}']".format(input))

            if 'M1TransientWind' in groups:
                self.logger.info(' . M1TransientWind: ')
                input = 'M1_distributed_windF'
                key = '{0}/{1}/M1_distF.mat'.format(s3path,cfd_case)
                s3  = s3fs.S3FileSystem(anon=False)
                with s3.open(key,'rb') as f:
                    data = loadmat(f)
                windload = data['transientWindM1']['signals'][0,0]['values'][0,0]
                t = np.arange(windload.shape[0])/5
                try:
                    data_time = df['Physical Time: Physical Time (s)'].values
                    t = t - t[-1] + data_time[-1]
                except:
                    pass
                if time_range:
                    idx = np.logical_and(t>=time_range[0],t<=time_range[1]);
                    FM_IM = add_colored_noise(windload[idx,:],fs,5)
                else:
                    FM_IM = add_colored_noise(windload,fs,5)
                self.state['Groups'][input] = {'u':None,'y':None}
                self.state['Groups'][input]['u'] = FM_IM
                self.logger.info("['{}']".format(input))

            try:
                self.state['Time'] = np.arange(FM_IM.shape[0])/fs
            except:
                self.state['Time'] = np.arange(FM_IM[0].shape[0])/fs

            self.state['step'] = 0
        return "WindLoad"

    def Init(self):
        pass

    def Update(self,**kwargs):
        self.logger.debug('windload next step #%d',self.state['step'])
        for g in self.state['Groups']:
            self.state['Groups'][g]['y'] = self.state['Groups'][g]['u'][self.state['step'],...].ravel('F')
        self.state['step']+=1

    def Outputs(self,**kwargs):
        """
        if kwargs:
            for g in kwargs:
                kwargs[g][:] = self.state['Groups'][g]['y'][:]
        else:
        """
        return dict(zip(self.state['Groups'].keys(),
                        [self.state['Groups'][x]['y'] for x in self.state['Groups']]))

    def Terminate(self,**kwargs):
        return "Wind loads deleted!"

    def mount_dump(self,filename):
        FMxyz = self.state['Groups']['mount.M1']['u']
        data = []
        for k in range(7):
            for l in range(3):
                data += [FMxyz[:,l,k]]
            for l in range(3):
                data += [FMxyz[:,l+3,k]]

        FMxyz = self.state['Groups']['OSS_Truss_6F']['u']
        truss = ['Lower Truss +Y','Lower Truss +X','Lower Truss -Y'] + \
            ['Upper Truss +Y','Upper Truss +X','Upper Truss -Y']
        for k in range(6):
            for l in range(3):
                data += [FMxyz[:,l,k]]
            for l in range(3):
                data += [FMxyz[:,l+3,k]]

        FMxyz = self.state['Groups']['mount.top-end']['u']
        for l in range(3):
            data += [FMxyz[:,l]]
        for l in range(3):
            data += [FMxyz[:,l+3]]

        data = np.swapaxes(np.vstack(data),1,0)

        m1 = [f"M1S{k}" for  k in range(1,8)]
        fm = list(itertools.chain(*[[_+" Forces [N]"]+[_+" Moments [Nm]"] for _ in m1]))

        truss = ['Lower Truss +Y','Lower Truss +X','Lower Truss -Y'] + \
            ['Upper Truss +Y','Upper Truss +X','Upper Truss -Y']
        fm += list(itertools.chain(*[[_+" Forces [N]"]+[_+" Moments [Nm]"] for _ in truss]))

        fm += ["Top End Forces","Top End Moments"]

        xyz = list('XYZ')
        oss = [f"OSS {_}" for _ in xyz]

        t  = self.state['Time']
        idx = t+400>= self.state['Time'][-1]
        t = t[idx]
        t = t - t[0]

        index = pd.MultiIndex.from_product([fm,oss],names=['Time [s]',''])
        df = pd.DataFrame(data[idx,:], index=t, columns=index)

        try:
            case = self.case.split("_")
            z = int(case[1][:-1])
            lines = [self.case,
                     f"Date: {datetime.date.today()}",
             "Wind time series for the Mount",
            f"External {case[4][:-2]}m/s wind speed",
            f"Telescope azimuth: {case[2][:-2]} degree from wind direction",
            f"Telescope pointed {case[1][:-1]} degrees from zenith",
            ("Enclosure vents "+"open" if case[3]=="os" else "closed") +\
                     " & wind screen "+"stowed" if case[3]=="os" or z==60 else "deployed","\n"]
        except:
            lines = [self.case,
                     f"Date: {datetime.date.today()}",
                     "Wind time series for the Mount","\n"]

        f = open(filename,'w')
        f.write("\n".join(lines))
        df.to_csv(f,float_format="%.2f")
        f.close()


    def m2_dump(self):
        groups = self.state['Groups']
        IMLoads = {
            key: {
                "values": groups[key]["u"].reshape(groups[key]["u"].shape[0],-1),
                "dimensions": np.prod(groups[key]["u"].shape[1:])
            } for key in groups}
        IMLoads.update({"time": self.state["Time"]})
        savemat(self.case+".mat",{"IMLoads":IMLoads})

