#import pickle
import  numpy as np
import matplotlib.pyplot as plt
#from scipy import sparse
#from scipy.linalg import block_diag

asec2rad = 4.84814e-6
Txyz_scale = 1.0e6 # micro meters
Rxyz_scale = 1.0e3 # m rad

def plot_science(scienceDT, m1EsDT=[]):
    wfe_rms = scienceDT['wfe_rms'].timeSeries
    seg_wfe_rms = scienceDT['segment_wfe_rms'].timeSeries
    segpiston = scienceDT['segment_piston'].timeSeries
    tt = scienceDT['tiptilt'].timeSeries
    segtt = scienceDT['segment_tiptilt'].timeSeries
    pssn = scienceDT['PSSn'].timeSeries
    try:
        ESdeltas = m1EsDT['deltas'].timeSeries
    except:
        print('')    

    print('Final values:\n WFE (nm):',wfe_rms[1][-1]*1.0e9,
          '\n', seg_wfe_rms[1][-1,:]*1.0e9,
          '\nPSSn:',pssn[1][-1], 
          '\nsum of abs seg piston (um):',np.sum(np.abs(segpiston[1][-1]))*1.0e6,
          '\n', segpiston[1][-1,:]*1.0e6)

    plt.figure(figsize=(18,6))
    plt.subplot(321)
    plt.plot(wfe_rms[0],wfe_rms[1]*1.0e9,'x--')
    plt.grid(True)
    plt.ylabel('WFE RMS (nm)')

    plt.subplot(322)
    plt.plot(wfe_rms[0],seg_wfe_rms[1]*1.0e9,'x--')
    plt.grid(True)
    plt.ylabel('WFE RMS (nm)')

    plt.subplot(323)
    plt.plot(pssn[0],pssn[1],'x--')
    plt.grid(True)
    plt.ylabel('PSSn')

    plt.subplot(325)
    plt.plot(segpiston[0],np.sum(np.abs(segpiston[1]),axis=1)*1.0e6,'x--')
    plt.grid(True)
    plt.ylabel('sum of abs Seg piston (um)')

    plt.subplot(326)
    plt.plot(segpiston[0],segpiston[1]*1.0e6,'x--')
    plt.grid(True)
    plt.ylabel('Segment piston (um)') 

    try:
        ESdeltas[0].shape
        plt.figure(figsize=(18,2))
        plt.subplot(221)
        plt.plot(segpiston[0],np.sum(np.abs(ESdeltas[1]),axis=1)*1.0e6,'x--')
        plt.grid(True)
        plt.ylabel('sum of abs ES deltas') 

        plt.subplot(222)
        plt.plot(segpiston[0],ESdeltas[1],'x--')
        plt.grid(True)
        plt.ylabel('ES deltas')
    except:
        pass

    plt.figure(figsize=(18,2))
    plt.subplot(121)
    plt.plot(tt[0],tt[1],'x--')
    plt.grid(True)
    plt.ylabel('TT') 

    plt.subplot(122)
    plt.plot(segtt[0],segtt[1][:,:7],'x--')
    plt.plot(segtt[0],segtt[1][:,7:],'+--')
    plt.grid(True)
    plt.ylabel('Seg TT')



def plot_X0loadComp(m1_x0_dt, m2_x0_dt, controllerDT, show_delta, colors, markers):
    rm_samples = 0
    M1Txyz, M1Rxyz, M2Txyz, M2Rxyz, M1BM = {}, {}, {}, {}, {}
    M1Txyz_ = controllerDT['M1 Txyz'].timeSeries
    M1Txyz[0] = np.delete(M1Txyz_[0],np.arange(rm_samples))
    M1Txyz[1] = np.delete(M1Txyz_[1],np.arange(rm_samples),axis=2)
    M1Rxyz_ = controllerDT['M1 Rxyz'].timeSeries
    M1Rxyz[1] = np.delete(M1Rxyz_[1],np.arange(rm_samples),axis=2)
    M2Txyz_ = controllerDT['M2 Txyz'].timeSeries
    M2Txyz[1] = np.delete(M2Txyz_[1],np.arange(rm_samples),axis=2)
    M2Rxyz_ = controllerDT['M2 Rxyz'].timeSeries
    M2Rxyz[1] = np.delete(M2Rxyz_[1],np.arange(rm_samples),axis=2)
    M1BM_ = controllerDT['M1 BM'].timeSeries
    M1BM[1] = np.delete(M1BM_[1],np.arange(rm_samples),axis=2)

    deltaM1Txyz, deltaM1Rxyz = 1.0*np.zeros_like(M1Txyz[1]), 1.0*np.zeros_like(M1Rxyz[1])
    deltaM2Txyz, deltaM2Rxyz = 1.0*np.zeros_like(M2Txyz[1]), 1.0*np.zeros_like(M2Rxyz[1])
    deltaBM = 1.0*np.zeros_like(M1BM[1])

    aux_plt = M1Txyz[0][-1] + 4
    
    plt.figure(figsize=(16,8))
    plt.subplot(221)
    for kmode in range(3):
        for kseg in range(7):            
            deltaM1Txyz[kseg,kmode,:] = M1Txyz[1][kseg,kmode,:]
            if(show_delta):
                deltaM1Txyz[kseg,kmode,:] = deltaM1Txyz[kseg,kmode,:] + np.array(m1_x0_dt['state']['Txyz'])[kseg,kmode]
            else:
                plt.plot(aux_plt, -np.array(m1_x0_dt['state']['Txyz'])[kseg,kmode]*Txyz_scale,'-',
                    color=colors[kseg], marker=markers[kmode])    
            plt.plot(deltaM1Txyz[kseg,kmode,:]*Txyz_scale,'--', color=colors[kseg], marker=markers[kmode])
    plt.grid(True)
    plt.ylabel('M1 Txyz [um]')
    
    plt.subplot(222)
    for kmode in range(3):
        for kseg in range(7):
            deltaM1Rxyz[kseg,kmode,:] = M1Rxyz[1][kseg,kmode,:]
            if(show_delta):
                deltaM1Rxyz[kseg,kmode,:] = deltaM1Rxyz[kseg,kmode,:] + np.array(m1_x0_dt['state']['Rxyz'])[kseg,kmode]
            else:
                plt.plot(aux_plt, -np.array(m1_x0_dt['state']['Rxyz'])[kseg,kmode]*Rxyz_scale,'-',
                    color=colors[kseg], marker=markers[kmode])
            if(not kseg == 6) or (not kmode == 2):
                plt.plot(deltaM1Rxyz[kseg,kmode,:]*Rxyz_scale,'--', color=colors[kseg], marker=markers[kmode])
    plt.grid(True)
    plt.ylabel('M1 Rxyz [mrad]')

    plt.subplot(223)
    for kmode in range(3):
        for kseg in range(7):
            deltaM2Txyz[kseg,kmode,:] = M2Txyz[1][kseg,kmode,:]
            if(show_delta):
                deltaM2Txyz[kseg,kmode,:] = deltaM2Txyz[kseg,kmode,:] + np.array(m2_x0_dt['state']['Txyz'])[kseg,kmode]
            else:
                plt.plot(aux_plt, -np.array(m2_x0_dt['state']['Txyz'])[kseg,kmode]*Txyz_scale,'-',
                    color=colors[kseg], marker=markers[kmode])    
            plt.plot(deltaM2Txyz[kseg,kmode,:]*Txyz_scale,'--', color=colors[kseg], marker=markers[kmode])
    plt.grid(True)
    plt.ylabel('M2 Txyz [um]')

    plt.subplot(224)
    for kmode in range(3):
        for kseg in range(7):
            deltaM2Rxyz[kseg,kmode,:] = M2Rxyz[1][kseg,kmode,:]
            if(show_delta):
                deltaM2Rxyz[kseg,kmode,:] = deltaM2Rxyz[kseg,kmode,:] + np.array(m2_x0_dt['state']['Rxyz'])[kseg,kmode]
            else:
                plt.plot(aux_plt, -np.array(m2_x0_dt['state']['Rxyz'])[kseg,kmode]*Rxyz_scale,'-',
                    color=colors[kseg], marker=markers[kmode])
            if(not kseg == 6) or (not kmode == 2):    
                plt.plot(deltaM2Rxyz[kseg,kmode,:]*Rxyz_scale,'--', color=colors[kseg], marker=markers[kmode])
    plt.grid(True)
    plt.ylabel('M2 Rxyz [mrad]')
    plt.show()

    plt.figure(figsize=(16,4))
    for kmode in range(len(M1BM[1][0,:,0])):
        for kseg in range(7):
            deltaBM[kseg,kmode,:] = M1BM[1][kseg,kmode,:]
            if(show_delta):
                deltaBM[kseg,kmode,:] = deltaBM[kseg,kmode,:] + np.array(m1_x0_dt['state']['modes'])[kseg,kmode]
            else:
                plt.plot(aux_plt, -np.array(m1_x0_dt['state']['modes'])[kseg,kmode],'.',
                    color=colors[kseg])    
            plt.plot(deltaBM[kseg,kmode,:],'.--', color=colors[kseg])
    plt.grid(True)
    plt.ylabel('u: M1 BM cmd')


def plot_states(X_timeseries, n_bm, colors, markers):

    isample = X_timeseries[0]
    # Indices for M1 states
    indX = np.reshape(np.arange(X_timeseries[1].shape[1]), [7, 6 + 6 + n_bm])
    [i1,i2,i3,i4,i5] = np.split(indX,[3, 6, 9, 12], axis=1)

    xM1_Txyz = X_timeseries[1][:,np.reshape(i1,[7,3])]
    xM1_Rxyz = X_timeseries[1][:,np.reshape(i2,[7,3])]
    xM2_Txyz = X_timeseries[1][:,np.reshape(i3,[7,3])]
    xM2_Rxyz = X_timeseries[1][:,np.reshape(i4,[7,3])]
    xM1_BM = X_timeseries[1][:,np.reshape(i5,[7,n_bm])]
    

    plt.figure(figsize=(16,8))

    plt.subplot(221)
    for kmode in range(3):
        for kseg in range(7):
            plt.plot(isample,xM1_Txyz[:,kseg,kmode]*Txyz_scale,'--', color=colors[kseg], marker=markers[kmode])
    plt.grid(True)
    plt.ylabel('M1 Txyz states [um]')

    plt.subplot(222)
    for kmode in range(3):
        for kseg in range(7):
            plt.plot(isample,xM1_Rxyz[:,kseg,kmode]*Rxyz_scale,'--', color=colors[kseg], marker=markers[kmode])
    plt.grid(True)
    plt.ylabel('M1 Rxyz states [mrad]')    

    plt.subplot(223)
    for kmode in range(3):
        for kseg in range(7):
            plt.plot(isample,xM2_Txyz[:,kseg,kmode]*Txyz_scale,'--', color=colors[kseg], marker=markers[kmode])
    plt.grid(True)
    plt.ylabel('M2 Txyz states [um]')

    plt.subplot(224)
    for kmode in range(3):
        for kseg in range(7):
            plt.plot(isample,xM2_Rxyz[:,kseg,kmode]*Rxyz_scale,'--', color=colors[kseg], marker=markers[kmode])
    plt.grid(True)
    plt.ylabel('M2 Rxyz states [mrad]')   
    plt.show()

    plt.figure(figsize=(16,4))
    for kmode in range(n_bm):
        for kseg in range(7):
            plt.plot(isample,xM1_BM[:,kseg,kmode],'.--', color=colors[kseg])
    plt.grid(True)
    plt.ylabel('M1 BM states')
    plt.show()    


# def plot_m1Forces(controllerDT, show_delta, colors):
#     M1BM = controllerDT['M1 BM'].timeSeries    

#     aux_plt = M1BM[0][-1] + 4
    
#     plt.figure(figsize=(16,4))
#     for kmode in range(len(M1BM[1][0,:,0])):
#         for kseg in range(7):
#             deltaBM[kseg,kmode,:] = M1BM[1][kseg,kmode,:]
#             if(show_delta):
#                 deltaBM[kseg,kmode,:] = deltaBM[kseg,kmode,:] + np.array(m1_x0_dt['state']['modes'])[kseg,kmode]
#             else:
#                 plt.plot(aux_plt, -np.array(m1_x0_dt['state']['modes'])[kseg,kmode],'.',
#                     color=colors[kseg])    
#             plt.plot(deltaBM[kseg,kmode,:],'.--', color=colors[kseg])
#     plt.grid(True)
#     plt.ylabel('u: M1 BM cmd')
