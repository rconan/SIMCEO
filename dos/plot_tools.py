#import pickle
import  numpy as np
import matplotlib.pyplot as plt
#from scipy import sparse
#from scipy.linalg import block_diag

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

    print('Final values:\n WFE:',wfe_rms[1][-1]*1.0e9,
          '\n', seg_wfe_rms[1][-1,:]*1.0e9,
          '\nPSSn:',pssn[1][-1], 
          '\nsum of abs seg piston:',np.sum(np.abs(segpiston[1][-1]))*1.0e6,
          '\n', segpiston[1][-1,:]*1.0e6)

    plt.figure(figsize=(18,10))
    plt.subplot(521)
    plt.plot(wfe_rms[0],wfe_rms[1]*1.0e9,'x--')
    plt.grid(True)
    plt.ylabel('WFE RMS')

    plt.subplot(522)
    plt.plot(wfe_rms[0],seg_wfe_rms[1]*1.0e9,'x--')
    plt.grid(True)
    plt.ylabel('WFE RMS')

    plt.subplot(523)
    plt.plot(pssn[0],pssn[1],'x--')
    plt.grid(True)
    plt.ylabel('PSSn')

    plt.subplot(525)
    plt.plot(segpiston[0],np.sum(np.abs(segpiston[1]),axis=1)*1.0e6,'x--')
    plt.grid(True)
    plt.ylabel('sum of abs Seg piston')

    plt.subplot(526)
    plt.plot(segpiston[0],segpiston[1]*1.0e6,'x--')
    plt.grid(True)
    plt.ylabel('Segment piston') 

    try:
        plt.subplot(527)
        plt.plot(segpiston[0],np.sum(np.abs(ESdeltas[1]),axis=1)*1.0e6,'x--')
        plt.grid(True)
        plt.ylabel('sum of abs ES deltas') 

        plt.subplot(528)
        plt.plot(segpiston[0],ESdeltas[1],'x--')
        plt.grid(True)
        plt.ylabel('ES deltas') 
    except:
        print('')

    plt.subplot(529)
    plt.plot(tt[0],tt[1],'x--')
    plt.grid(True)
    plt.ylabel('TT') 

    plt.subplot(5,2,10)
    plt.plot(segtt[0],segtt[1],'x--')
    plt.grid(True)
    plt.ylabel('Seg TT')


def plot_X0loadComp(m1_x0_dt, m2_x0_dt, controllerDT, show_delta, colors, markers):
    M1Txyz = controllerDT['M1 Txyz'].timeSeries
    M1Rxyz = controllerDT['M1 Rxyz'].timeSeries
    M2Txyz = controllerDT['M2 Txyz'].timeSeries
    M2Rxyz = controllerDT['M2 Rxyz'].timeSeries
    M1BM = controllerDT['M1 BM'].timeSeries

    deltaM1Txyz, deltaM1Rxyz = 1.0*np.zeros_like(M1Txyz[1]), 1.0*np.zeros_like(M1Rxyz[1])
    deltaM2Txyz, deltaM2Rxyz = 1.0*np.zeros_like(M2Txyz[1]), 1.0*np.zeros_like(M2Rxyz[1])
    deltaBM = 1.0*np.zeros_like(M1BM[1])

    aux_plt = M1Txyz[0][-1] + 4
    Txyz_scale = 1.0e6 # micro meters
    asec2rad = 4.84814e-6
    
    plt.figure(figsize=(16,10))
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
                plt.plot(aux_plt, -np.array(m1_x0_dt['state']['Rxyz'])[kseg,kmode]/asec2rad,'-',
                    color=colors[kseg], marker=markers[kmode])    
            plt.plot(deltaM1Rxyz[kseg,kmode,:]/asec2rad,'--', color=colors[kseg], marker=markers[kmode])
    plt.grid(True)
    plt.ylabel('M1 Rxyz [asec]')

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
                plt.plot(aux_plt, -np.array(m2_x0_dt['state']['Rxyz'])[kseg,kmode]/asec2rad,'-',
                    color=colors[kseg], marker=markers[kmode])    
            plt.plot(deltaM2Rxyz[kseg,kmode,:]/asec2rad,'--', color=colors[kseg], marker=markers[kmode])
    plt.grid(True)
    plt.ylabel('M2 Rxyz [asec]')
    plt.show()

    plt.figure(figsize=(16,5))
    for kmode in range(len(M1BM[1][0,:,0])):
        for kseg in range(7):
            deltaBM[kseg,kmode,:] = M1BM[1][kseg,kmode,:]
            if(show_delta):
                deltaBM[kseg,kmode,:] = deltaBM[kseg,kmode,:] + np.array(m1_x0_dt['state']['modes'])[kseg,kmode]
            else:
                plt.plot(aux_plt, -np.array(m1_x0_dt['state']['modes'])[kseg,kmode],'.',
                    color=colors[kseg])    
            plt.plot(deltaBM[kseg,kmode,:],'--', color=colors[kseg])
    plt.grid(True)
    plt.ylabel('u: M1 BM cmd')


def plot_states(X_timeseries, n_bm):

    isample = X_timeseries[0]
    # Indices for M1 states
    indX = np.reshape(np.arange(X_timeseries[1].shape[1]), [7, 6 + 6 + n_bm])
    i1,i2,i3,i4,i5 = np.split(indX,[3, 6, 9, 12], axis=1)

    xM1_Txyz = X_timeseries[1][:,np.reshape(i1,[21])]
    xM1_Rxyz = X_timeseries[1][:,np.reshape(i2,[21])]
    xM2_Txyz = X_timeseries[1][:,np.reshape(i3,[21])]
    xM2_Rxyz = X_timeseries[1][:,np.reshape(i4,[21])]
    xM1_BM = X_timeseries[1][:,np.reshape(i5,[46*7])]

    plt.figure(figsize=(16,10))
    plt.subplot(221)
    plt.plot(isample,xM1_Txyz*1.0e6,'x--')
    plt.grid(True)
    plt.ylabel('M1 Txyz states')

    plt.subplot(222)
    plt.plot(isample,xM1_Rxyz,'x--')
    plt.grid(True)
    plt.ylabel('M1 Rxyz states')    

    plt.subplot(223)
    plt.plot(isample,xM2_Txyz*1.0e6,'x--')
    plt.grid(True)
    plt.ylabel('M2 Txyz states')

    plt.subplot(224)
    plt.plot(isample,xM2_Rxyz,'x--')
    plt.grid(True)
    plt.ylabel('M2 Rxyz states')   
    plt.show()

    plt.figure(figsize=(16,5))
    plt.plot(isample,xM1_BM,'x--')
    plt.grid(True)
    plt.ylabel('M1 BM states')
    plt.show()    