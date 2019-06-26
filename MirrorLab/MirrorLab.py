import os
import yaml
import h5py
import numpy as np
import ceo
import matplotlib.pyplot as plt

with open(os.path.join(os.path.dirname(__file__), 'MirrorLabMaps.yaml'),'r') as f:
    maps = yaml.load(f)
print(maps)

datapath = '/home/rconan/Documents/GMT/Notes/M1/M1_polishing_error/Data/2019.06'

def load(filename):
    print('>> loading data from {}'.format(filename))
    data = h5py.File(os.path.join(datapath,filename),'r')
    dataset = data['/dataset']
    Smap = dataset.value*1e-6
    pixelSize = (dataset.attrs.get('pixelSize'))[0]
    centerRow = (dataset.attrs.get('centerRow'))[0]
    centerCol = (dataset.attrs.get('centerCol'))[0]
    return (Smap,pixelSize,centerRow,centerCol)

def info(filename):
    Smap,pixelSize,centerRow,centerCol = load(filename)
    print('Map: {}'.format(filename))
    print(' * shape:', Smap.shape)
    print(' * sampling: {:f}m'.format(pixelSize))
    print(' * mean: {:.0f}nm'.format(1e9*np.nanmean(Smap)))
    print(' * std : {:.0f}nm'.format(1e9*np.nanstd(Smap)))

def show(filename):
    Smap,pixelSize,centerRow,centerCol = load(filename)
    n,m = Smap.shape
    L = pixelSize*n
    extent = np.array([-1,1,-1,1])*L/2
    fig,ax = plt.subplots()
    h = ax.imshow(Smap*1e9,extent=extent)
    plt.colorbar(h,ax=ax)
    plt.show()

def PSSn(filename):
    Smap,pixelSize,centerRow,centerCol = load(filename)

    print('>> prep data for PSSn estimate')
    [rows, cols] = np.shape(Smap)
    xVec = np.linspace(1,cols,cols)
    xVec = (xVec - centerCol) * pixelSize # m
    yVec = np.linspace(1,rows,rows)
    yVec = (yVec - centerRow) * pixelSize # m, increasing upward
    [x,y] = np.meshgrid(xVec,yVec)
    xy = np.vstack([x.flatten(),y.flatten()]).T
    z = Smap.reshape(-1,1)
    S = ceo.Mapping(xy,z)
    S = S(rows,pixelSize*rows)
    S.dump('.segmentmaps')

    print('>> PPSn estimation')
    N = int(25.5/pixelSize)
    gmt = ceo.GMT_MX(M1_mirror_modes='.segmentmaps',M1_N_MODE=1)
    src = ceo.Source('Vs',zenith=0,azimuth=0,
                    rays_box_sampling=N,rays_box_size=25.5,
                    rays_origin=[0,0,25])
    srcH = ceo.Source('H',zenith=0,azimuth=0,
                    rays_box_sampling=N,rays_box_size=25.5,
                    rays_origin=[0,0,25])
    src>>(gmt,)
    srcH>>(gmt,)
    +src
    print("Ideal WFE rms [nm]:",src.wavefront.rms(-9))

    ~gmt
    gmt.M1.modes.a[:,0] = 1
    gmt.M1.modes.update()
    +src
    +srcH

    fig,ax=plt.subplots()
    fig.set_size_inches(10,10)
    h=ax.imshow(src.phase.host(units='nm'))
    plt.colorbar(h,label='WFE [nm]')

    print("Segment polishing WFE rms {}[nm]".format(np.array_str(src.phaseRms(where='segments',units_exponent=-9),precision=0)))
    print("Whole polishing WFE rms {:.0f}[nm]".format(src.wavefront.rms(-9)[0]))

    print("V & H PSSn:",gmt.PSSn(src),gmt.PSSn(srcH))
    plt.show()
