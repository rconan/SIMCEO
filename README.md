# SIMCEO

**SIMCEO** is a client/server application to run Dynamic Optical Simulations (**DOS**) for the GMT.
The server acts as a remote interface for [`CEO`](https://github.com/rconan/CEO).

A simulation takes as parameter the path to the directory that contains the simulation configuration and parameter files.
A new simulation is invoked first by starting the server:
```python
>>> python simceo.py
```
Then, in a different terminal, the client is invoked with:
```python
>>> import dos
>>> sim = dos.DOS(path_to_config_dir)
>>> sim.start()
```
An any time, the status of the simulation can be checked by calling:
```python
>>> sim.pctComplete
```
that return the simulation percentage to completion.

Or by checking the logs:
```python
>>> sim.logs
```

# Install

SIMCEO server and client along with the reference manual are contained in the `simceo.nw` file.
The server is extracted with `make server`.
The client module `dos` is build with `make client`.
`noweb` needs to be installed to build both.

Alternatively, you can download the latest [release](https://github.com/rconan/SIMCEO/releases) DOS-x.x.tar.gz where the source files have already been generated.
To install SIMCEO, perfom the following steps:
```shell
tar zxf DOS-x.x.tar.gz
cd DOS-x.x
python setup.py install
```

# DOS configuration

Each simulation is defined by a set of YAML files gather in the same directory.
The simulation configuration is always given by the `dos.yaml` file.
Some parameters are given a default value.
If the simulation use the same values, they need not to be set.

```yaml
simulation:
  sampling frequency: # [Hertz]
  duration: # [seconds]
  server:
    IP: # 127.0.0.1
drivers:
  device name:
    server: true
    delay: 0 # [sample]
    sampling rate: 1 # [sample]
    inputs:
      input name:
        size: 0
        lien: # [device, device output name]
    outputs:
      output name:
        sampling rate: 1 # [sample]
        size: 0
        lien: # [device, device input name]
```

## Atmosphere driver

If an atmosphere need to be included in the simulation, then the atmosphere driver must be the first in the list of drivers and the atmosphere parameters are listed in the `atmosphere.yaml` file.

```yaml
drivers:
  atmosphere: {}
```
The `atmosphere.yaml` file sets the main parameters of the atmosphere.
The atmospheric Cn2 and wind vector profiles correspond to the GMT median profile from Goodwin data.
```yaml
# The Fried parameter [m]
r0: 0.15
# The outer scale [m]
L0: 30.0
# The size of the phase screen at the ground [m]
L: 25.5
# The resolution of the phase screen at the ground [pixel]
NXY_PUPIL: 321
# The field of view that defines the size of the phase screen above ground [arcsec]
fov: 0.0
# The time span of the phase screen [second]
duration: 10
# The filename where to store the phase screens, if the file exist it is loaded automatically
filename: phase_screen_meaningfull_name.bin
```

## Mirror drivers

The section of a mirror driver is either `M1` or `M2` and the parameters of a mirror are in either `M1.yaml` or `M2.yaml`.
There are only inputs to the mirrors:

 * the 6 rigid body modes of each segment: `TxyzRxyz`
 * the tip and tilt of each segment: `Rxy`
 * the mirror mode shape coefficients: `mode_coefs`

```yaml
drivers:
  M1:
    inputs:
      TxyzRxyz:
        size: [7,6]
      Rxy:
        size: [7,2]
      mode_coefs:
        size: [7,n_mode]
  M2:
    inputs:
      TxyzRxyz:
        size: [7,6]
      Rxy:
        size: [7,2]
      mode_coefs:
        size: [7,n_mode]
```
Both `M1.yaml` and `M2.yaml` have the same syntax.
If the degrees of freedom are only the rigid body motions, then the file is:
```yaml
mirror: # M1 or M2
mirror_args: {}
```

if the mirror modes are included, then the file becomes:
```yaml
mirror: # M1 or M2
mirror_args:
  M1_mirror_modes: # zernike ,Karhunen-Loeve or bending modes
  M2_mirror_modes: # zernike or Karhunen-Loeve
  M1_N_MODE: # The number of modes
  M2_N_MODE: # The number of modes
```

## Optical path driver

An optical path is composed of a source and a wavefront sensor.
The optical path consists in the atmosphere if it is specified, the GMT and the wavefront sensor.
The optical path section of a driver has only outputs.
If an interaction matrix is made between the WFS and some degrees of freedom of M1 and/or M2, the name of one output must match the name given to the interaction matrix.
There are a few outputs that provides wavefront and image quality information:

 * `wfe_rms`: the wavefront standart deviation [micron]
 * `piston`: the piston over the whole GMT pupil [micron]
 * `segment_piston`: the piston associated to each segment [micron]
 * `tiptilt`: the wavefront finite difference average over the whole GMT pupil [arcsec]
 * `segment_tiptilt`: the wavefront finite difference average over each segment [arcsec]
 * `PSSn`: the normalized point source sensitivity

```yaml
driver:
  name:
    outputs:
      calibration name:
        size: [n,m]
      wfe_rms: # micron
        size: 1
      piston: # micron
        size: 1
      segment_piston: # micron
        size: 7
      tiptilt: # arcsec
        size: 2
      segment_tiptilt: # arcsec
        size: 14
      PSSn:
        size:1
```

The parameter file contains the source, sensor and calibration parameters.
```yaml
source:
  photometric_band: # the source wavelength, one of V,R,I,J,K
  zenith: # the source zenith angle [rd]
  azimuth: # the source azimuth angle [rd]
# An alternative syntax for zenith and azimuth is:
  zenith:
    value: # the source zenith angle
    units: # the angle units, degree, arcmin, arcsec or mas
  azimuth:
    value: # the source azimuth angle
    units: # the angle units, degree, arcmin, arcsec or mas
  magnitude: # the star magnitude
  rays_box_size: # the size of the square area sampled by ray tracing
  rays_box_sampling: # the resolution of one side of the ray tracing square sample [N_SIDE_LENSLET*N_PX_LENSLET+1]
  rays_origin: # A 3 element list [x,y,z] where x and y are the coordinates of the chief ray intersection
               # with the entrance pupil (M1) and z is the altitude above the telescope
               # where ray tracing starts [0,0,25]
sensor:
  class: # GeometricShackHartmann, ShackHartmann, Pyramid
  args:
    N_SIDE_LENSLET: # the size of the lenslet array
    N_PX_LENSLET: # the number of pixel per lenslet
    d: # the lenslet array pitch [rays_box_size/N_SIDE_LENSLET]
  calibrate_args:
    threshold: # the lenslet illumination threshold (this apply solely to Shack-Hartmann sensors)
    ...: ... # arguments for the sensor calibrate method
interaction matrices: 
  calibrations:
    calibration name:  
      method_id: # the calibration method: calibrate, AGWS_calibrate, NGAO_calibrate
      args: # the calibration method arguments
        mirror: #
        mode: #
        stroke: #
        ...
  filename: # the filename where the calibration matrix is stored, if it exist it is loaded automatically
```


## Controller driver

A controller can be placed between a sensor and a mirror.
The controller runs on the client side of the application, `server` must then be set to `false`.
The inputs are given as a list with the sensor section name and sensor output name.
The outputs are given as a list with the mirror name and mirror input name.

```yaml
driver:
  ctrlr name:
    server: false
    inputs:
      input data:
        lien: [sensor,output]
    outputs:
      output data:
        lien: [mirror,dof]
```

The `ctrlr name.yaml` file contains the controller parameter as

 * the numerator and denominator coefficients of the transfer function in the Z domain [n,d]
 * the zeros, poles and gains of the transfer function [z,p,g]
 * a state space representation [A,B,C,D]
 
```yaml
system:
  sampling time: # The sampling time [second]
  transfer function:
    num: [...] 
    denom: [...]
```
