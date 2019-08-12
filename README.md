# DOS configuration file

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

## Optical path driver

An optical path is composed of a source and a wavefront sensor.
The optical path will be the atmosphere if it is specified, the GMT and the wavefront sensor.
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
## Controller driver

A controller can be placed between a sensor and a mirror.
The inputs are given as a list with the sensor section name and sensor output name.
The outputs are given as a list with the mirror name and mirror input name.

```yaml
driver:
  server: false
  inputs:
    input data:
      lien: [sensor,output]
  outputs:
    output data:
      lien: [mirror,dof]
```
