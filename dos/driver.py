from scipy import signal
import numpy as np
import yaml
import logging
logging.basicConfig()
class IO:
    def __init__(self,tag,size=0, lien=None, logs=None):
        self.logger = logging.getLogger(tag)
        self.size = size
        self.data = np.zeros(size)
        self.lien = lien
        self.logs = logs
class Input(IO):
    def __init__(self,*args,**kwargs):
        IO.__init__(self,*args,**kwargs)
    def tie(self,drivers):
        if self.lien is not None:
            d,io = self.lien
            self.logger.info('Linked to %s from %s',io,d)
            self.data = drivers[d].outputs[io].data
            self.size = self.data.shape
class Output(IO):
    def __init__(self,*args,sampling_rate=1,**kwargs):
        IO.__init__(self,*args,**kwargs)
        self.sampling_rate = sampling_rate
    def tie(self,drivers):
        if self.lien is not None:
            d,io = self.lien
            self.logger.info('Linked to %s from %s',io,d)
            self.data = drivers[d].inputs[io].data
            self.size = self.data.shape
class Driver:
    def __init__(self,tau,tag):
        self.tau = tau
        self.tag = tag
    def start(self):
        pass
    def init(self):
        pass
    def update(self,_):
        pass
    def output(self,_):
        pass
    def terminate(self):
        pass
class Server(Driver):
    def __init__(self,tau,tag,logs,server,delay=0,sampling_rate=1,
             verbose=logging.INFO,**kwargs):
        Driver.__init__(self,tau,tag)
        self.logger = logging.getLogger(tag)
        self.logger.setLevel(verbose)
        self.delay         = delay
        self.sampling_rate = sampling_rate
        self.inputs        = {}
        if 'inputs' in kwargs:
            for k,v in kwargs['inputs'].items():
                self.logger.info('New input: %s',k)
                self.inputs[k] = Input(k,**v)
        self.outputs       = {}
        if 'outputs' in kwargs:
            for k,v in kwargs['outputs'].items():
                self.logger.info('New output: %s',k)
                if 'logs' in v:
                    logs.add(tag,k,v['logs']['decimation'])
                    v['logs'] = logs.entries[tag][k]
                    self.logger.info('Output logged in!')
                self.outputs[k] = Output(k,**v)
        self.server        = server
        self.msg = {'class_id':'',
                    'method_id':'',
                    'args':{}}
        self.msg_args = {'Start':{},
                    'Init':{},
                    'Update':{'inputs':{}},
                    'Outputs':{'outputs':[]},
                    'Terminate':{'args':None}}

    def start(self):
        self.logger.debug('Starting!')
        m = 'Start'
        self.msg['method_id'] = m
        self.msg['args'].update(self.msg_args[m])
        self.server._send_(self.msg)
        self.msg['method_id'] = ''
        self.msg['args'].clear()
        reply = self.server._recv_()
        self.msg['class_id'] = reply
        self.logger.info('%s',reply)
    def init(self):
        self.logger.debug('Initializing!')
        m = 'Init'
        self.msg['method_id'] = m
        self.msg['args'].update(self.msg_args[m])
        self.server._send_(self.msg)
        self.msg['method_id'] = ''
        self.msg['args'].clear()
        reply = self.server._recv_()
        self.logger.info('%s',reply)
    def update(self,step):
        if step>=self.delay and step%self.sampling_rate==0:
            self.logger.debug('Updating!')
            m = 'Update'
            self.msg['method_id'] = m
            self.msg['args'].update(self.msg_args[m])
            self.server._send_(self.msg)
            self.msg['method_id'] = ''
            self.msg['args'].clear()
            reply = self.server._recv_()
    def output(self,step):
        if step>=self.delay:
                m = 'Outputs'
                if self.msg_args[m]['outputs']:
                    self.msg['method_id'] = m
                    self.msg['args'].update(self.msg_args[m])
                    self.server._send_(self.msg)
                    self.msg['method_id'] = ''
                    self.msg['args'].clear()
                    reply = self.server._recv_()
                    self.logger.debug("Reply: %s",reply)
                    for k,v in self.outputs.items():
                        if step%v.sampling_rate==0:
                            self.logger.debug('Outputing %s!',k)
                            try:
                                v.data[...] = np.asarray(reply[k]).reshape(v.size)
                            except ValueError:
                                self.logger.warning('Resizing %s!',k)
                                __red = np.asarray(reply[k])
                                v.size = __red.size
                                v.data = np.zeros(__red.shape)
                                v.data[...] = __red
                            if v.logs is not None and step%v.logs.decimation==0:
                                self.logger.debug('LOGGING')
                                v.logs.add(v.data.copy())

    def terminate(self):
        self.logger.debug('Terminating!')
        m = 'Terminate'
        self.msg['method_id'] = m
        self.msg['args'].update(self.msg_args[m])
        self.server._send_(self.msg)
        self.msg['method_id'] = ''
        self.msg['args'].clear()
        reply = self.server._recv_()
        self.logger.info(reply)

    def associate(self,prm_file):
        base_units = np.pi/180
        units = {'degree': base_units,
                 'arcmin': base_units/60,
                 'arcsec': base_units/60/60,
                 'mas': base_units/60/60/1e3}
        with open(prm_file) as f:
            prm = yaml.load(f)
        if 'mirror' in prm:
            self.msg['class_id'] = 'GMT'
            self.msg_args['Start'].update(prm)
            self.msg_args['Update']['mirror'] = prm['mirror']
            self.msg_args['Update']['inputs'].update(\
                    {k_i:v_i.data for k_i,v_i in self.inputs.items()})
        else:
            self.msg['class_id'] = 'OP'
            if isinstance(prm['source']['zenith'],dict):
                prm['source']['zenith'] = np.asarray(prm['source']['zenith']['value'])*\
                                          units[prm['source']['zenith']['units']]
            if isinstance(prm['source']['azimuth'],dict):
                prm['source']['azimuth'] = np.asarray(prm['source']['azimuth']['value'])*\
                                          units[prm['source']['azimuth']['units']]
            prm['source'].update({'samplingTime':self.tau*self.sampling_rate})
            self.msg_args['Start'].update({'source_args':prm['source'],
                                           'sensor_class':prm['sensor']['class'],
                                           'sensor_args':{},
                                           'calibration_source_args':None,
                                           'calibrate_args':None})

            if 'source_attributes' in prm['source']:
                if 'rays' in prm['source']['source_attributes'] and \
                   'rot_angle' in  prm['source']['source_attributes']['rays'] and \
                   isinstance(prm['source']['source_attributes']['rays']['rot_angle'],dict):
                    prm['source']['source_attributes']['rays']['rot_angle'] = \
                      np.asarray(prm['source']['source_attributes']['rays']['rot_angle']['value'])*\
                       units[prm['source']['source_attributes']['rays']['rot_angle']['units']]
                    self.msg_args['Start'].update({'source_attributes':prm['source']['source_attributes']})

            if prm['sensor']['class'] is not None:
                self.msg_args['Start']['sensor_args'].update(prm['sensor']['args'])
                self.msg_args['Start']['calibrate_args'] = prm['sensor']['calibrate_args']
            if 'interaction matrices' in prm:
                self.msg_args['Init'].update(prm['interaction matrices'])
            self.msg_args['Outputs']['outputs'] += [k_o for k_o in self.outputs]
class Client(Driver):
    def __init__(self,tau,tag,logs,delay=0,sampling_rate=1,
                 verbose=logging.INFO,**kwargs):
        Driver.__init__(self,tau,tag)
        self.logger = logging.getLogger(tag)
        self.logger.setLevel(verbose)
        self.delay         = delay
        self.sampling_rate = sampling_rate
        self.inputs        = {}
        if 'inputs' in kwargs:
            for k,v in kwargs['inputs'].items():
                self.logger.info('New input: %s',k)
                self.inputs[k] = Input(k,**v)
        self.outputs       = {}
        if 'outputs' in kwargs:
            for k,v in kwargs['outputs'].items():
                self.logger.info('New output: %s',k)
                if 'logs' in v:
                    logs.add(tag,k,v['logs']['decimation'])
                    v['logs'] = logs.entries[tag][k]
                    self.logger.info('Output logged in!')
                self.outputs[k] = Output(k,**v)
        self.system = None
        self.__xout = np.zeros(0)
        self.__yout = np.zeros(0)

    def start(self):
        self.logger.debug('Starting!')
        pass
    def init(self):
        self.logger.debug('Initializing!')
        self.system = self.system._as_ss()
        self.__xout = np.zeros((1,self.system.A.shape[0]))
        self.__yout = np.zeros((1, self.system.C.shape[0]))
    def update(self,step):
        if step>=self.delay and step%self.sampling_rate==0:
            self.logger.debug('Updating!')
            u = np.hstack([_.data.reshape(1,-1) for _  in self.inputs.values()])
            self.logger.debug('u: %s',u)
            self.__yout = np.dot(self.system.C, self.__xout) + np.dot(self.system.D, u)
            self.__xout = np.dot(self.system.A, self.__xout) + np.dot(self.system.B, u)

    def output(self,step):
        if step>=self.delay:
            for k,v in self.outputs.items():
                if step%v.sampling_rate==0:
                    self.logger.debug('Outputing %s!',k)
                    a = 0
                    for k in self.outputs:
                        b = a + self.outputs[k].data.size
                        self.outputs[k].data[...] = \
                                    self.__yout[0,a:b].reshape(self.outputs[k].size)
                        a = b

    def terminate(self):
        self.logger.debug('Terminating!')

    def associate(self,prm_file):
        with open(prm_file) as f:
            prm = yaml.load(f)
        if 'transfer function' in prm['system']:
            system = prm['system']
            self.system = signal.dlti(system['transfer function']['num'],
                                      system['transfer function']['denom'])
        elif 'zeros poles gain' in prm['system']:
            system = prm['system']
            self.system = signal.dlti(system['transfer function']['zeros'],
                                      system['transfer function']['poles'],
                                      system['transfer function']['gain'])
        else:
            raise Exception("System should be of the type "+\
                            "'transfer function' or 'zeros poles gains'")
class Atmosphere(Driver):
    def __init__(self,tau,tag,server,verbose=logging.INFO,**kwargs):
        Driver.__init__(self,tau,tag)
        self.logger = logging.getLogger(tag)
        self.logger.setLevel(verbose)
        self.server = server
        self.inputs = {}
        self.outputs= {}
        self.msg    = {'class_id':'ATM',
                       'method_id':'Start',
                       'args':{}}
    def start(self):
        self.server._send_(self.msg)
        reply = self.server._recv_()
        self.logger.info('%s',reply)

    def terminate(self):
        self.server._send_({'class_id':'ATM',
                            'method_id':'Terminate',
                            'args':{'args':None}})
        reply = self.server._recv_()
        self.logger.info('%s',reply)

    def associate(self,prm_file):
        with open(prm_file) as f:
            prm = yaml.load(f)
        self.msg['args'].update(prm)
