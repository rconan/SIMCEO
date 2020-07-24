import numpy as np
import logging

logging.basicConfig()

class Constant:

    def __init__(self,**kwargs):
        self.logger = logging.getLogger(name='Constant')
        self.logger.setLevel(logging.INFO)
        self.val = kwargs['parameters']['value']
        self.y = self.val.copy()

    def init(self):
        pass

    def update(self,u=0.0):
        self.y = self.val + u

    def output(self):
        return np.atleast_2d(self.y)

    def terminate(self):
        pass

