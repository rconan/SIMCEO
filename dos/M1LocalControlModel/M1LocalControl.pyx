cdef extern from "M1LocalControl.h":
	ctypedef double real_T
	ctypedef struct ExtU_M1LocalControl_T:
		real_T HP_LC[42]   #              /* '<Root>/HP_LC' */
	ExtU_M1LocalControl_T M1LocalControl_U
	ctypedef struct ExtY_M1LocalControl_T:
		real_T M1_ACT_F[2316]     #              /* '<Root>/M1_ACT_F' */
	ExtY_M1LocalControl_T M1LocalControl_Y
	void M1LocalControl_initialize()
	void M1LocalControl_step()
	void M1LocalControl_terminate()

cimport numpy as np
import numpy as np

class M1LocalControl: 
	'''
	This class was automatically generated from the simulink subsystem 'M1LocalControl'.

	For more information on how this code was generated access:
	https://github.com/feippolito/NSEElib/tree/master/MATLAB/%2Bcompile - pycreate.m

	Generated on 04-Jun-2020.
	'''

	def __init__(self):
		pass

	def init(self):
		M1LocalControl_initialize()
		self.__yout = np.zeros(2316)

	def update(self, np.ndarray u):
		cdef double[:] __u
		cdef int k
		__u = np.ravel(np.asarray(u))
		for k in range(0,42):
			j = k - 0
			M1LocalControl_U.HP_LC[j] = __u[k]
		M1LocalControl_step()

	def output(self):
		cdef int k
		for k in range(0,2316):
			j = k - 0
			self.__yout[k] = M1LocalControl_Y.M1_ACT_F[j]
		return np.atleast_2d(self.__yout)

	def terminate():
		M1LocalControl_terminate()