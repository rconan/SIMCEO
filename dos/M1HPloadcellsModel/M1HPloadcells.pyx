cdef extern from "M1HPloadcells.h":
	ctypedef double real_T
	ctypedef struct ExtU_M1HPloadcells_T:
		real_T M1_HP_D[84]   #              /* '<Root>/M1_HP_D' */
		real_T M1_HP_cmd[42]   #              /* '<Root>/M1_HP_cmd' */
	ExtU_M1HPloadcells_T M1HPloadcells_U
	ctypedef struct ExtY_M1HPloadcells_T:
		real_T M1_HP_LC[42]     #              /* '<Root>/M1_HP_LC' */
	ExtY_M1HPloadcells_T M1HPloadcells_Y
	void M1HPloadcells_initialize()
	void M1HPloadcells_step()
	void M1HPloadcells_terminate()

cimport numpy as np
import numpy as np

class M1HPloadcells: 
	'''
	This class was automatically generated from the simulink subsystem 'M1HPloadcells'.

	For more information on how this code was generated access:
	https://github.com/feippolito/NSEElib/tree/master/MATLAB/%2Bcompile - pycreate.m

	Generated on 04-Jun-2020.
	'''

	def __init__(self):
		pass

	def init(self):
		M1HPloadcells_initialize()
		self.__yout = np.zeros(42)

	def update(self, np.ndarray u):
		cdef double[:] __u
		cdef int k
		__u = np.ravel(np.asarray(u))
		for k in range(0,84):
			j = k - 0
			M1HPloadcells_U.M1_HP_D[j] = __u[k]
		for k in range(84,126):
			j = k - 84
			M1HPloadcells_U.M1_HP_cmd[j] = __u[k]
		M1HPloadcells_step()

	def output(self):
		cdef int k
		for k in range(0,42):
			j = k - 0
			self.__yout[k] = M1HPloadcells_Y.M1_HP_LC[j]
		return np.atleast_2d(self.__yout)

	def terminate():
		M1HPloadcells_terminate()