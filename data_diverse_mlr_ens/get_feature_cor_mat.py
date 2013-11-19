#! /usr/bin/python

# this script trains mlr with different weights to each sample

import sys
import math
from os import system
import numpy
import pickle


feature_mat_file = sys.argv[1] # the matrix containing the 0 and 1 indicating how features are put together
A = numpy.loadtxt(feature_mat_file,delimiter=',')

num_feats = A.shape[1]
num_clasf = A.shape[0]
B = numpy.zeros((num_feats,num_feats))

for iter_clasf in range(0,num_clasf):
	for iter_r in range(0,num_feats):
		for iter_c in range(0,num_feats):
			print A[iter_clasf,iter_r],A[iter_clasf,iter_c]
			if ((A[iter_clasf,iter_r] == 1) and (A[iter_clasf,iter_c] == 1)):
				print "hi"
				B[iter_r,iter_c] += 1
				print B[iter_r,iter_c]

B = (1.0/num_clasf)*B
print B	
