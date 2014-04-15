### Find Density Networks         							      ############
### Douglas Fernando da Silva - doug.fernando@gmail.com          ############

import numpy
import sys

# NUM_NODES = 50
NUM_NODES = 500000
NUM_NODES_RANGE = range(NUM_NODES)

# sum(x) = |x| since only 0 and 1
def density(_S, _ES):
	s_es, s_s = _ES.sum(), _S.sum()

	if s_s == 0 and s_es == 0: return 0

	return  0.5 * s_es / s_s #  two points attached to the edge

# Find the induced edge set
def find_ES(_S):
	result = numpy.zeros(NUM_NODES)
	
	with open(sys.argv[1], "r") as f:
		for line in f:
			source, destiny = line.split(None, 1)
			source, destiny = int(source), int(destiny)

			if (_S[source] > 0 and _S[destiny] > 0): # both end points belongs to S?
				result[source] = result[source] + 1
				result[destiny] = result[destiny] + 1

	return result

# main algorithm for finding dense networks
def find_dense(S, epsilon):
	# ~S, S <= V
	St = numpy.copy(S)
	ES = find_ES(S)
	den_S = density(S, ES)
	den_St = den_S

	# return items for item ii)
	den_Si, ESi, Si, Ii = [], [], [], []

	iteration = 0
	sum_S = S.sum() # sum = |S| since only 0 and 1
	while sum_S > 0:
		den_limit = 2 * (1 + epsilon) * den_S
		# S <= S \ A(S)
		for i in NUM_NODES_RANGE: 
			if ES[i] <= den_limit: 
				S[i] = 0

		ES = find_ES(S)
		den_S = density(S, ES)

		if den_S > den_St:
			den_St = den_S
			St = numpy.copy(S)

		if (epsilon == 0.05): ###### add items to item iii)
			Si.append(S.sum()) 
			den_Si.append(den_S) 
			ESi.append(ES.sum()) 
			Ii.append(iteration)

		iteration = iteration + 1
		sum_S = S.sum()

	return St, iteration, Si, den_Si, ESi, Ii



