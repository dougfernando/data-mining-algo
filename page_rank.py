###### PAGE RANK IMPLEMENTATION ############
### Power-iteration and Monte Carlo ########
### Douglas Fernando da Silva   ############

import numpy
import sys
import time
import random

NUM_PAGES = 100 # num of nodes
NUM_ITERATIONS = 40
TEL_PROB = 0.2

#### "TRUE" PAGERANK - Power Iteration ##########
#################################################

def buildM(filepath): # Build matrix M based on the graph in the input file
	M = numpy.zeros(shape=(NUM_PAGES, NUM_PAGES))

	f = open(filepath,"r")
	for line in f:
		source, destiny = line.split()
		source, destiny = int(source), int(destiny)
		M[destiny - 1, source - 1] = 1
	
	for i in range(NUM_PAGES):
		num_conn = sum(M[:, i])
		M[:,i] = M[:,i] / num_conn

	return M

def power_iter(M): # finds the vector Rj based on the power iteration method
	v1T = numpy.ones((NUM_PAGES, 1)) # 1' 
	R = (1.0 / NUM_PAGES) * v1T

	for i in range(NUM_ITERATIONS):
		R1 = (TEL_PROB /  NUM_PAGES) * v1T + (1 - TEL_PROB) * numpy.dot(M, R) # PageRank equation
		R = R1
	
	return R

# PageRank algorithm using the Power Iteration method
def main_true_pagerank():
	print "\n### True PageRank - Power Iteration (40 iteration)..."

	t0 = time.clock() # init time

	# init matrix M from input file
	M = buildM(sys.argv[1])

	# run the power iteration algorithm
	R = power_iter(M)
	
	t = time.clock() - t0 # total time
	print("Execution time: %s " % (t))

	return R

#### PAGERANK - Monte Carlo implementation ##########
#####################################################

# find the next "random" edge from a given node to the next walk step
def find_next_node(vector, count): 
	return random.choice(list(numpy.nonzero(vector)[0]))

# Main PageRank Monte Carlo implementation
def MC(M, numR):
	Rj = numpy.zeros(NUM_PAGES)

	count = 1
	for c in range(M.shape[1]): # random walks from each node (columns of M)
		for r in range(numR): # R walks per node
			column = M[:, c] # M column = node
			while not random.random() < TEL_PROB:
				next_column = find_next_node(column, count)

				# before moving to the next node, add 1 to the destiny
				Rj[next_column] = Rj[next_column] + 1 
				
				column = M[:, next_column] #next node

				count = count + 1

	Rj = Rj * TEL_PROB / (NUM_PAGES * numR)

	return Rj

# Main MC implementation of PageRank
def main_MC():
	print "\n### PageRank MC implementation, R={1,3,5}... "

	# init matrix M from input file
	M = buildM(sys.argv[1])

	for _r in [1, 3, 5]: # num of random walks from a given node
		t0 = time.clock() # init time
		R = MC(M, _r) # MC algorithm execution
		t = time.clock() - t0 # total time for a given R
		print("R: %s | Execution time: %s " % (_r, t))

# Main MC implementation of PageRank, but calculting the error from Power Iteration results
def main_MC_Error(Rt): # Rt = Page rank result from Power Iteration method
	print "\n### PageRank MC impl with average error (Error/K) calculation..."

	# init matrix M from input file
	M = buildM(sys.argv[1])

	for _r in [1, 3, 5]: # num of random walks from a given node
		R = MC(M, _r) # MC algorithm execution
		top_indexes = numpy.argsort(R, axis=0)[::-1] # sort Rj descending order, return the indexes
		for k in [10, 30, 50, 100]: # top K results
			error = 0 # total error the the top K results
			for i in range(k):
				error = error + abs(Rt[top_indexes[i]] - R[top_indexes[i]])
			error = error / k
			print("R: %s | K: %03d | Error/K: %s | Error: %s" % (_r, k, error, k*error))
		print ""

if __name__ == '__main__':
	# Power iteration
	Rt = main_true_pagerank()
	# Monte Carlo
	main_MC()
	# Error between Montecarlo and PowerIteration
	main_MC_Error(Rt)
