###### STOCHASTIC GRAD. DESC.   ############
# Latent Features for Recommendations #####
### Douglas Fernando da Silva   ############
### doug.fernando@gmail.com     ############
############################################

import numpy
import sys
import math
from datetime import datetime

# Finds the dimensions of the matrices P and Q by analyzing the input file
def find_dimensions(filepath):
	size_Q, size_P = 0, 0
	with open(filepath,"r") as f:
		for line in f:
			user, item, rating = line.split()
			user, item = int(user), int(item)
			if size_Q < item: size_Q = item
			if size_P < user: size_P = user

	return size_Q, size_P

# init P and Q with random values between 0 and math.sqrt(5.0 / k)
def init_P_Q(filepath, k):
	size_Q, size_P = find_dimensions(filepath)

	Q = numpy.array(numpy.random.rand(size_Q, k))
	P = numpy.array(numpy.random.rand(size_P, k))

	max_value = math.sqrt(5.0 / k)
	Q = max_value * Q
	P = max_value * P

	return Q, P

# update P and Q based on the update equations (1a)
def update_P_Q(P, Q, filepath, learning_rate, regulation):
	with open(filepath,"r") as f:
		for line in f:
			user, item, rating = line.split(None, 2)
			user, item, rating = int(user)-1, int(item)-1, int(rating)
			
			oldP = P[user, :]
			oldQ = Q[item, :]
			
			error = rating - numpy.dot(oldQ, oldP.T)

			newP = oldP + learning_rate * (error * oldQ - regulation * oldP)
			newQ = oldQ + learning_rate * (error * oldP - regulation * oldQ)

			P[user, :] = newP
			Q[item, :] = newQ

def main_sgd(filepath, k, learning_rate, regulation, iterations):
	Q, P = init_P_Q(filepath, k)

	iter_i = range(iterations)
	error_i = numpy.zeros(iterations)

	for i in iter_i:
		# update P & Q for the iteration
		update_P_Q(P, Q, filepath, learning_rate, regulation)

		# calculates the error for the iteration (item b)
		with open(filepath, "r") as f:
			for line in f:
				user, item, rating = line.split(None, 2)
				user, item, rating = int(user)-1, int(item)-1, int(rating)
				
				p = P[user, :]
				q = Q[item, :]

				error_i[i] = error_i[i] + (rating - numpy.dot(q, p.T))**2

		for user_i in range(P.shape[0]):
			p = P[user_i, :]
			error_i[i] = error_i[i] + regulation * numpy.dot(p, p.T) # ||p||2

		for item_i in range(Q.shape[0]):
			q = Q[item_i, :]
			error_i[i] = error_i[i] + regulation * numpy.dot(q, q.T) # ||q||^2

		# print "Sys time: %s | K: %s | Iteration: %s | Error: %s " % (str(datetime.now()), k, i, error_i[i]) #item b

	return iter_i, P, Q

# calculates the error without regulation using P & Q based on the reference file provided
# used to calculate Etr and Ete
def calc_error(P, Q, filename, k, reg, etr=False):
	error = 0
	with open(filename, "r") as f:
		for line in f:
			user, item, rating = line.split(None, 2)
			user, item, rating = int(user)-1, int(item)-1, int(rating)
			
			p = P[user, :]
			q = Q[item, :]

			error = error + (rating - numpy.dot(q, p.T))**2
	if etr:
		print "Etr: %s|%s|%s" % (k, reg, error)
	else:
		print "Ete: %s|%s|%s" % (k, reg, error)

if __name__ == '__main__':
	K = 20 
	regulation = 0.2

	iterations = 40
	learning_rate = 0.03 # learning rate produces error below 83k

	main_sgd(sys.argv[1], K, learning_rate, regulation, iterations)

	for k in range(1,11):
		for reg in [0.0, 0.2]:
			Iter_i, P, Q = main_sgd(sys.argv[1], k, learning_rate, reg, iterations)
			calc_error(P, Q, sys.argv[1], k, reg, etr=True) # Etr
			calc_error(P, Q, sys.argv[2], k, reg, etr=False) # Ete
	



