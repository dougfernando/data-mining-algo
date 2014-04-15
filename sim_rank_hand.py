###### SIMRANK IMPLEMENTATION   ############
### Douglas Fernando da Silva   ############
### doug.fernando@gmail.com     ############
############################################
import numpy


# G = numpy.matrix('0 1 1 0 1; 0 1 0 1 0; 1 0 0 0 0')
# G = numpy.squeeze(numpy.asarray(G))

# K2,1
# G = numpy.matrix('1; 1')

# K2,2
G = numpy.matrix('1 1; 1 1')
G = numpy.squeeze(numpy.asarray(G))


print G.shape

# initialization: if x == x: 1: else: 0
#######################################
Sa_init = numpy.identity(G.shape[0])  
Sb_init = numpy.identity(G.shape[1])

for iteration in [1, 2, 3]:
	##############
	# induction
	##############
	
	Sa = numpy.identity(G.shape[0])/2.0 # => symmetric (p1), to compensate tranpose (p2)
	for i in [0,G.shape[0]-1]:
		for j in range(i+1, G.shape[0]): # nchoosek(3,2)
			# Ox: destination nodes in the G graph for i and j
			_Oi = numpy.where(G[i, :] != 0)[0]
			_Oj = numpy.where(G[j, :] != 0)[0]

			for Oi in _Oi:
				for Oj in _Oj:
					# print "(i,j,sb)=(%s,%s," %(Oi, Oj, Sb_i)
					Sa[i, j] = Sa[i, j] + Sb_init[Oi, Oj];

			Sa[i, j] = 0.8 / (_Oi.shape[0] * _Oj.shape[0]) * Sa[i,j]
	Sa = Sa + Sa.T # add the 2nd symmetric (p2)

	# same approach, but now for Sb
	Sb = numpy.identity(G.shape[1])/2.0
	for i in range(G.shape[1]-1):
		for j in range(i+1, G.shape[1]): # nchoosek(5,2)
			_Ii = numpy.where(G[:, i] != 0)[0]
			_Ij = numpy.where(G[:, j] != 0)[0]
			
			for Ii in _Ii:
				for Ij in _Ij:
					Sb[i, j] = Sb[i, j] + Sa_init[Ii, Ij]

			Sb[i, j] = 0.8 / (_Ii.shape[0] * _Ij.shape[0])* Sb[i, j]
	Sb = Sb + Sb.T

	Sa_init = Sa
	Sb_init = Sb

	numpy.set_printoptions(precision=3)
	print "Iteration: %s" % (iteration)
	print "Sa:" 
	print Sa
	print "Sb:" 
	print Sb

