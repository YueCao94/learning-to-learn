import numpy as np
from numpy.linalg import inv
from Confiuration import config

class posterior(object):

	def __init__(self, H = 10,  l = 1, epsilon = 0.5, rho_zero = 1):

		self.l = l
		self.H = H
		self.epsilon = epsilon
		self.rho_zero = rho_zero
		self.dim = len(sample['x'][0])


	def kernel(self, x1,x2):

		return np.exp(-np.linalg.norm(np.subtract(x1,x2))**2 / (2*self.l**2) )


	def fit(self, sample):

		self.n = len(sample['x'])
		n = self.n
		self.sample = sample


		K = np.zeros((n,n))

		for i in range(n):
			for j in range(n):
				K[i][j] = self.kernel(self.sample['x'][i], self.sample['y'][i])

				if(i==j):
					K[i][j] += self.epsilon


		self.kriging_coef = np.matmul(inv(K), sample['y'])


	def kriging(self, x):

		n = self.n
		score= 0.

		for i in range(n):
			score += self.kernel(x, self.sample['x'][i]) * self.kriging_coef[i]

		return score


	def rho(self):

		return self.rho_zero*exp(1.0/self.H*pow(double(self.n), 1.0/(self.dim/2.0+2.0)));




	def stepsize_adaption(self, stepsize, acratio, t2):


		if ( (t2+1) % (config.burning_MCMC/100) == 0):
					
			if ( (acratio) / double(self.burning_MCMC/100) > 0.55):
					
				stepsize = stepsize * 1.1;
					
			else:
				
				if ((acratio) / double(self.burning_MCMC/100) < 0.45):
					
					stepsize = stepsize * 0.9;
					
			
			acratio = 0.0

		return stepsize, acratio





	def Metropolis_criteria(self, x, stepsize):


		 #-------------------------------------generate proposal
		cand = np.random.normal(x[i], stepsize)
	


		#--------------------------------------------------- calculate acceptance probability
			
		candidate = self.rho * self.kriging(cand)
	         

		double u = np.random.uniform(0 , 1)


		if(judgebound(cand) && u<np.exp(candidate - x)):
			return cand, 1


		return x,0



	def MCMC(self):

		
		x_new = initial_coefficient()
		
		ac_total = 0.0
		acratio = 0.0
		stepsize = config.Metropolis_Steplenth * 4.0 

		entropy_sample= []
		mcmc_sample= []


		rho = self.rho()

		current = rho*kriging(x_new)

#------------------------------------------------------------------burning MCMC-------------------------
		for t2 in range(config.burning_MCMC):

			x_new, augment = Metropolis_criteria(x_new, stepsize)

			ac_total+=augment
			acratio+=augment

			
			#------------------------------------------tuning the stepsize of MCMC
			stepsize, acratio = stepsize_adaption(stepsize, acratio, t2)
	

#--------------------------------------------------------------------------pick MCMC

		acratio = 0
		for t2 in range(config.n_sample):

			for t3 in range(config.pick_MCMC):
				
				x_new, augment = Metropolis_criteria(x_new, sample, &current, stepsize)


				acratio+=augment;


				entropy_sample.append(x_new)


				stepsize, acratio = stepsize_adaption(stepsize, acratio, t3+t2*config.pick_MCMC)




			#--------------------------------------pick a new sample point
			mcmc_sample.append(x_new)



		print "ac_ratio:", ac_total/config.burning_MCMC
		return mcmc_sample, entropy_sample

		
