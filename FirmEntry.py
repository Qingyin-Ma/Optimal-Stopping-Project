from scipy.interpolate import LinearNDInterpolator
from numpy import maximum as npmax
from numpy import minimum as npmin
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from scipy.stats import lognorm
import time



class FirmEntry(object):
	"""
	A class to store a given parameterization of the firm entry
	model.


	Original paper:
	Fajgelbaum, Schaal, and Taschereau-Dumouchel (2015)


	x = theta + eps_x, eps_x ~ N(0, gam_x)
	where: 
		   x: output, not observed when the firm makes decisions 
		   	  at the beginning of each period.


	y = theta + eps_y, eps_y ~ N(0, gam_y)
	where: 
		   y: a public signal, observed after the firm makes decision
		   	  of whether to invest or not


	The Bayesian updating process:
	theta ~ N(mu, gam) : prior
	theta|y ~ N(mu', gam') : posterior after observing public 
							 signal y
	where: 
		   gam' = 1 / (1/gam + 1/gam_y)
		   mu' = gam'*(mu/gam + y/gam_y)


	The firm has constant absolute risk aversion:
	u(x) = (1/a) * (1 - exp(-a*x))
	where: a is the coefficient of absolute risk aversion


	f ~ h(f) = LN(mu_f, gam_f) : the entry cost / the investment cost


	The value function:
	V(f,mu,gam) = max{ E[u(x)|mu,gam]-f, beta*E[V(f',mu',gam')|mu,gam] }
	where: 
		   RHS 1st: the value of entering the market and investing
		   RHS 2nd: the expectated value of waiting


	Parameters
	----------
	beta : scalar(float), optional(default=0.95)
	       the discount factor
	a : scalar(float), optional(default=0.6)
	    the coefficient of absolute risk aversion
	gam_x : scalar(float), optional(default=1.0)
	   		the variance of eps_x, eps_x ~ N(0, gam_x)
	gam_y : scalar(float), optional(default=1.5)
			the variance of eps_y, eps_y ~ N(0, gam_y)
	mu_f : scalar(float), optional(default=0.05)
		   mean of the cost distribution f~h(f)=LN(mu_f, gam_f)
	gam_f : scalar(float), optional(default=0.0005)
			variance of the cost distribution f~h(f)=LN(mu_f, gam_f)  
	musize : scalar(int), optional(default=100)
			 the number of grid points over mu
			 an even integer 
	gamsize : scalar(int), optional(default=50)
			  the number of grid points over gam 
	N_mc : scalar(int), optional(default=10000)
		   the number of Monte Carlo samples 

	"""


	def __init__(self, beta=0.95, a=0.6, gam_x=1.0, gam_y=1.5,
				mu_f=0.05, gam_f=0.0005, musize=100, gamsize=50,
				N_mc=10000):

		self.beta, self.a, self.N_mc = beta, a, N_mc
		self.gam_x, self.gam_y = gam_x, gam_y
		self.mu_f, self.gam_f = mu_f, gam_f
		self.musize, self.gamsize = musize, gamsize
		
		# === Make grid points over mu === #
		self.mumin, self.mumax = 1e-3, 50.0
		self.muscale = 1.0
		# Make grids on the right side of the real line
		mugrid_pos = self.makegrid(self.mumin, self.mumax,
								   self.musize/2, self.muscale)
		# Make grids on the left side of the real line		
		mugrid_neg = - mugrid_pos
		mugrid = np.concatenate((mugrid_neg, mugrid_pos))
		self.mugrid = np.sort(mugrid)

		# === Make grid points over gam === #
		self.gammin, self.gammax = 1e-4, 25.0
		self.gamscale = 1.0
		self.gamgrid = self.makegrid(self.gammin, self.gammax,
									 self.gamsize, self.gamscale)

		self.x, self.y = np.meshgrid(self.mugrid, self.gamgrid)
		self.grid_points = np.column_stack((self.x.ravel(1), 
											self.y.ravel(1)))



	def makegrid(self, amin, amax, asize, ascale):
		"""
		Generates grid a with asize number of points ranging
		from amin to amax. 

		Parameters
		----------
		amin: the minimum grid 
		amax: the maximum grid 
		asize: the number of grid points to be generated

		ascale=1: generates equidistant grids, same as np.linspace
		ascale>1: the grid is scaled to be more dense in the 
				  lower value range

		Returns
		-------
		a : array_like(float, ndim=1, length=asize)
			The generated grid points
		
		"""
		a = np.empty(asize)
		adelta = (amax - amin) / ((asize - 1) ** ascale)
		
		for i in range(asize):
			a[i] = amin + adelta * (i ** ascale)
		
		return a



	def res_rule_operator(self, phi):
		"""
		The reservation rule operator
		-----------------------------
		Qphi(mu,gam) = 
		beta*integral( max{reward,phi(mu',gam')} * h(f')*l(y|mu,gam) )df'dy
		where:
			   f ~ h(f) = LN(mu_f, gam_f) : the entry/investment cost
			   gam' = 1/(1/gam + 1/gam_y)
			   mu' = gam' * (mu/gam + y/gam_y)
			   reward = (1/a) - (1/a)*exp[-a*mu' + (a**2)*(gam'+gam_x)/2] - f'
   			   l(y|mu, gam) = N(mu, gam + gam_y)


		The operator Q is a well-defined contraction mapping from 
		the complete metric space (b_kappa \Theta, rho_kappa) into 
		itself, where (b_kappa \Theta, rho_kappa) is the reweighted 
		space constructed by the weight function kappa.


		Parameters
		----------
		phi : array_like(float, ndim=1, length=len(grid_points))
			  An approximate fixed point represented as a one-dimensional
			  array.


		Returns
		-------
		new_phi : array_like(float, ndim=1, length=len(grid_points))
				  The updated fixed point.

		"""
		beta, a, N_mc = self.beta, self.a, self.N_mc
		gam_x, gam_y = self.gam_x, self.gam_y
		mu_f, gam_f = self.mu_f, self.gam_f

		interp_phi = LinearNDInterpolator(self.grid_points, phi)

		def phi_f(mu, gam):
			"""
			Interpolate but extrapolate using the nearest value
			on the grid.
			"""
			mu = npmax(self.mumin, mu)
			gam = npmax(self.gammin, gam)
			mu = npmin(self.mumax, mu)
			gam = npmin(self.gammax, gam)
			return interp_phi(mu, gam)

		N = len(phi)
		new_phi = np.empty(N)
		draws = randn(N_mc, 2) # initial Monte Carlo samples

		for i in range(N):
			mu, gam = self.grid_points[i, :]

			# sample f' from h(f') = LN(mu_f, gam_f)
			draws_f = np.exp(mu_f + np.sqrt(gam_f) * draws[:,0])

			# sample y from l(y|mu,fam) = N(mu, gam + gam_f)
			draws_y = mu + np.sqrt(gam + gam_y) * draws[:,1]

			# the updated belief: mu', the update is based on
			# the public signal y.
			gam_prime = 1.0 / (1.0 / gam + 1.0 / gam_y)
			mu_prime = gam_prime * (mu / gam + draws_y / gam_y)

			# the updated belief: gam'
			gam_prime = gam_prime * np.ones(N_mc)

			term_1 = np.exp(-mu_prime*a + (a**2)*(gam_x + gam_prime)/2.0)
			term_1 = (1.0 - term_1) / a - draws_f
			term_2 = phi_f(mu_prime, gam_prime)
			expectation = np.mean(npmax(term_1, term_2))
			new_phi[i] = beta * expectation

		return new_phi



	def kappa_func(self, mu, gam):
		"""
		The kappa function / weight function used for constructing
		a new complete metric space (b_kappa \Theta, rho_kappa).

		"""
		a, gam_x = self.a, self.gam_x 
		
		return np.exp(-a * mu + (a**2) * (gam_x + gam) / 2.0) + 1.0



	def compute_fixed_point(self, T, v, error_tol=1e-4,
							max_iter=100, verbose=1):
		"""
		Compute the fixed point of the reservation rule operator.
		"""
		grid_points = self.grid_points
		kappa_func = self.kappa_func

		iterate = 0
		error = error_tol + 1
		while iterate < max_iter and error > error_tol:
			new_v = T(v)
			iterate += 1
			error = np.max(abs(new_v - v) / kappa_func(grid_points[:,0], 
													   grid_points[:,1]))
			if verbose:
				print "Compute iterate ", iterate, " with error ", error
			v = new_v
			if iterate == max_iter:
				print "Maximum iteration reached ..."

		return v



	def recover_res_rule(self, fixedpoint):
		"""
		Recover the reservation cost
		"""
		a, gam_x, grid_points = self.a, self.gam_x, self.grid_points
		term_1 = (a**2) * (gam_x + grid_points[:,1]) / 2.0
		term_2 = np.exp(-grid_points[:,0] * a + term_1) 
		term_3 = (1 - term_2) / a

		return term_3 - fixedpoint



start = time.time()
		
fe = FirmEntry()
phi_init = np.ones(len(fe.grid_points)) # initial guess of the fixed point

# compute the fixed point
fixedpoint = fe.compute_fixed_point(T=fe.res_rule_operator, v=phi_init)

# recover the reservation cost from the fixed point 
res_cost = fe.recover_res_rule(fixedpoint)

# calculate the perceived probability of investment p
# p(mu,gam) = F(res_cost(mu,gam)), F: cdf of LN(mu_f, gam_f)
prob_invest = lognorm.cdf(res_cost, s=np.sqrt(fe.gam_f), 
					      scale=np.exp(fe.mu_f))

# reshape the reservation cost and the perceived prob. of investment
res_cost = np.reshape(res_cost, (fe.musize, fe.gamsize))
prob_invest = np.reshape(prob_invest, (fe.musize, fe.gamsize))


# === plot perceived probability of investment === #

# Plot the figure on the important part of the grid range 
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
mu_meshgrid, gam_meshgrid = fe.x, fe.y
ax.plot_surface(mu_meshgrid[:,40:75], gam_meshgrid[:,40:75], 
				prob_invest.T[:,40:75],
				rstride=2, cstride=3, cmap=cm.jet,
				alpha=0.5, linewidth=0.25)
ax.set_xlabel('mu', fontsize=14)
ax.set_ylabel('gamma', fontsize=14)
ax.set_zlabel('Perceived prob. of investment', fontsize=14)

ax.set_xlim(-10, 25)
ax.set_ylim(0, 25)
ax.set_zlim(0, 1)

plt.show()


"""
# Plot the figure on the whole grid range

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
mu_meshgrid, gam_meshgrid = fe.x, fe.y
ax.plot_surface(mu_meshgrid, gam_meshgrid, prob_invest.T,
				rstride=2, cstride=3, cmap=cm.jet,
				alpha=0.5, linewidth=0.25)
ax.set_xlabel('mu', fontsize=14)
ax.set_ylabel('gamma', fontsize=14)
ax.set_zlabel('Perceived prob. of investment', fontsize=14)

plt.show()
"""


end = time.time()
elapsed = end - start

print ""
print "Time taken:", elapsed / 60.0, "minutes..."