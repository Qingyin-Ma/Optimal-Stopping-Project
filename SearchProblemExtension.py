from scipy.interpolate import LinearNDInterpolator
from numpy import maximum as npmax
from numpy import minimum as npmin
from numpy.random import randn
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm


class SearchProblemExtension(object):
	"""
	A class to store a given parameterization of the generalized
	job search model.

	The value function:
	V(w,mu,gam) = max{ u(w), c0+beta*E[V(w',mu',gam')|mu,gam] }
	
	The Bayesian updating process:
	w = theta + eps_w, eps_w ~ N(0, gam_w)
	prior: theta ~ N(mu, gam)
	posterior: theta|w' ~ N(mu', gam')
			   gam' = 1/(1/gam + 1/gam_w)
			   mu' = gam'*(mu/gam + w'/gam_w)

	Agents have constant absolute risk aversion:
	u(w) = (1/a) * (1 - exp(-a*w))

	Parameters
	----------
	beta : scalar(float), optional(default=0.95)
		   The discount factor
	c0 : scalar(float), optional(default=0.05)
		 The unemployment compensation
	a : scalar(float), optional(default=2.5)
		The coefficient of absolute risk aversion
	gam_w : scalar(float), optional(default=0.005)
			The variance of eps_w
	N_mc : scalar(int), optional(default=10000)
		   The number of Monte Carlo samples
	musize : scalar(int), optional(default=100)
			 The number of grid points over mu
			 An even integer
	gamsize : scalar(int), optional(default=50)
			  The number of grid points over gam

	"""


	def __init__(self, beta=0.95, c0=0.05, a=2.5, gam_w=0.005,
				 N_mc=1000, musize=100, gamsize=50):
		
		self.beta, self.c0, self.a = beta, c0, a
		self.gam_w, self.N_mc = gam_w, N_mc
		self.musize, self.gamsize = musize, gamsize
		
		# === Make grid points over mu === #
		self.mumin, self.mumax = 1e-3, 50.0
		self.muscale = 4.0
		# Make grids on the right side of the real line
		mugrid_pos = self.makegrid(self.mumin, self.mumax, 
								   self.musize/2, self.muscale)
		# Make grids on the left side of the real line
		mugrid_neg = - mugrid_pos
		# Concatenate and sort to obtain the desired grids
		mugrid = np.concatenate((mugrid_neg, mugrid_pos))
		self.mugrid = np.sort(mugrid)
		
		# === Make grid points over gam === #
		self.gammin, self.gammax = 1e-4, 25.0
		self.gamscale = 4.0
		self.gamgrid = self.makegrid(self.gammin, self.gammax, 
									 self.gamsize, self.gamscale)
		
		self.x, self.y = np.meshgrid(self.mugrid, self.gamgrid)
		self.grid_points = np.column_stack((self.x.ravel(1), 
											self.y.ravel(1)))

		self.draws = randn(self.N_mc)  # initial Monte Carlo draws



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



	def kappa_func(self, mu, gam):
		"""
		The kappa function / weight function used for constructing
		a new complete metric space (b_kappa \Theta, rho_kappa)

		"""
		gam_w, a = self.gam_w, self.a
		term1 = np.exp(-a * mu)
		term2 = np.exp((a ** 2) * (gam + gam_w) / 2.0)

		return term1 * term2 + 1.0



	def u(self, w):
		"""
		The reward function
		"""
		a = self.a
		return (1.0 - np.exp(-a * w)) / a 



	def res_rule_operator(self, phi):
		"""
		The reservation rule operator
		-----------------------------
		Qphi = c0*(1-beta) + 
			   beta*integral( max{u(w'),phi(mu',gam')} * f(w'|mu,gam) )dw'
		where:
			   u(w) = (1/a) * (1 - exp(-a*w))
			   f(w'|mu, gam) = N(mu, gam + gam_w)
			   gam' = 1/(1/gam + 1/gam_w)
			   mu' = gam' * (mu/gam + w'/gam_w)

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
		beta, gam_w, N_mc = self.beta, self.gam_w, self.N_mc
		N_mc, a, c0 = self.N_mc, self.a, self.c0
		draws = self.draws
		u =  self.u
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

		for i in range(N):
			mu, gam = self.grid_points[i, :]
			
			# sample w' from f(w'|mu,gam) = N(mu, gam + gam_w)
			draws_w = mu + np.sqrt(gam + gam_w) * draws
			
			# the updated belief: mu', the update is based on
			# the next period observation w'.
			gam_prime = 1.0 / (1.0 / gam + 1.0 / gam_w) # a scalar
			b1 = gam_prime / gam
			mu_prime = b1 * mu + (1.0 - b1) * draws_w # an array with length N_mc

			# the updated belief: gam'
			gam_prime = gam_prime * np.ones(N_mc) # an array with length N_mc
			
			expected_term = npmax(u(draws_w), phi_f(mu_prime, gam_prime))
			expectation = np.mean(expected_term)
			new_phi[i] = c0 * (1.0 - beta) + beta * expectation

		return new_phi



	def compute_fixed_point(self, T, v, error_tol=1e-6, 
							max_iter=500, verbose=1):
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
			error = max(abs((new_v - v) / kappa_func(grid_points[:,0], grid_points[:,1])))
			if verbose:
				print "compute iterate ", iterate, " with error ", error
			v = new_v
			if iterate == max_iter:
				print "maximum iteration reached ... "

		return v



sp = SearchProblemExtension()
phi_init = np.ones(len(sp.grid_points)) # initial guess of the fixed point

# compute the fixed point: reservation utility
res_util = sp.compute_fixed_point(T=sp.res_rule_operator, v=phi_init)
res_util = np.reshape(res_util, (sp.musize, sp.gamsize))


# === plot reservation utility === #
"""
# Plot the figure on the whole grid range

mu_meshgrid, gam_meshgrid = sp.x, sp.y

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(mu_meshgrid, gam_meshgrid, res_util.T,
				rstride=2, cstride=3, cmap=cm.jet,
				alpha=0.5, linewidth=0.25)

ax.set_xlabel('mu', fontsize=14)
ax.set_ylabel('gamma', fontsize=14)
ax.set_zlabel('Reservation utility', fontsize=14)

"""

# Plot the figure on the important part of the grid range
mu_meshgrid, gam_meshgrid = sp.x, sp.y

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(mu_meshgrid[:,15:85], gam_meshgrid[:,15:85], 
				res_util.T[:,15:85],
				rstride=2, cstride=3, cmap=cm.jet,
				alpha=0.5, linewidth=0.25)

ax.set_xlabel('mu', fontsize=14)
ax.set_ylabel('gamma', fontsize=14)
ax.set_zlabel('Reservation utility', fontsize=14)

ax.set_xlim((-12, 10))
ax.set_ylim((0, 25))
ax.set_zlim((0.1, 0.4))


plt.show()

