#!/usr/bin/env python
# -*- coding: utf-8 -*-
from approxposterior import approx, gpUtils, likelihood as lh, utility as ut
import numpy as np
import corner

# Define algorithm parameters
m0 = 50                           # Initial size of training set
m = 20                            # Num points to find each iteration
nmax = 2                          # Maximum number of iterations
bounds = [(-5,5), (-5,5)]         # Prior bounds
algorithm = "bape"                # Use the Kandasamy et al. (2017) formalism
np.random.seed(57)
samplerKwargs = {"nwalkers" : 20}        # emcee.EnsembleSampler
mcmcKwargs = {"iterations" : int(2.0e4)} # emcee.EnsembleSampler.run_mcmc

# Evaluate model lnlikelihood + lnprior for each theta sampled from prior
theta = lh.rosenbrockSample(m0)
y = np.zeros(len(theta))
for ii in range(len(theta)):
    y[ii] = lh.rosenbrockLnlike(theta[ii]) + lh.rosenbrockLnprior(theta[ii])

# Use default GP with squared exponential kernel
gp = gpUtils.defaultGP(theta, y, white_noise=-12)

# Initialize object using the Wang & Li (2018) Rosenbrock function example
ap = approx.ApproxPosterior(theta=theta, y=y, gp=gp,
                            lnprior=lh.rosenbrockLnprior,
                            lnlike=lh.rosenbrockLnlike,
                            priorSample=lh.rosenbrockSample,
                            bounds=bounds, algorithm=algorithm)

# Run!
ap.run(m=m, nmax=nmax, estBurnin=True, nGPRestarts=3,
       mcmcKwargs=mcmcKwargs, ache=False, samplerKwargs=samplerKwargs,
       verbose=True, thinChains=False, onlyLastMCMC=True)

# Load in MCMC chain, plot the corner plot and where AP selected points
samples = ap.sampler.get_chain(discard=ap.iburns[-1], flat=True,
                               thin=ap.ithins[-1])
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                    scale_hist=True, plot_contours=True)
fig.axes[2].scatter(ap.theta[m0:,0], ap.theta[m0:,1], s=10,
                    color="red", zorder=20)
fig.savefig("finalPosterior.png", bbox_inches="tight")
