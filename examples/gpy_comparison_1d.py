# Copyright (c) 2021, Jeremy Roy
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Offroad Robotics Lab at Queen's University nor the
#    names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY <COPYRIGHT HOLDER> ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This file shows example usage of gpr-lib to perform Gaussian Process
# Regression on a noisy 1D sinusoidal wave.
#
# Author: Jeremy Roy <jeremy.roy@queensu.ca>
# License: BSD 2.0

import GPy
from .. import scripts
import numpy as np
import matplotlib.pyplot as plt
import time

def gpy_comparison_1d(
    latent_function_id=1, 
    num_restarts=10,
    observation_points=30,
    noise_var=0.05,
    test_points=200
):
    """
    Function showing example usage of the gpr_lib.GPRegressor object with a 1D
    latent function and it's comparitive interface and performace with the GPy
    library. An RBF kernel is used with the regressors from each library.

    This function shows example usage of gpr-lib to perform Gaussian Process
    Regression on a noisy 1D sinusoidalwave,  f(x) = sin(0.9x) for x=[-5,5].

    Kwargs:
        num_restarts (int):
            Number of hyperparameter optimization restarts.
        observation_points (int):
            Number of observation points from the latent function (Training points).
        noise_var (float):
            The sampling noise variance on the latent function.
        test_points (int):
            Number of test points from the fitted gaussian process.
    """

    ########################################
    # Specify latent function, test points #
    ########################################

    f = lambda x: np.sin(0.9*x).flatten()
    X = np.random.uniform(-5, 5, size=(observation_points,1))
    y = (f(X) + noise_var*np.random.randn(observation_points)).reshape(-1,1)

    Xtest = np.linspace(-5, 5, test_points).reshape(-1,1)

    #################
    # Using gpr_lib #
    #################
    print("\n\n*********Using gpr_lib:********\n")

    # Start Timer
    start_1 = time.time()

    # Create GPR Kernel
    kernel_1 = scripts.RBF(input_dim=1, l=0.1, ARD=True)

    # Create GPR model
    model_1 = scripts.GPRegressor(kernel_1, sig_var=1.)
    model_1.set_XY(X, y, sample_noise=noise_var)

    # Fit the GP (optimize kernel hyperparameters)
    model_1.fit()

    # Perform prediction
    mean_1, var_1 = model_1.predict(Xtest)
    std_1 = var_1 ** 0.5

    # Print hyperparameters
    print("LML: " + str(float(model_1.LML)))
    print('l: ' +  str(model_1.kernel.hprmtrs[0]) + "\tsig_var: " + str(model_1.sig_var) + "\tnoise_var: " + str(model_1.sample_noise))

    # Stop Timer
    stop_1 = time.time()
    elapsed_1 = stop_1 - start_1
    print("Time Elapsed: " + str(elapsed_1))

    # Plot results
    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_title("gpr_lib")
    ax1.plot(X, y, 'k+', ms=18)
    ax1.plot(Xtest, f(Xtest), 'b-')
    ax1.fill_between(Xtest.ravel(), (mean_1-(2*std_1)).ravel(), (mean_1+(2*std_1)).ravel(), color="#dddddd")
    ax1.set_ylabel("System Output")
    ax1.plot(Xtest, mean_1, 'r--', lw=2)

    #############
    # Using GPY #
    #############
    print("\n\n************Using GPy:************\n")

    # Start Timer
    start_2 = time.time()

    # Create GPR kernel
    kernel_2 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=0.1, ARD=True) 

    # Create GPR model
    model_2 = GPy.models.GPRegression(X, y, kernel_2)

    # Fit the GP (optimize kernel hyperparameters)
    model_2.optimize() #_restarts(num_restarts=1, messages=False)

    # Perform prediction
    mean_2, var_2 = model_2.predict(Xtest,include_likelihood=False)
    std_2 = var_2 ** 0.5

    # Print hyperparameters
    print("LML: " + str(float(model_2._log_marginal_likelihood)))
    print("l: " + str(model_2.rbf.lengthscale[0]) + "\tsig_var: " + str(model_2.rbf.variance[0]) + "\tnoise_var: " + str(model_2.Gaussian_noise.variance[0]))

    # Stop Timer
    stop_2 = time.time()
    elapsed_2 = stop_2 - start_2
    print("Time Elapsed: " + str(elapsed_2))

    # Plot results
    ax2.set_title("GPy")
    ax2.plot(X, y, 'k+', ms=18)
    ax2.plot(Xtest, f(Xtest), 'b-')
    ax2.fill_between(Xtest.ravel(), (mean_2-(2*std_2)).ravel(), (mean_2+(2*std_2)).ravel(), color="#dddddd")
    ax2.set_ylabel("System Output")
    ax2.set_xlabel("System Input")
    ax2.plot(Xtest, mean_2, 'r--', lw=2)

    plt.show()