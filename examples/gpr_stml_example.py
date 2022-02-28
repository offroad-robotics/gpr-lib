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
# Regression with SpatialTemporal Memory Loss on a noisy 1D sinusoidal 
# wave that undergoes a piecewise evolution.  The shape of the evolution
# can be modified with the latent_function_id variable.
#
# Author: Jeremy Roy <jeremy.roy@queensu.ca>
# License: BSD 2.0


from __future__ import print_function

from .. import scripts as gpr_lib

import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

import sys
import copy
import time


def gpr_stml_example(
    kernel=None, 
    latent_function_id=1, 
    num_restarts=10,
    observation_points=30,
    noise_var=0.0025,
    test_points=200
):
    """
    Function showing example usage of the gpr_lib.GPRegressorSTML object for a
    1D input space.

    This function shows example usage of gpr-lib to perform Gaussian Process
    Regression with SpatialTemporal Memory Loss on a noisy 1D sinusoidal 
    wave that undergoes a piecewise evolution.  The shape of the evolution
    can be modified with the latent_function_id variable. f(x) = sin(0.9x)
    for x=[-5,5]

    Kwargs:
        kernel (gpr_lib.Kernel):
            The kernel. If None, a default RBF kernel is used.
        latent_function_id (int): 
            Which type of updated latent function to use.  1, 2, or 3.
            1. -f(x)
            2. f(x) * u(x)
            3. f(u(x))
            Where u(x) is the unit step function.
        num_restarts (int):
            Number of hyperparameter optimization restarts.
        observation_points (int):
            Number of observation points from the latent function (Training points).
        noise_var (float):
            The sampling noise variance on the latent function.
        test_points (int):
            Number of test points from the fitted gaussian process.

    """
    # This kernel will be used by all GPR methods
    if kernel is None:
        kernel = gpr_lib.RBF(input_dim=1, l=0.1, l_range=(10e-10, 100), ARD=True)

    ###########################
    # Specify latent function #
    ###########################

    # Original latent function
    f = lambda x: np.sin(0.9*x).flatten()

    # Updated latent function
    if latent_function_id == 1:
        f_new = lambda x: -f(x)
    elif latent_function_id == 2:
        f_new = lambda x: (f(x) > 0) * f(x)
    elif latent_function_id == 3:
        f_new = lambda x: (x > 0).reshape(-1) * f(x)
    else:
        print("Unknown latent function ID: " + str(latent_function_id))

    #########################
    # Specify training data #
    #########################
    # Original training data
    X = np.random.uniform(-5, 5, size=(observation_points,1))
    X = np.sort(X,0)
    y = (f(X) + (noise_var**0.5)*np.random.randn(observation_points)).reshape(-1,1)

    # Updated training data
    X_new = np.random.uniform(-5, 5, size=(observation_points,1))
    X_new = np.sort(X_new,0)
    y_new = (f_new(X_new) + (noise_var**0.5)*np.random.randn(observation_points)).reshape(-1,1)

    # Combined training data
    X_comb = np.append(X, X_new).reshape(-1,1)
    y_comb = np.append(y, y_new).reshape(-1,1)

    #####################
    # Specify test data #
    #####################
    # Domain
    X_test = np.linspace(-5, 5, test_points).reshape(-1,1)

    # True latent function evaluation before the training set is modified
    f_test = f(X_test)

    # True latent function evaluation after the training set is modified
    f_test_new = f_new(X_test)

    #################################
    # Specify some helper functions #
    #################################

    # Optimize hyperparameters of a model, predict at test points
    def fit_and_predict(model, X, y, sample_noise, Xtest, **kwargs):
        # Set the training data
        model.set_XY(X, y, sample_noise=noise_var)

        # Fit the GP (optimize kernel hyperparameters)
        model.fit(optimize_sample_noise=True, random_restarts=num_restarts, verbose=False, **kwargs)

        # Perform prediction
        return model.predict(Xtest)

    # Calculate the residuals between two vectors and print to console
    def print_residuals(actual, predict, indent_level=0):
        if len(actual) != len(predict):
            print("Can't calculate residuals: unequal len of actual (" + str(len(actual)) + ") and predict (" + str(len(predict)) +")")
            return

        residuals = abs(actual.reshape(-1) - predict.reshape(-1))
        sum_res = np.sum(residuals)
        max_res = np.max(residuals)
        print(('\t'*indent_level) + "Residuals:\tsum=" + str(sum_res) + "\tmax=" + str(max_res))
        return sum_res, max_res

    # Print to stderr
    def eprint(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)

    ###############
    # Standard GPR #
    ###############

    #####
    # Standard GPR with original trend
    #####
    # Create GPR Kernel
    kernel_1 = copy.deepcopy(kernel)

    # Create GPR model
    model_1 = gpr_lib.GPRegressor(kernel_1, sig_var=1., sigvar_range=(1e-10,100))

    # Timer checkpoint
    t1 = time.time()

    # Predict model before the trend is changed
    mean_1a, var_1a = fit_and_predict(model_1, X, y, noise_var, X_test)
    std_1a = var_1a ** 0.5

    # Timer checkpoint
    t2 = time.time()
    dt_1 = t2 - t1

    #####
    # Standard GPR with new trend
    #####
    # Create GPR Kernel
    kernel_2 = copy.deepcopy(kernel)

    # Create GPR model
    model_2 = gpr_lib.GPRegressor(kernel_2, sig_var=1., sigvar_range=(1e-10,100))

    # Timer checkpoint
    t3 = time.time()

    # Predict model that only reflects the new trend
    mean_1b, var_1b = fit_and_predict(model_2, X_new, y_new, noise_var, X_test)
    std_1b = var_1b ** 0.5

    # Timer checkpoint
    t4 = time.time()
    dt_2 = t4 - t3

    #####
    # Standard GPR with combined trends
    #####
    # Create GPR Kernel
    kernel_3 = copy.deepcopy(kernel)

    # Create GPR model
    model_3 = gpr_lib.GPRegressor(kernel_3, sig_var=1., sigvar_range=(1e-10,100))

    # Timer checkpoint
    t5 = time.time()

    # Predict model that reflects the combined trends
    mean_1c, var_1c = fit_and_predict(model_3, X_comb, y_comb, noise_var, X_test)
    std_1c = var_1c ** 0.5

    # Timer checkpoint
    t6 = time.time()
    dt_3 = t6 - t5


    ############
    # GPR-STML #
    ############

    #####
    # GPR-STML with polynomial time function
    #####
    # Create GPR Kernel
    kernel_4 = copy.deepcopy(kernel)

    # Create GPR model
    model_4 = gpr_lib.GPRegressorSTML(kernel_4, sig_var=1.0, sigvar_range=(1e-10,100), sample_rate=50, forget_factor=0.001, time_function_type=4)

    # Timer checkpoint
    t7 = time.time()

    # Predict model that reflects the combined trends
    mean_4, var_4 = fit_and_predict(model_4, X_comb, y_comb, noise_var, X_test)
    std_4 = var_4 ** 0.5

    # Timer checkpoint
    t8 = time.time()
    dt_4 = t8 - t7

    #####
    # GPR-STML with exponential time function
    #####
    # Create GPR Kernel
    kernel_5 = copy.deepcopy(kernel)

    # Create GPR model
    model_5 = gpr_lib.GPRegressorSTML(kernel_5, sig_var=1.0, sigvar_range=(1e-10,100), sample_rate=50, forget_factor=0.001, time_function_type=3)

    # Timer checkpoint
    t9 = time.time()

    # Predict model that reflects the combined trends
    mean_5, var_5 = fit_and_predict(model_5, X_comb, y_comb, noise_var, X_test)
    std_5 = var_5 ** 0.5

    # Timer checkpoint
    t10 = time.time()
    dt_5 = t10 - t9


    ########################################################
    # Print residuals, hyperparameters, and execution time #
    ########################################################
    # Standard, original trend:
    print("Standard GPR: Original Trend")
    print("\tExecution time:\t" + str(dt_1))
    sr1, mr1 = print_residuals(mean_1a, f_test, indent_level=1)
    model_1.print_hprmtrs(indent_level=1)
    print('')

    # Standard, new trend: 
    print("Standard GPR: New Trend")
    print("\tExecution time:\t" + str(dt_2))
    sr2, mr2 = print_residuals(mean_1b, f_test_new, indent_level=1)
    model_1.print_hprmtrs(indent_level=1)
    print('')

    # Standard, combined trends:
    print("Standard GPR: Combined Trends")
    print("\tExecution time:\t" + str(dt_3))
    sr3, mr3 = print_residuals(mean_1c, f_test_new, indent_level=1)
    model_1.print_hprmtrs(indent_level=1)
    print('')

    # GPR-STML: polynomial time function:
    print("GPR-STML: Polynomial Time Function")
    print("\tExecution time:\t" + str(dt_4))
    sr4, mr4 = print_residuals(mean_4, f_test_new, indent_level=1)
    model_4.print_hprmtrs(indent_level=1)
    print('')

    # GPR-STML: exponential time function:
    print("GPR-STML: Exponential Time Function")
    print("\tExecution time:\t" + str(dt_5))
    sr5, mr5 = print_residuals(mean_5, f_test_new, indent_level=1)
    model_5.print_hprmtrs(indent_level=1)
    print('')

    ################################################
    # Check if we have an abnormally high residual #
    ################################################
    HIGH_RES = False
    sr_lim = 30

    # Skip standard combined (sr3) -> we already know it'll be high
    sr_array = np.array([sr1, sr2, 0, sr4, sr5])

    if (sr_array > sr_lim).any():
        HIGH_RES = True

    if HIGH_RES:
        # Print error file
        eprint('Abnormally high residual in test case(s): ', end='')
        num_left = np.sum(sr_array > sr_lim)
        for j, sr in enumerate(sr_array):
            if sr > sr_lim:
                eprint(str(j+1), end='')
                num_left -= 1
                if num_left > 0:
                    eprint(', ', end='')
        eprint('\n', end='')

    #############################
    # Generate predictive plots #
    #############################

    fig, ax = plt.subplots(3, 2, figsize=(8,8))

    ax[0,0].set_title("Standard GPR, Original Trend")
    ax[0,0].plot(X, y, 'k+', ms=12, label="Original Training Samples")
    ax[0,0].plot(X_test, f_test, 'b-', label="Latent Function")
    ax[0,0].plot(X_test, mean_1a, 'r--', lw=2, label="Predictive mean")
    ax[0,0].fill_between(X_test.ravel(), (mean_1a-(2*std_1a)).ravel(), (mean_1a+(2*std_1a)).ravel(), color="#dddddd", label="$2\sigma$ error bounds")
    ax[0,0].set_ylabel("System Output")
    ax[0,0].set_xlabel("System Input")
    ax[0,0].set_ylim((-1.6, 1.6))
    ax[0,0].set_xlim((-5, 5))

    ax[0,1].set_title("Standard GPR, New Trend")
    ax[0,1].plot(X_new, y_new, 'r+', ms=12, label="New Training Samples")
    ax[0,1].plot(X_test, f_test_new, 'b-', label="Latent Function")
    ax[0,1].plot(X_test, mean_1b, 'r--', lw=2, label="Predictive mean")
    ax[0,1].fill_between(X_test.ravel(), (mean_1b-(2*std_1b)).ravel(), (mean_1b+(2*std_1b)).ravel(), color="#dddddd", label="$2\sigma$ error bounds")
    ax[0,1].set_ylabel("System Output")
    ax[0,1].set_xlabel("System Input")
    ax[0,1].set_ylim((-1.6, 1.6))
    ax[0,1].set_xlim((-5, 5))

    ax[1,0].set_title("Standard GPR, Combined Trend")
    ax[1,0].plot(X, y, 'k+', ms=12, label="Original Training Samples")
    ax[1,0].plot(X_new, y_new, 'r+', ms=12, label="New Training Samples")
    ax[1,0].plot(X_test, f_test_new, 'b-', label="Latent Function")
    ax[1,0].plot(X_test, mean_1c, 'r--', lw=2, label="Predictive mean")
    ax[1,0].fill_between(X_test.ravel(), (mean_1c-(2*std_1c)).ravel(), (mean_1c+(2*std_1c)).ravel(), color="#dddddd", label="$2\sigma$ error bounds")
    ax[1,0].set_ylabel("System Output")
    ax[1,0].set_xlabel("System Input")
    ax[1,0].set_ylim((-1.6, 1.6))
    ax[1,0].set_xlim((-5, 5))

    ax[1,1].set_title("GPR-STML, Polynomial Time Function")
    ax[1,1].plot(X, y, 'k+', ms=12, label="Original Training Samples")
    ax[1,1].plot(X_new, y_new, 'r+', ms=12, label="New Training Samples")
    ax[1,1].plot(X_test, f_test_new, 'b-', label="Latent Function")
    ax[1,1].plot(X_test, mean_4, 'r--', lw=2, label="Predictive mean")
    ax[1,1].fill_between(X_test.ravel(), (mean_4-(2*std_4)).ravel(), (mean_4+(2*std_4)).ravel(), color="#dddddd", label="$2\sigma$ error bounds")
    ax[1,1].set_ylabel("System Output")
    ax[1,1].set_xlabel("System Input")
    ax[1,1].set_ylim((-1.6, 1.6))
    ax[1,1].set_xlim((-5, 5))

    ax[2,0].set_title("GPR-STML, Exponential Time Function")
    ax[2,0].plot(X, y, 'k+', ms=12, label="Original Training Samples")
    ax[2,0].plot(X_new, y_new, 'r+', ms=12, label="New Training Samples")
    ax[2,0].plot(X_test, f_test_new, 'b-', label="Latent Function")
    ax[2,0].plot(X_test, mean_5, 'r--', lw=2, label="Predictive mean")
    ax[2,0].fill_between(X_test.ravel(), (mean_5-(2*std_5)).ravel(), (mean_5+(2*std_5)).ravel(), color="#dddddd", label="$2\sigma$ error bounds")
    ax[2,0].set_ylabel("System Output")
    ax[2,0].set_xlabel("System Input")
    ax[2,0].set_ylim((-1.6, 1.6))
    ax[2,0].set_xlim((-5, 5))

    # Legend
    ax[2,1].clear()
    ax[2,1].set_axis_off()
    lines, labels = ax[2,0].get_legend_handles_labels()
    ax[2,1].legend(lines, labels, loc='center')

    # Show all figures
    plt.tight_layout()
    plt.show()
