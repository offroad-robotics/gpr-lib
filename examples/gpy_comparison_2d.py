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
# Regression on vibration data (output) from a mobile robot driving across
# a 2D terrain (input).
#
# Author: Jeremy Roy <jeremy.roy@queensu.ca>
# License: BSD 2.0

import GPy
from .. import scripts
from .ground_truth_2d_example import plot_ground_truth
import sys

import pickle

import numpy as np
import matplotlib.pyplot as plt

import time

def gpy_comparison_2d(ard=True, kernel="RBF"):
    '''
    Function showing example usage of the gpr_lib.GPRegressorSTML object for a
    2D input space.

    This file shows example usage of gpr-lib to perform Gaussian Process
    Regression on vibration data (output) from a mobile robot driving across
    a 2D terrain (input). See the README for a picture of the terrain.

    Kwargs:
        ard (float);
            Automatic Relevance Determination. If True, a seperate length scale
            is fit to each dimension.
        kernel (string);
            String describing the kernel to use in regression. Options are RBF, 
            Mat32, or Mat52.
    '''    

    ########################################
    # Specify latent function, test points #
    ########################################

    # Specify latent function
    with open('gpr_lib/data/2d_sample_data.pkl', 'rb') as f:  # Python 2.7: open(...)
        X, y = pickle.load(f, encoding="latin1") # Python 2.7: X, y = pickle.load(f)

    # Calculate map limits
    w_origin = 0.0
    h_origin = 0.0
    width = 860
    height = 860
    resolution = 0.01
    max_Kstar_len = 10000
    num_predict_groups = int((width * height) / max_Kstar_len)

    min_w = w_origin - (width*resolution)/2.
    max_w = w_origin + (width*resolution)/2.
    min_h = h_origin - (height*resolution)/2.
    max_h = h_origin + (height*resolution)/2.

    # Create iterable map indices
    W = np.linspace(min_w, max_w, width)
    H = np.linspace(min_h, max_h, height)

    # Create list of inputs
    Xtest = np.zeros((width*height,2))
    i = 0
    for h in H:
        for w in W:
            Xtest[i] = [w,h]
            i += 1

    # Prepare variables for plotting
    min_x = w_origin - (width*resolution)/2.
    max_x = w_origin + (width*resolution)/2.
    min_y = h_origin - (height*resolution)/2.
    max_y = h_origin + (height*resolution)/2.
    plot_extent = [min_x, max_x, min_y, max_x]

    #################
    # Using gpr_lib #
    #################
    print("\n\n*********Using gpr_lib:********\n")

    # Start Timer
    start_1 = time.time()

    # Create GPR Kernel
    if kernel == "RBF":
        kernel_1 = scripts.RBF(input_dim=2, l=0.1, ARD=ard)
    elif kernel == "Mat32":
        kernel_1 = scripts.Mat32(input_dim=2, l=0.1, ARD=ard)
    elif kernel == "Mat52":
        kernel_1 = scripts.Mat52(input_dim=2, l=0.1, ARD=ard)
    else:
        print("Error: cant handle kernel of type " + kernel)
        exit(1)

    # Create GPR model
    model_1 = scripts.GPRegressor(kernel_1, sig_var=1.)
    model_1.set_XY(X, y)

    # Fit the GP (optimize kernel hyperparameters)
    model_1.fit(random_restarts=3, verbose=True)

    # Print hyperparameters
    print("LML: " + str(float(model_1.LML)))
    if ard:
        print('l1: ' +  str(model_1.kernel.hprmtrs[0]) + '\tl2: ' +  str(model_1.kernel.hprmtrs[1]) + "\tsig_var: " + str(model_1.sig_var) + "\tnoise_var: " + str(model_1.sample_noise))
    else:
        print('l: ' +  str(model_1.kernel.hprmtrs[0]) + "\tsig_var: " + str(model_1.sig_var) + "\tnoise_var: " + str(model_1.sample_noise))

    # Perform prediction
    mean_1 = np.zeros(width*height)
    var_1 = np.zeros(width*height)
    # Make first (num_predict_groups * self.max_Kstar_len) predictions
    for i in range(num_predict_groups):
        print("Predict group " + str(i) + "/" + str(num_predict_groups))
        sys.stdout.write("\033[F\033[K")
        start = i * max_Kstar_len
        end = (i+1) * max_Kstar_len
        predicted = model_1.predict(Xtest[start:end])
        mean_1[start:end], var_1[start:end] = predicted[0][:,0], predicted[1][:,0]
    # Make leftover predictions
    predicted = model_1.predict(Xtest[end:])
    mean_1[end:], var_1[end:] = predicted[0][:,0], predicted[1][:,0]

    # Truncate negative values to 0
    for i in range(len(mean_1)):
        if mean_1[i] < 0.0:
            mean_1[i] = 0.0

    # Arrange mean_2 and var_2 in 2D array to be plotted
    mean_1_2d = np.zeros((height, width))
    var_1_2d = np.zeros((height, width))
    col = 0
    row = 0
    for i in range(len(mean_1)):
        mean_1_2d[row][col] = mean_1[i]
        var_1_2d[row][col] = var_1[i]
        # Increment column
        col += 1
        if col >= width:
            row += 1
            col = 0

    # Stop Timer
    stop_1 = time.time()
    elapsed_1 = stop_1 - start_1
    print("Time Elapsed: " + str(elapsed_1))

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3.5))
    ax1.set_title("gpr_lib: Predictive Mean")
    ax1.set_ylabel("x Position [m]")
    ax1.set_xlabel("y Position [m]")
    im1 = ax1.imshow(mean_1_2d, cmap='gray_r', origin='lower', extent=plot_extent)
    fig.colorbar(im1, ax=ax1)
    ax1.plot(X[:,0], X[:,1], '.r', alpha=0.5, label="Sample Points")
    # Overlay ground truth
    plot_ground_truth(ax1)


    #############
    # Using GPY #
    #############
    print("\n\n************Using GPy:************\n")

    # Start Timer
    start_2 = time.time()

    # Create GPR kernel
    if kernel == "RBF":
        kernel_2 = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=0.1, ARD=ard) 
    elif kernel == "Mat32":
        kernel_2 = GPy.kern.Matern32(input_dim=2, variance=1., lengthscale=0.1, ARD=ard) 
    elif kernel == "Mat52":
        kernel_2 = GPy.kern.Matern52(input_dim=2, variance=1., lengthscale=0.1, ARD=ard) 
    else:
        print("Error: cant handle kernel of type " + kernel)
        exit(1)

    # Create GPR model
    model_2 = GPy.models.GPRegression(X, y, kernel_2)

    # Fit the GP (optimize kernel hyperparameters)
    # model_2.optimize()
    model_2.optimize_restarts(num_restarts=4, messages=False)

    # Print hyperparameters
    print("LML: " + str(float(model_2._log_marginal_likelihood)))
    if kernel == "RBF":
        l_2 = model_2.rbf.lengthscale
        sig_var_2 = str(model_2.rbf.variance[0])
    elif kernel == "Mat32":
        l_2 = model_2.Mat32.lengthscale
        sig_var_2 = str(model_2.Mat32.variance[0])
    elif kernel == "Mat52":
        l_2 = model_2.Mat52.lengthscale
        sig_var_2 = str(model_2.Mat52.variance[0])

    if ard:
        print("l1: " + str(l_2[0]) + "\tl2: " + str(l_2[1]) + "\tsig_var: " + sig_var_2 + "\tnoise_var: " + str(model_2.Gaussian_noise.variance[0]))
    else:
        print("l: " + str(l_2[0]) + "\tsig_var: " + sig_var_2 + "\tnoise_var: " + str(model_2.Gaussian_noise.variance[0]))

    # Perform prediction
    mean_2 = np.zeros(width*height)
    var_2 = np.zeros(width*height)
    num_predict_groups = int((width * height) / max_Kstar_len)
    # Make first (num_predict_groups * self.max_Kstar_len) predictions
    for i in range(num_predict_groups):
        print("Predict group " + str(i) + "/" + str(num_predict_groups) + "\r")
        sys.stdout.write("\033[F\033[K")
        start = i * max_Kstar_len
        end = (i+1) * max_Kstar_len
        predicted = model_2.predict(Xtest[start:end],include_likelihood=False)
        mean_2[start:end], var_2[start:end] = predicted[0][:,0], predicted[1][:,0]
    # Make leftover predictions
    predicted = model_2.predict(Xtest[end:],include_likelihood=False)
    mean_2[end:], var_2[end:] = predicted[0][:,0], predicted[1][:,0]

    # Truncate negative values to 0
    for i in range(len(mean_2)):
        if mean_2[i] < 0.0:
            mean_2[i] = 0.0

    # Arrange mean_2 and var_2 in 2D array to be plotted
    mean_2_2d = np.zeros((height, width))
    var_2_2d = np.zeros((height, width))
    col = 0
    row = 0
    for i in range(len(mean_2)):
        mean_2_2d[row][col] = mean_2[i]
        var_2_2d[row][col] = var_2[i]
        # Increment column
        col += 1
        if col >= width:
            row += 1
            col = 0

    # Stop Timer
    stop_2 = time.time()
    elapsed_2 = stop_2 - start_2
    print("Time Elapsed: " + str(elapsed_2))

    # Plot results
    ax2.set_title("GPy: Predictive Mean")
    ax2.set_ylabel("x Position [m]")
    ax2.set_xlabel("y Position [m]")
    im2 = ax2.imshow(mean_2_2d, cmap='gray_r', origin='lower', extent=plot_extent)
    fig.colorbar(im2, ax=ax2)
    ax2.plot(X[:,0], X[:,1], '.r', alpha=0.5, label="Sample Points")
    ax2.legend()
    # Overlay ground truth
    plot_ground_truth(ax2)

    plt.tight_layout()
    plt.show()