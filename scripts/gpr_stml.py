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

# Author: Jeremy Roy <jeremy.roy@queensu.ca>
# License: BSD 2.0

from .gpr_lib import GPRegressor, is_vector

import numpy as np
from scipy.linalg import toeplitz

class GPRegressorSTML(GPRegressor):
    '''
    Periodic GPR-STML Regressor.

    This regressor assumes that the time between each training sample is
    identical (periodic sampling, no gaps).
    '''

    def __init__(self, kernel, sample_rate=50., forget_factor=0., forget_factor_range=(0,float('inf')), time_function_type=1, config=1, **kwargs):
        '''
        Creates a GPR-STML Regressor object.


        Args:
            kernel (Kernel):
                A Kernel object - a measure of distance in the sample's input space.

        Kwargs:
            sample_rate (float):
                Sampling rate of the input samples.
            forget_factor (float):
                The initial forgetting factor of the STML regressor.
            forget_factor_range (2-tuple):
                The allowable (min,max) range of the forgetting factor during optimization
            time_function_type (int):
                Time function to use.
                    1: Linear time function
                    2: Quadratic time function
                    3: Exponential time function
                    4: Polynomial time function
                If set to 3 or 4, the initial degree of the function (alpha) is set to
                the value of forget_factor.
            config (int):
                For research/debugging purposes.  Use default value of 1.
                How the discount D is included in K_y.
                    1: Addition
                    2: Element-wise multiplication
            Inherited arguments (from GPRegressor):
                sig_var, sigvar_range, samplenoise_range.

        '''
        super(GPRegressorSTML, self).__init__(kernel, **kwargs)
        self.time_function_type = int(time_function_type)
        self.ff = np.array(forget_factor).reshape(1)
        if type(forget_factor_range) is tuple:
            self.ff_range = [forget_factor_range]
        elif (type(forget_factor_range) is list) and (forget_factor_range[0] == 2):
            assert(len(forget_factor_range) == len(forget_factor))
            self.ff_range = forget_factor_range
        else:
            print("Invalid range for forgetting factor.  Using default values.")
            self.ff_range = [(0,float('inf'))]
        # Exponential and polynomial time functions have both a forgetting factor
        # and a degree of the function.  Both of these variables are stored in
        # self.ff...  See calc_T for usage.
        if ((self.time_function_type == 3) or (self.time_function_type == 4)) and (len(self.ff) == 1):
            self.ff = np.append(self.ff, self.ff)
            self.ff_range.append(self.ff_range[0])
        self.sample_rate = float(sample_rate)
        self.config = config

    def calc_T(self, func_type=1, *args):
        '''
        Calculate time matrix.

        Args:
            Variables used when calculating the time matrix. The number of
            variables and their usage depends on the specified time function.

        Kwargs:
            func_type (int):
                Time function to use.
                    1: Linear time function (one arg: ff)
                    2: Quadratic time function (one arg: ff)
                    3: Exponential time function (two args: ff, deg)
                    4: Polynomial time function (two args: ff, deg)

        Returns:
            The time matrix.
        '''
        if func_type == 1: # Linear time function
            # Default to member forgetting factor if none specified
            if len(args) > 0:
                ff = args[0]
            else:
                ff = self.ff[0]
            # Calculate time array
            T = np.arange(0, self.num_train_samples) / self.sample_rate
            # Multiply by forgetting factor
            T *= ff
        elif func_type == 2: # Quadratic time function
            # Default to member forgetting factor if none specified
            if len(args) > 0:
                ff = args[0]
            else:
                ff = self.ff[0]
            # Calculate time array
            T = np.arange(0, self.num_train_samples) / self.sample_rate
            T *= T
            # Multiply by forgetting factor
            T *= ff
        elif func_type == 3: # Exponential time function
            # Default to member forgetting factor if none specified
            if len(args) > 1:
                ff = args[0]
                alpha = args[1]
            elif len(args) > 0:
                ff = args[0]
                alpha = 1
            else:
                ff = self.ff[0]
                alpha = self.ff[1]

            # Calculate time array
            T = np.arange(0, self.num_train_samples) / self.sample_rate
            # Multiply by forgetting factor
            T = ff * (np.exp(alpha * T) - 1)
        elif func_type == 4: # Polynomial time function
            # Default to member forgetting factor if none specified
            if len(args) > 1:
                ff = args[0]
                alpha = args[1]
            elif len(args) > 0:
                ff = args[0]
                alpha = 1
            else:
                ff = self.ff[0]
                alpha = self.ff[1]

            # Calculate time array
            T = np.arange(0, self.num_train_samples) / self.sample_rate
            T = T ** alpha
            # Multiply by forgetting factor
            T *= ff
        return T

    def noisy_gram_matrix(self, kernel=None, sample_noise=None, sig_var=None, *args):
        '''
        Calculate the spatio-temporal memory loss (STML)-augmented Gram matrix.

        Optionally calculate the gram matrix for a kernel, sample noise, and/or
        signal variance that don't belong to this regressor object via Kwargs.
        '''
        # Default to member variables if none specified
        if kernel is None:
            kernel = self.kernel
        if sample_noise is None:
            sample_noise = self.sample_noise
        if sig_var is None:
            sig_var = self.sig_var

        # Get sample noise matrix
        if not is_vector(sample_noise):
            sample_noise = np.ones(self.num_train_samples) * sample_noise

        # Calculate the Gram matrix
        Kf = sig_var * kernel.evaluate(self.X, self.X)

        # Include signal variance and noise in the modified Gram matrix
        Ky = Kf
        Ky.ravel()[::Ky.shape[1]+1] += sample_noise.reshape(-1)

        # Calculate the spatio-temporal memory loss terms
        time_func = self.calc_T(self.time_function_type, *args)
        # time = np.triu(toeplitz(time_func))
        time = toeplitz(np.zeros(len(time_func)), time_func)
        D_mat = np.multiply(np.triu(Kf,1), time)
        D = np.sum(D_mat,1)
        if self.config == 2: # Multiply K_y by D (elem-wise) - need to make sure K_y is PD
            D += 1

        # Add the spatio-temporal memory loss terms to the Gram matrix
        if self.config == 1:
            Ky.ravel()[::Ky.shape[1]+1] += D
        elif self.config == 2:
            Ky.ravel()[::Ky.shape[1]+1] *= D

        return Ky

    def __fit__(self, x0, bounds, optimize_ff=False, optimize_sample_noise=True, optimize_sig_var=True, **kwargs):
        '''
        Hidden function that performs the hyperparameter optimization.

        Wraps the GPRegressor's __fit__() function with additional logic to
        include the forgetting factor(s) in the set of hyperparameters to
        optimize.
        '''
        # Specify prior hyperparameters and bounds on hyperparameters
        if optimize_ff:
            # for i in range(len(self.ff)):
            #     x0.append(self.ff[i])
            #     bounds.append(self.ff_range[i])
            x0.append(self.ff[0])
            bounds.append(self.ff_range[0])
            if (self.time_function_type == 3) or (self.time_function_type == 4):
                x0.append(self.ff[1])
                bounds.append(self.ff_range[1])

        # Call the parent's __fit__ function to run optimization
        res = super(GPRegressorSTML, self).__fit__(x0, bounds, **kwargs)

        # Set the forgetting factor
        n = len(self.kernel.hprmtrs)
        if optimize_ff:
            if optimize_sample_noise and optimize_sig_var:
                self.ff = np.array(res.x[n+2:])
            elif optimize_sample_noise or optimize_sig_var:
                self.ff = np.array(res.x[n+1:])
            else:
                self.ff = np.array(res.x[n:])

        # Return the result
        return res

    def fit(self, optimize_ff=True, **kwargs):
        '''
        Fits the GP to the training data by maximizing the Log Marginal Likelihood.

        Kwargs:
            optimize_ff (bool):
                If True, includes time function's forgetting factor(s) in the set
                of parameters to optimize.
            Inherited arguments (from GPRegressor):
                optimize_sample_noise, optimize_sig_var, random_restarts, kwargs
                for __fit__().
        '''
        # Call the parent's fit function to run optimization and multiple restarts
        return super(GPRegressorSTML, self).fit(optimize_ff=optimize_ff, **kwargs)

    def print_hprmtrs(self, append_new_line=False, indent_level=0):
        '''
        Prints the hyperparameters of the model to stdout, including the
        forgetting factor(s).
        '''
        super(GPRegressorSTML, self).print_hprmtrs(append_new_line=False, indent_level=indent_level)      
        print(('\t'*indent_level) + 'Forgetting Factor: ' + str(self.ff))
        if append_new_line:
            print('')