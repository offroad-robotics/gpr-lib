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

from __future__ import print_function
import pickle
import sys
from pathlib2 import Path

from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.linalg.lapack import dpotrs

import time

import numpy as np

import copy

def is_vector(var):
    '''
    Checks if a variable is an array-like vector.

    Args:
        var (array_like): Some variable.

    Returns:
        True if the variable is an array-like vector, False otherwise.
    '''
    if hasattr(var, "__len__") and not isinstance(var,str):
        # It's array-like.  Check it's shape
        var = np.array(var)
        if (len(var.shape) != 2) or (var.shape[1] != 1):
            vector = False
        else:
            vector = True
    else:
        vector = False
    return vector


def jitchol(A, max_tries=5):
    '''
    Computes the Cholesky decomposition of A using np.linalg.cholesky().

    Retries by adding increasing the jitter to the diagonal of A until the
    Cholesky decomposition succeeds.
    Inspired by the jitchol function in GPy (https://gpy.readthedocs.io/en/deploy/_modules/GPy/util/linalg.html#jitchol)

    Args:
        A (array_like):
            Hermitian (symmetric if all elements are real), positive-definite
            input matrix.

    Kwargs:
        max_tries (int):
            The maximum number of times to increase the diagonal jitter of A.
            Jitter is increased by a decade each try, starting at 1e-6.
    '''
    try:
        # Compute normal cholesky decomposition
        L = np.linalg.cholesky(A)
        return L
    except Exception:
        # Failure.  Try adding jitter to the diagonal...
        diagA = np.diag(A)
        if np.any(diagA <= 0):
            print("Failed to compute Cholesky decomposition: The diagonal of A contains a non-positive number", file=sys.stderr)
            return None
        jitter = diagA.mean() * 1e-6
        num_tries = 1
        while num_tries <= max_tries and np.isfinite(jitter):
            try:
                L = np.linalg.cholesky(A)
                return L
            except: 
                jitter *= 10
            num_tries += 1
    # Failed to compute Cholesky decomposition
    print("Failed to compute Cholesky decomposition: A is not positive definite, even with jitter.", file=sys.stderr)
    return None


def LML(hprmtrs, model, optimize_sample_noise, optimize_sig_var):
    '''Log marginal likelyhood function.  Used to optimize the kernel hyperparameters.'''
    # Make a local copy of the kernel (to preserve kernel type)
    kernel = copy.deepcopy(model.kernel)

    # Make sure we have enough hyperparameters
    n = len(kernel.hprmtrs)
    if optimize_sample_noise and optimize_sig_var:
        min_hprmtrs = n + 2
    elif optimize_sample_noise or optimize_sig_var:
        min_hprmtrs = n + 1
    else:
        min_hprmtrs = n
    
    N = len(hprmtrs) - min_hprmtrs
    if N < 0:
        print("Error: invalid number of hyperparameters specified in LML: " + str(len(hprmtrs)), file=sys.stderr)
        print("\tExpected minimum of " + min_hprmtrs + " hyperparameters", file=sys.stderr)
        exit(1)
    
    # Set the kernel hyperparameters
    kernel.hprmtrs = hprmtrs[0:n]

    # Change behaviour w.r.t. args
    if optimize_sig_var and optimize_sample_noise:
        sig_var = hprmtrs[n]
        sample_noise = np.ones(model.num_train_samples) * hprmtrs[n+1]
    elif optimize_sig_var:
        sig_var = hprmtrs[n]
        if not is_vector(model.sample_noise):
            sample_noise = np.ones(model.num_train_samples) * model.sample_noise
    elif optimize_sample_noise:
        sig_var = model.sig_var
        sample_noise = np.ones(model.num_train_samples) * hprmtrs[n]
    else:
        sig_var = model.sig_var
        if not is_vector(model.sample_noise):
            sample_noise = np.ones(model.num_train_samples) * model.sample_noise

    # Collect un-used hyperparameters in a tuple
    extra_hprmtrs = hprmtrs[min_hprmtrs:]

    # Calculate the noise-augmented Gram matrix
    Ky = model.noisy_gram_matrix(kernel, sample_noise, sig_var, *extra_hprmtrs)

    # Calculate the quatratic of inverse: Y.T(Ky^-1)Y
    Ky_chol = jitchol(Ky)
    if Ky_chol is None:
        # Failed to calculate a cholesky decomposition, even with jitter
        # Return an impossibly low LML
        return -float('inf')

    # Make sure everything is in the right shape for dpotrs
    Y = model.Y.reshape(-1,1)
    try:
        # Compute dpotrs
        alpha, _ = dpotrs(Ky_chol, Y, lower=True)
    except Exception as e:
        # Print error
        print(e, file=sys.stderr)
        # Save program state to logfile
        error_log_dir = "/home/offroad/temp/"
        Path(error_log_dir).mkdir(exist_ok=True)
        error_log_filename = error_log_dir + "error" + str(time.time()) + ".pkl"
        with open(error_log_filename, 'wb') as f:
            pickle.dump([Y, model.Y, Ky_chol, model], f)
        print("Model state has been saved to " + error_log_filename, file=sys.stderr)
        exit(1)

    a = np.sum(alpha * model.Y)

    # Calculate the natural log of the determinant of Ky
    # This is more numerically stable than calculating the 
    # determinant of Ky and then taking the natural logarithm,
    # since det(Ky) could be very small or very large and, due
    # to numerical errors, return a value of zero.
    _, logdet_Ky = np.linalg.slogdet(Ky)            

    # Calculate and return the LML
    b = logdet_Ky
    c = model.num_train_samples * np.log(2*np.pi)
    LML = -0.5 * (a + b + c)
    return LML

def neg_LML(hprmtrs, *args):
    ''' Returns the negative log marginal likelihood of the hyperparameters'''
    return - LML(hprmtrs, *args)

class Kernel(object):
    '''Superclass for kernels.'''
    def __init__(self, input_dim=2):
        self.input_dim = input_dim
        self.hprmtrs = []
        self.hprmtrs_range = []
        pass
    
    def evaluate(self, full_covariance=False):
        '''All Kernel objects MUST have an evaluate function.'''
        return NotImplementedError()

class RBF(Kernel):
    '''RBF kernel'''

    def __init__(self, input_dim=2, l=1, l_range=(1e-10,float('inf')), ARD=False):
        '''
        Create an RBF Kernel.

        Kwargs:
            input_dim (int):
                The number of input dimensions
            l (int):
                The initial length scale of the kernel along each input
                dimension.
            l_range (2-tuple):
                The (min,max) values of l
            ARD (bool):
                If True, uses independent length-scales for each input
                dimension. If false, uses a single length-scale for each input
                dimension.
        '''
        super(RBF, self).__init__(input_dim)
        assert(len(l_range) == 2)
        if ARD:
            for i in range(input_dim):
                self.hprmtrs.append(l)
                self.hprmtrs_range.append(l_range)
        else:
            self.hprmtrs.append(l)
            self.hprmtrs_range.append(l_range)
        self.ARD = ARD
        self.num_hyperparameters = 2

    def sqdist(self, X1, X2):
        '''
        Calculate the squared Euclidean distance between two sets of input data.

        Args:
            X1 (array_like): Input data 1 of length N.
            X2 (array_like): Input data 2 of length M.

        Returns:
            An NxM numpy array of distances between the samples from X1 and X2.
        '''
        if self.ARD:
            # V = self.hprmtrs[0:self.input_dim]
            # diff = np.repeats(X1, X2.shape[0], axis=1) - np.repeats(X2, X1.shape[1], axis=0)
            # return np.sqrt(np.dot(np.dot(diff, np.diag(1/V)), diff.T))
            return self.dist(X1, X2)**2
        else:
            return cdist(X1,X2, metric='sqeuclidean')

    def dist(self, X1, X2):
        '''
        Calculate the Euclidean distance between two sets of input data.

        Args:
            X1 (array_like): Input data 1 of length N.
            X2 (array_like): Input data 2 of length M.

        Returns:
            An NxM numpy array of distances between the samples from X1 and X2.
        '''
        if self.ARD:
            V = np.array(self.hprmtrs[0:self.input_dim])**2
            return cdist(X1,X2,V=V, metric='seuclidean')
        else:
            return cdist(X1,X2, metric='euclidean')
    
    def evaluate(self, X1, X2, full_covariance=True):
        '''
        Evaluate the kernel at all combinations of inputs in X1 and X2

        Args:
            X1 (array_like): Input data 1 of length N.
            X2 (array_like): Input data 2 of length M.

        Kwargs:
            full_covariance (bool): If True, calculates a full covariance

        Returns:
            An NxM numpy array of distances between the samples from X1 and X2.
        '''
        if (not full_covariance) and (X1.shape == X2.shape):
            return np.ones(X1.shape[0]).reshape(-1,1)
        else:
            if self.ARD:
                a = -0.5 * self.sqdist(X1, X2)
            else:
                a = -0.5 * (1/(self.hprmtrs[0]**2)) * self.sqdist(X1, X2)
            return np.exp(a)


class Mat32(RBF):
    '''Matern kernel with v=3/2'''
    def __init__(self, **kwargs):
        '''
        Create a Matern kernel with v=3/2.

        See Kwargs of RBF Kernel's constructor.
        '''
        super(Mat32, self).__init__(**kwargs)
    
    # Evaluate the kernel at all combinations of inputs in X1 and X2
    def evaluate(self, X1, X2, full_covariance=True):
        '''
        Evaluate the kernel at all combinations of inputs in X1 and X2

        Args:
            X1 (array_like): Input data 1 of length N.
            X2 (array_like): Input data 2 of length M.

        Kwargs:
            full_covariance (bool): If True, calculates a full covariance

        Returns:
            An NxM numpy array of distances between the samples from X1 and X2.
        '''
        if (not full_covariance) and (X1.shape == X2.shape):
            return np.ones(X1.shape[0]).reshape(-1,1)
        else:
            if self.ARD:
                a = np.sqrt(3) * self.dist(X1, X2)
            else:
                a = np.sqrt(3) * self.dist(X1, X2) / self.hprmtrs[0]
            return ( 1 + a ) * np.exp(-a)

class Mat52(RBF):
    '''Matern kernel with v=5/2'''
    def __init__(self, **kwargs):
        '''
        Create a Matern kernel with v=5/2.

        See Kwargs of RBF Kernel's constructor.
        '''
        super(Mat52, self).__init__(**kwargs)
    
    # Evaluate the kernel at all combinations of inputs in X1 and X2 
    def evaluate(self, X1, X2, full_covariance=True):
        '''
        Evaluate the kernel at all combinations of inputs in X1 and X2

        Args:
            X1 (array_like): Input data 1 of length N.
            X2 (array_like): Input data 2 of length M.

        Kwargs:
            full_covariance (bool): If True, calculates a full covariance

        Returns:
            An NxM numpy array of distances between the samples from X1 and X2.
        '''
        if (not full_covariance) and (X1.shape == X2.shape):
            return np.ones(X1.shape[0]).reshape(-1,1)
        else:
            r = self.dist(X1, X2)
            r_sqr = r**2
            if self.ARD:
                a = np.sqrt(5) * r
                b = (5. * r_sqr) / 3.
            else:
                a = np.sqrt(5) * r / self.hprmtrs[0]
                b = (5. * r_sqr) / (3. * self.hprmtrs[0]**2)
            return ( 1 + a + b) * np.exp(-a)

# Class that fits GPs to training data and calculates regression 
# predictions for test data
class GPRegressor(object):
    def __init__(self, kernel, sig_var=1., sigvar_range=(1e-10,float('inf')), samplenoise_range=(1e-10,float('inf'))):
        '''
        Creates a GPRegressor object

        Args:
            kernel (Kernel):
                A Kernel object - a measure of distance in the sample's input space.

        Kwargs:
            sig_var (float):
                The default signal variance.
            sigvar_range (2-tuple):
                The allowable (min,max) range of the signal's variance during optimization.
            samplenoise_range (2-tuple):
                The allowable (min,max) range of the signal's sample noise during optimization.
        '''
        self.kernel = kernel

        # Initialize class attributes
        self.training_set_changed = True
        self.hyperparameters_changed = False
        self._C = None
        self.X = np.empty((0, self.kernel.input_dim))
        self.Y = np.empty((0, 1))
        self.num_train_samples = 0
        self.LML = None
        self.sig_var = sig_var
        
        assert(len(sigvar_range) == 2)
        self.sigvar_range = sigvar_range
        assert(len(samplenoise_range) == 2)
        self.samplenoise_range = samplenoise_range

    def noisy_gram_matrix(self, kernel=None, sample_noise=None, sig_var=None, *args):
        '''
        Calculate the Gram matrix with noise

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
        
        # Get saple noise matrix
        if not is_vector(sample_noise):
            sample_noise = np.ones(self.num_train_samples) * sample_noise
        
        # Calculate the Gram matrix with noise.
        Ky = sig_var * kernel.evaluate(self.X, self.X)
        Ky.ravel()[::Ky.shape[1]+1] += sample_noise.reshape(-1)
        
        return Ky

    def calc_C(self):
        '''Calculates the inverted, noisy Gram matrix.'''
        # Only re-calculates if the training set or hyperparameters have changed.
        if self.training_set_changed or self.hyperparameters_changed:
            self.training_set_changed = False
            self.hyperparameters_changed = False

            # Get sample noise matrix
            Ky = self.noisy_gram_matrix()

            # Calculate C
            self._C = np.linalg.inv(Ky)

        return self._C
            
    # Specify the training data
    def set_XY(self, X, Y, sample_noise=1.):
        '''
        Specify the training data.

        Args:
            X (array_like):
                Array of input data.  Must match the Kernel's input dimension.
            Y (array_like):
                Array of output data.  Only 1D data is supported.

        Kwargs:
            sample_noise (float or array_like):
                The training data's sample noise.  Must be greater than zero.
                Otherwise it defaults to a value of 1.
        '''
        self.training_set_changed = True
        # Set the training data
        if X.shape[1] != self.kernel.input_dim:
            print("Error: X not same dimension " + str(X.shape[1]) + " as specified input dimension " + str(self.kernel.input_dim), file=sys.stderr)
            return False
        if X.shape[0] != Y.shape[0]:
            print("Error: Number of X samples " + str(X.shape[0]) + " not equal to number of Y samples " + str(Y.shape[0]), file=sys.stderr)
            return False
        self.X = X # TODO: should I make a deep copy?
        self.Y = Y
        self.num_train_samples = X.shape[0]
        if (sample_noise < 0):
            self.set_sample_noise(None)
        else:
            self.set_sample_noise(sample_noise)

    def set_sample_noise(self, sample_noise=1.):
        '''
        Specify the sample noise, either as an integer or as a 1D array.

        If the Sample noise is None, then it is by default set to 1.
        '''
        if sample_noise is None:
            self.sample_noise = 1.
        elif is_vector(sample_noise):
            # We're dealing with a vector
            if self.num_train_samples != sample_noise.shape[0]:
                print("Error: Number of noise samples " + str(sample_noise.shape[0]) + " not equal to number of training samples " + str(self.num_train_samples), file=sys.stderr)
            else:
                self.sample_noise = np.array(sample_noise) # Make sure it's a numpy array
        else: # We're dealing with a scalar
            self.sample_noise = float(sample_noise)


    def __fit__(self, x0, bounds, optimize_sample_noise=True, optimize_sig_var=True, random_restarts=0, method='L-BFGS-B', maxiter=1000, verbose=False, **kwargs):
        '''Hidden function that performs the hyperparameter optimization with multiple restarts'''

        # Perform optimization: Maximize the log marginal likelyhood
        best_res = None
        assert random_restarts >= 0
        for i in range(random_restarts + 1):
            # Maximize the log marginal likelyhood
            res = minimize(neg_LML, x0, bounds=bounds, method=method, args=(self, optimize_sample_noise, optimize_sig_var), options={'maxiter':maxiter, 'maxfun':maxiter})
            
            # TODO: See if using f_min_l_bfgs_b could speed up optimization.
            #       I might need to provide a gradient for the function.
            # fmin_l_bfgs_b(neg_LML, x0, approx_grad=True)
            
            if best_res is None:
                best_res = res
            elif res.fun < best_res.fun:
                best_res = res

            if random_restarts > 0:
                if verbose:
                    print("Optimization restart {0}/{1}, f = {2}".format(i, random_restarts, res.fun))
                    print("\tStarting Hyperparameters:\t" + str(x0))
                    print("\tEnding Hyperparameters:\t\t" + str(res.x))
                # Randomize the hyperparameter values for the next optimization run
                x0 = np.abs(np.random.randn(len(x0)))

        return best_res # Return the best optimization result


    def fit(self, optimize_sample_noise=True, optimize_sig_var=True, **kwargs):
        '''
        Fits the GP to the training data by maximizing the Log Marginal Likelihood.

        Observation: GPy's hyperparameter optimization is much more efficient
                     than this.  Why?  Do they have a more efficient way of
                     calculating the LML?

        Kwargs:
            optimize_sample_noise (bool):
                If True, considers the sample noise as a parameter to optimize.
            optimize_sig_var (bool):
                If True, considers the input signal's variance as a parameter
                to optimize.
            random_restarts (int):
                After an optimization run that uses the specified initial
                hyperparameters, additional optimizations using randomly
                generated hyperparameters will be run this many times.
            Other arguments:
                May be used by a child classe's __fit__() function.

        '''
        # Specify the prior hyperparameters
        x0 = copy.deepcopy(self.kernel.hprmtrs)
        if optimize_sig_var:
            x0.append(self.sig_var)
        if optimize_sample_noise:
            if is_vector(self.sample_noise):
                x0.append(self.sample_noise[0])
            else:
                x0.append(self.sample_noise)
        
        # Specify the bounds on the hyperparameters
        bounds = []
        for i in range(len(self.kernel.hprmtrs)):
            bounds.append(self.kernel.hprmtrs_range[i])
        if optimize_sig_var:
            bounds.append(self.sigvar_range)
        if optimize_sample_noise:
            bounds.append(self.samplenoise_range)
        
        # Perform optimization: Maximize the log marginal likelyhood
        res = self.__fit__(x0, bounds, optimize_sample_noise=optimize_sample_noise, optimize_sig_var=optimize_sig_var, **kwargs)

        # Set the hyperparameters to the optimized values
        n = len(self.kernel.hprmtrs)
        self.kernel.hprmtrs = res.x[0:n].tolist()
        if optimize_sample_noise and optimize_sig_var:
            self.sig_var = res.x[n]
            self.sample_noise = res.x[n+1]
        elif optimize_sample_noise:
            self.sample_noise = res.x[n]
        elif optimize_sig_var:
            self.sig_var = res.x[n]
        
        self.LML = -res.fun

        self.hyperparameters_changed = True

        return res
    
    def predict(self, X_test, full_covariance=False):
        '''
        Calculates the predictive mean and (co)variance at the specified points.

        Args:
            X_test (array-like):
                A list of points at which the GP should be sampled.

        Kwargs:
            full_covariance:
                If true, the entire predictive covariance is calculated.
                If false, only the predictive variance is calculated.

        Returns:
            mean, covariance:
                The predictive mean and (co)variance for points X_test.
        '''
        # Calculate the Cholesky-decomposed Gram matrix
        C = self.calc_C()

        # Calculate other kernel matrices
        K_star = self.sig_var * self.kernel.evaluate(self.X, X_test)
        K_test = self.sig_var * self.kernel.evaluate(X_test, X_test, full_covariance)
        
        # Calculate the predictive mean
        C_star = np.dot(K_star.T, C)
        mean = np.dot(C_star, self.Y)

        # Calculate predictive covariance
        if full_covariance:
            C_star2 = np.dot(C_star, K_star)
            covariance = K_test - C_star2

            return mean, covariance
        else:
            # Only calculate the diagonal of C_star2
            C_star2 = (C_star * K_star.T).sum(-1).reshape(-1,1)
            variance = K_test - C_star2

            return mean, variance
    
    def print_hprmtrs(self, append_new_line=False, indent_level=0):
        '''
        Prints the hyperparameters of the model to stdout
        '''
        print(('\t'*indent_level) + "LML: " + str(float(self.LML)))
        print(('\t'*indent_level) + 'l: ' +  str(self.kernel.hprmtrs) + "\tsig_var: " + str(self.sig_var) + "\tnoise_var: " + str(self.sample_noise))
        if append_new_line:
            print('')