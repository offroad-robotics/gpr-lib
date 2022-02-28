<h1 style="text-align:center">GPR-Lib</h1>

This GPy-inspired library provides objects and methods for performing Gaussian Process Regression, both with and without SpatialTemporal Memory Loss.  It is compatible with both Python 2.7 and Python 3.*.

# Installaling dependencies

This library makes use of several third-party libraries.  We recommend installing them in a python venv.  Note that the list of dependencies in `requirements.txt` is set for Python3.8.  For different version of Python, different version of the libraries in `requirements.txt` might be required.  E.g. for Python2.7 and Python3.5, use `numpy==1.16.04, pathlib2==2.3.2, scipy==1.1.0, matplotlib==2.2.2`

    python3.8 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

To exit the venv, type `deactivate`.  To re-enter the venv, type `source venv/bin/activate`.

# Using this library

Example usage of the `GPRegressor` class with comparisons to GPy is available in the files `gpy_comparison_1d.py` and `gpy_comparison_2d.py`.

Example usage of the `GPRegressorSTML` class is available in the file `gpr_stml_example.py`.

## Creating a kernel

This library currently implements three kernels: `RBF`, `Mat32`, and `Mat52`.  Other kernels can be implemented by inheriting from the `Kernel` class, and must implement an `evaluate()` function that returns a 2D np.array of distance measurements.  See the `RBF` class implementation in `gpr_lib.py` for an example.

Example: An RBF kernel for a 2D input space can be specified as follows:

    kernel = gpr_lib.RBF(input_dim=2, l=0.1, ARD=True)

where `l` is the initial length-scale (before optimization) and `ARD` selects whether the length scale along each input dimension is independently optimized.

## Creating a regressor

This library implements two regressors: `GPRegressor` and `GPRegressorSTML`.  The `GPRegressor` is a standard Gaussian Process regressor whose Kernel hyperparameters (e.g. length scale), sampling noise, and signal variance can each be jointly optimized to fit to the training data.

Example: A standard GPR regressor can be created as follows:

    model = gpr_lib.GPRegressor(kernel, sig_var=1.0)

The `GPRegressorSTML` implements the Gausian Process Regression with SpatioTemporal Memory Loss (GPR-STML) algorithm for the case where all training samples are taken periodically and contain no temporal gaps.  The time function, whose hyperparameters can be automatically optimized, is chosen via the `time_function_type` parameter in the constructor.

Example: A GPR-STML regressor can be created as follows:

    model = gpr_lib.GPRegressorSTML(kernel, sig_var=1.0, sample_rate=50, forget_factor=0.001, time_function_type=4)

For more information, see the api documentation under `docs/_build/html/index.html`

## Creating a fitted model

The following code snippet shows an example of how the regressor can be fit to a given input data.  Note that `X` is a numpy array of all of the system's sample locations, `y` is a numpy array of the sample values, and `noise_var` is the variance of the sample noise. For example data, see the provided examples (`gpr_lib/examples`).

```python
# Create GPR Kernel
kernel = scripts.RBF(input_dim=1, l=0.1, ARD=True)

# Create GPR model
model = scripts.GPRegressor(kernel, sig_var=1.)
model.set_XY(X, y, sample_noise=noise_var)    

# Fit the GP (optimize kernel hyperparameters)
model.fit()
```

## Predicting the system output at new locations

Once a model is fit, it can be used to predict the value of the system at new locations, `Xtest` (numpy array of test coordinates).

```python
predictive_mean, predictive_variance = model.predict(Xtest)
```

# Running the examples

The `examples` directory of this project contains three example usages of `gpr_lib`. The first two examples (`gpy_comparison_`) are provided to compare the interface, output, and performance of `gpr_lib` and `gpy` for 1D and 2D output spaces.  The final example provides an example usage of the `GPRegressorSTML` to perform Gaussian Process Regression with SpatialTemporal Memory Loss.  Each example is encapsulated in a function that can be imported via `gpr_lib.examples`.

```python
from gpr_lib.examples import gpy_comparison_1d
```

## gpy_comparison_1d

This example performs GPR on a noisy 1D sinusoidal function.

```python
from gpr_lib.examples import gpy_comparison_1d
gpy_comparison_1d(observation_points=20, noise_var=0.1, test_points=200)
```

**Output**:

![Output plot from gpr_comparison_1d example](/docs/images/gpy_comparison_1d.png)

## gpy_comparison_2d

This example performs GPR on vibration data collected by driving a mobile robot around the indoor terrain shown in the picture below.
![indoor 2D terrain](/docs/images/2d_terrain.png)

```python
from gpr_lib.examples import gpy_comparison_2d
gpy_comparison_2d(ard=True, kernel="Mat32")
```

**Output**:

The red dots in the plot correspond to sample points along the robot's path.  The coloured boxes represent the outlines of the terrain features as shown in the picture above.

![Output plot from gpr_comparison_2d example](/docs/images/gpy_comparison_2d.png)

## gpr_stml_example

This example performs GPR with SpatioTemporal Memory Loss (GPR-STML) on a 1D picewise-evolving sinusoid. Half of the samples are taken before the evolution and the other half are taken after the evolution.  A series of plots show the comparative performance of standard GPR and GPR-STML at predicting the final state of the system given the entire set of samples and no prior knowledge about the time that the evolution occurs.  For more information on GPR-STML, see [1].

```python
from gpr_lib.examples import gpr_stml_example
gpr_stml_example(observation_points=30, noise_var=0.05, test_points=200)
```

**Output**:

![Output plot from GPR-STML example](/docs/images/gpr_stml_example.png)

# Publications / References
[1]  J. Roy, “Autonomous Ground Vehicle Guidance and Mapping in Challenging Terrains by using On-Board Vibration Measurements,” Master’s thesis, Department of Electrical and Computer Engineering, Queen’s University, Kingston, Ontario, Canada, July 2021. [Online]. Available: https://qspace.library.queensu.ca/handle/1974/28980
