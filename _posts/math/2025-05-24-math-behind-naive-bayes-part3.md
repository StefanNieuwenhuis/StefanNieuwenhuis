---
title: "Gaussian Naive Bayes: Part 3 — From Theory to Practice with Python"
description: "Implement Gaussian Naive Bayes from scratch using NumPy. Learn how to fit the model, compute log-likelihoods, apply numerical stability tricks like log-sum-exp, and build a vectorized classifier step-by-step."
tags: [gaussian naive bayes, machine learning, python, numpy, implementation, logsumexp, numerical stability, tutorial]
series: "Gaussian Naive Bayes Classifier"
series_part: 3
---


{% include _series.html %}

In the previous part of the series, we took a deep dive into the mathematics behind Gaussian Naive Bayes, and what optimizations we could apply to prevent computation errors, and improve computational efficiency. In this part of the series, we will implement our insights, and build a Gaussian Naive Bayes classifier from scratch in Python, using Test-Driven Development (TDD) to ensure that each mathematical concept is correctly translated into the working code. We will train the model on the Iris dataset, and use it to classify.

## A quick mathematical recap

Recall Bayes' Theorem:

$$P(H\vert e) = \frac{P(e\vert H) P(H)}{P(e)}$$

Given the class variable $$y$$, and dependent feature vector $$\vec{x}=\begin{bmatrix} x_i & \dots & x_n \end{bmatrix}$$, the theorem defines the following relationship:

$$P(y\vert \vec{x}) = \frac{P(y) P(\vec{x}\vert y)}{P(\vec{x})}$$

Assuming [conditional independence]({ %link _posts/2025-05-24-math-behind-naive-bayes-part3-implementation#a-quick-primer-independent-vs-conditionally-independent-events %}) for all $$x_i$$, this is simplified to

$$P(x_i\in\vec{x}\vert y, x_1, \dots, x_{i - 1}, x_{i + 1},\dots, x_n) = P(x_i\in \vec{x} \vert y)$$

Which simplifies the relationship to

$$P(y\vert \vec{x}) = \frac{P(y)\prod\limits_{i=1}^n P(x_i\vert y)}{P(\vec{x})}$$

Since $$P(\vec{x})$$ is constant given the input, we can derive the following classification rule

$$P(y\vert \vec{x}) \varpropto P(y) \prod\limits_{i=1}^n P(\vec{x} \vert y)$$

Applying log optimization results in the following classification rule

$$P(y\vert \vec{x}) \varpropto \log\biggl(P(y)\biggl) + \sum\limits_{i=1}^n \log\biggl(P(x_i\vert y)\biggl)$$

Where, for Gaussian Naive Bayes,

$$\log\biggl(P(x_i\vert y)\biggl) =  -\frac{1}{2}\log(2\pi\sigma_{ic}^2)-\frac{(x_{ic} - \mu_{ic})^2}{2\sigma_{ic}^2}$$

### Vectorization

We extract constants that can be precomputed during training by expanding the factor $$(x_{ic} - \mu_{ic})^2$$ , and enable extremely fast log-likelihood computation during inference time.

$$\log\biggl(P(x_i\vert y)\biggl)=\underbrace{-\frac{1}{2}\log(2\pi\sigma_{ic}^2)- \frac{\mu_{ic}^2}{2\sigma_{ic}^2}}_\text{log normalization constants}+\underbrace{\frac{\mu_{ic}}{\sigma_{ic}^2}}_\text{mean pull}x_{ic}\underbrace{-\frac{1}{2\sigma_{ic}^2}}_\text{curvature}x_{ic}^2$$

We can precompute **log normalization constants**, **mean pull**, and **curvature**, along with the **prior probabilities** for each class $$y$$, and each feature $$i$$, and compute log-likelihoods fast at inference time.

### Epsilon Variance smoothing

To prevent division by zero or numerical instability, we apply $$\epsilon$$ variance smoothing:

$$\sigma_{ic}^2\longrightarrow\sigma_{ic}^2 + \epsilon$$, where $$\epsilon=1e-9$$

The final classification rule is:

$$\boxed{P(y\vert \vec{x}) \varpropto \boldsymbol{\log\biggl(P(y)\biggl)} + \sum\limits_{i=1}^n -\frac{1}{2}\log(2\pi(\sigma_{ic}^2+\epsilon))- \frac{\mu_{ic}^2}{2(\sigma_{ic}^2+\epsilon)}+\frac{\mu_{ic}}{\sigma_{ic}^2+\epsilon}x_{ic}-\frac{1}{2(\sigma_{ic}^2 +\epsilon)}x_{ic}^2}$$

## Step-by-step implementation

Now that we have refreshed our mathematics, we can start implementing. Let's walk through each step.

### Imports

```python
# Libraries
import numpy as np

# Typings
from typing import Self, Any
from numpy.typing import NDArray

```

### Define the class

We need the following methods for training, classification, and probability estimations:

| Method | Description |
|--------|-------------|
| `.fit()` | Fits our Gaussian Naive Bayes model according to the given training data |
| `.predict()` | Perform classification on an array of test vectors $$X$$ |
| `.predict_proba()` | Estimate probability outputs on an array of test vectors |
| `.predict_log_proba()` | Estimate log probability outputs on an array of test vectors |

Let's create a class with a descriptive name, like `GaussianNaiveBayes` (but you are free to pick any name you like), and add the methods.

```python
class GaussianNaiveBayes:
    """
    A Gaussian Naive Bayes Classifier

    Implements Bayes' Theorem assuming feature
    independence and Gaussian likelihoods.
    """

    def __init__(self):
        pass


    def fit(self)-> Self:
    """
    Fits the Gaussian Naive Bayes model
    according to the given training data
    """
    pass


    def predict(self) -> Self:
    """
    Perform classification on an array of test vectors X
    """
    pass


    def predict_log_proba(self) -> NDArray[np.int64]:
    """
    Estimate log probability outputs on an array of test vectors
    """
    pass

    def predict_proba(self) -> NDArray[np.int64]:
    """
    Estimate probability outputs on an array of test vectors
    """
    pass

```

### Fitting the model

Next, we implement `.fit()` to train our model. The fit phase has two pre-computation steps:

1. Compute log prior probabilities $$P(y)$$
1. Compute log-likelihood constants `curvature`, `mean pull` & `log likelihood constants`

> Notice that we apply **log-optimization** ([part 2]({ %link _posts/2025-05-19-math-behind-naive-bayes-part2 %})).

The `.fit()` method accepts three parameters:

1. Feature matrix $$X$$
1. Target vector $$\vec{y}$$
1. Epsilon variance smoothing value (default:`1e-9`)

We added class attributes to store pre-computations; it makes them available in inference time.

```diff
def __init__(self):
+   self.classes_: NDArray[Any] = None
+   self.class_count_: NDArray[np.int64] = None
+   self.class_prior_: NDArray[np.float64] = None
+   self.class_curvature_: NDArray[np.float64] = None
+   self.class_mean_pull_: NDArray[np.float64] = None
+   self.class_log_likelihood_consts_: NDArray[np.float64] = None
-    pass

```

To keep the code legible, maintainable, and interoperable, we added two helper methods:

1. `.__compute_prior_probabilities()`
1.  `.__compute_log_likelihood_constants()`

```python
def __compute_log_prior_probabilities(self, y: NDArray) -> Self:
    """
    Computes log prior probabilities for each class

    Parameters
    ----------
    y: nd.array of shape (n_samples,)
        Target vector

    Returns
    -------
    self: class-instance
    """

    # Store unique classes and n_samples_per_class in class attributes
    self.classes_, self.class_count_ = np.unique(y, return_counts=True)

    # Compute log prior probabilities
    self.class_prior_ = np.log(self.class_count_ / self.class_count_.sum())

    return self

```
<br/>
```python
def __compute_log_likelihood_constants(self, X: NDArray, y: NDArray, epsilon: np.int64) -> Self:
    """
    Compute log-likelihood constants for feature matrix X

    Parameters
    ----------
    X: nd.array of shape (n_sample, n_features)
        Feature matrix
    y: nd.array of shape (n_samples,)
        Target vector
    epsilon: np.int64
        variance smoothing value to prevent division by zero, and provide numerical stability

    Returns
    -------
    self: class-instance
    """

    # First, we have to group the feature matrix by class
    # We use a boolean mask, a matrix (shape: n_classes x n_features)
    # mask[i,j] = True if sample j belongs to class i
    mask = (y == self.classes_[:, None])

    # Reshape class_count_ class attribute from shape: (n_classes, ) to shape: (n_classes, 1)
    # Required for matrix multiplication below
    counts = self.class_count_[:, None]

    # Compute per-class sums of features (shape: n_classes x n_features)
    sums = mask @ X

    # Compute per-class means
    thetas = sums / counts

    # Compute sums of squares for variance computation
    X_squared = X ** 2

    # Compute per-class sums of squares
    sums_of_squares = mask @ X_squared

    # Compute variances = E[x^2] - (E[x])^2 + ϵ
    variances = (sums_of_squares / counts) - (thetas ** 2) + epsilon

    # With thetas and variances computed, we can pre-compute the log-likelihood constants
    self.class_curvature_ = -0.5 / variances
    self.class_mean_pull_ = thetas / variances
    self.class_log_likelihood_consts_ = -0.5 * np.log(2 * np.pi * variances) - (thetas ** 2) / (2 * variances)

    return self

```

With the helpers implemented, we can call them in `.fit()`.

```python
def fit(self, X: NDArray, y: NDArray, epsilon=1e-9)-> Self:
    """
    Fits the Gaussian Naive Bayes model
    according to the given training data

    Parameters
    ----------
    X: nd.array of shape (n_sample, n_features)
        Feature matrix
    y : nd.array of shape (n_samples,)
        Target vector
    epsilon: np.int64
        variance smoothing value to prevent division by zero, and provide numerical stability

    Returns
    -------
    self: class-instance
    """

    # Always check if the provided data meets the most basic needs.
    # In our case: Both X and y cannot be empty
    if len(X) == 0:
        raise ValueError("Feature Matrix cannot be empty")

    if len(y) == 0:
        raise ValueError("Target Vector cannot be empty")

    # Compute prior probabilities
    self.__compute_log_prior_probabilities(y)

    # Compute log-likelihood constants for feature matrix X, and target vector y
    self.__compute_log_likelihood_constants(X, y)

    return self

```

### Classification

Classification is done on an array of test vectors $$X$$ that we provide as parameters to `.predict()`. This method has two steps:

1. Compute log-likelihood probabilities
1. Perform classification by selecting the class with the highest posterior probability score with `argmax`.

To keep the code legible, maintainable, and interoperable, we added another helper method:

1. `.__compute_joint_log_likelihood()`


```python
def __compute_joint_log_likelihood(self, X: NDArray) -> NDArray:
    """
    Compute joint log likelihood

    Parameters
    ----------
    X: nd.array of shape (n_sample, n_features)
        Feature matrix

    Returns
    -------
    C: nd.array of shape (n_classes, )
        Classification results vector
    """

    # Compute log-likelihood terms
    log_likelihood = (
            (X ** 2) @ self.class_curvature_.T +
            X @ self.class_mean_pull_.T +
            np.sum(self.class_log_likelihood_consts_, axis=1)
    )

    # Add prior (log) probabilities to each row
    joint_log_likelihood = log_likelihood + self.class_prior_

    return joint_log_likelihood

```
<br/>
```python
def predict(self, X: NDArray) -> NDArray:
    """
    Perform classification on an array of test vectors X

    Parameters
    ----------
    X: nd.array of shape (n_sample, n_features)
        Feature matrix

    Returns
    -------
    C: nd.array of shape (n_classes, )
        Classification results vector
    """
    joint_log_likelihood = self.__compute_joint_log_likelihood(X)

    return np.argmax(joint_log_likelihood, axis=1)

```

### Compute posterior (log) probabilities

The final step is to estimate (log) probability outputs on an array of test vectors. We have two methods `.predict_log_proba()`, and `.predict_proba()`, where the latter normalizes the log posterior probabilities computed in the first.

`.predict_log_proba()` has two steps:

1. Compute log-likelihood probabilities
1. Normalize with `log-sum-exp` transform


```python
def predict_log_proba(self, X: NDArray) -> NDArray:
    """
    Estimate log probability outputs on an array of test vectors

    Parameters
    ----------
    X: nd.array of shape (n_sample, n_features)
        Feature matrix

    Returns
    -------
    C: nd.array of shape (n_samples, n_classes)
        Returns the log probability of the samples for each class in
        the model
    """

    # Compute joint log likelihood
    joint_log_likelihood = self.__compute_joint_log_likelihood(X)

    # Apply log-sum-exp transform
    max_joint_log_likelihood = np.max(joint_log_likelihood, axis=1, keepdims=True)
    logsumexp = max_joint_log_likelihood + np.log(np.sum(np.exp(joint_log_likelihood - max_joint_log_likelihood), axis=1, keepdims=True))

    return joint_log_likelihood - logsumexp

```
<br/>
```python
def predict_proba(self, X: NDArray) -> NDArray:
    """
    Return probability estimates for the test vector X.

    Parameters
    ----------
    X: nd.array of shape (n_sample, n_features)
        Feature matrix

    Returns
    -------
    C : nd.array of shape (n_samples, n_classes)
        Returns the probability of the samples for each class in
        the model
    """

    return np.exp(self.predict_log_proba(X))

```

## Conclusion

In this tutorial, we translated theory into practice by implementing a Gaussian Naive Bayes classifier from scratch using nothing but NumPy. We explored every component — from fitting the model with class-wise means and variances to efficiently computing log-likelihoods for predictions.

Along the way, we took care to address numerical stability with techniques like the log-sum-exp trick, and built a vectorized implementation that scales well and is easy to read.
