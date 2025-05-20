---
title: "Gaussian Naive Bayes: Part 2 — Mathematical Deep Dive and Optimization"
description: "We dissect the mathematical formulation of Gaussian Naive Bayes, explore key optimizations, and understand how assumptions about variance and log-probabilities lead to efficient real-world implementations."
tags: [gaussian naive bayes, probability, machine learning, linear algebra, optimization, log-space]
series: "Gaussian Naive Bayes Classifier"
series_part: 2
---

{% include _series.html %}

In the second part of the series, we dive into the mathematics behind Gaussian Naive Bayes. We'll derive the class-conditional probability density functions, explore why log-probabilities are used, and show how the entire prediction process can be transformed into a fast, vectorized dot-product computation. Along the way, we'll clarify common assumptions—such as constant variance across classes—and examine their practical consequences.

## From Bayes to Naive Bayes

Bayes' Theorem offers a formal way of updating our beliefs given new evidence. However, applying it directly in real-world classification tasks often becomes computationally intractable, especially dealing with high-dimensional data where we want to **estimate the joint probability distribution** $$P(x_1, x_2, \dots,x_n\vert y)$$. It's the [combinatorial explosion problem](https://en.wikipedia.org/wiki/Combinatorial_explosion), where the number of possible combinations of feature values grows exponentially with the number of features.

### A Quick Primer: Independent vs. Conditionally Independent Events
Let’s break down the concepts of **independence** and **conditional independence**, as they’re crucial for understanding the _"naivety"_ in Naive Bayes.

#### Independent events

Two events, $$A$$ and $$B$$, are **independent** if the occurrence of one has no effect on the probability of the other:

$$P(A\cap B)=P(A)\cdot P(B)$$

Or, equivalently:

$$P(A\vert B) = P(A)$$

, and

$$P(B\vert A) = P(B)$$

##### Example: Tossing two fair coins; The result of the first toss doesn’t affect the second.

![Quarter dollar head and tail](/assets/images/fair_coin.jpg)
<small>Image by [Great American Coin Co.](https://www.greatamericancoincompany.com/cdn/shop/articles/front-and-back_a6e2bed8-21a7-4f3f-b765-7ddf3570a7d7.jpg?v=1736183719&width=1400)</small>

When
- $$A=\{x\vert x = \text{first coin is heads}\}$$, and
- $$B=\{x\vert x = \text{second coin is heads}\}$$,

Then

$$P(A\cap B)=P(A)\cdot P(B) = \frac{1}{2} \cdot \frac{1}{2} = \frac{1}{4}$$

#### Conditional independent events

Two events $$A$$ and $$B$$ are **conditionally independent given a third event $$C$$** if, once we know $$C$$ has occurred, learning about $$A$$ gives us **no additional information about $$B$$**, and vice-versa. In other words, once we know the _"cause"_, the symptoms are treated as **independent**.

Formally:

$$P(A\cap B\vert C)=P(A\vert C)\cdot P(B\vert C)$$

Or, equivalently:

$$P(A\vert B, C)=P(A\vert C)$$

and

$$P(B\vert A, C)=P(B\vert C)$$

##### Example: Spam Email classifier

Let's say we're building a spam email classifier, where:

- $$C$$: Email is `spam` or `not spam`
- $$A$$: Email contains the word _"Viagra"_
- $$B$$: Email contains the words _"Money-back guarantee"_

In general, the presence of the words _"Viagra"_, and _"Money-back guarantee"_ might be correlated, since spammy phrases often appear together. Knowing the presence of $$A$$ might increase the likelihood of $$B$$. This means that both events are _dependent_ in isolation.

But here's the key. Once we conditioned on $$C$$ (email is `spam`), the presence of $$A$$ no longer gives us new infromation on the likelihood of $$B$$. Why? Both events are already explained by the fact the email is spam. The common cause $$C$$ makes the co-occurrence of the symptoms more likely.

In other words,

> Once we know that an email is spam, it does not matter wheter $$A$$ and $$B$$ often appear together in general. The only relevant question is how likely each word appears in spam emails?

So $$A$$ and $$B$$ become conditionally independent given $$C$$. This is the **Naive Bayes assumption** in action. It's what allows the model to break down complex joint probabilities into simpler, individual conditional probabilities (one per feature).

## The "Naive" assumption

Now that we understand how Bayes' Theorem updates our beliefs given evidence, we face a practical question:

> How do we estimate $$P(e\vert H)$$ when $$e$$ is a high-dimensional feature vector?

Given the hypothesis $$H$$, and dependent feature (or evidence) vector $$\vec{e} = \begin{bmatrix}e_1 & \dots & e_n\end{bmatrix}$$, Bayes' Theorem states:

$$P(H\vert e_1 \cap e_2 \cap \dots\cap e_n) = \frac{(e_1 \cap e_2 \cap \dots\cap e_n \vert H)\cdot P(H)}{P(e_1 \cap e_2 \cap \dots\cap e_n)}$$

In high-dimensional spaces, modelling this full joint probability distribution $$P(e_1 \cap e_2 \cap \dots\cap e_n\vert H)$$ becomes often computationally intractable, especially with limited training data.

### Why is it hard?

The number of parameters required to model a joint distribution grows **exponentially** with the number of features in a dataset. For example, suppose we wanted to model dependencies between just 100 binary features conditioned on a class label. This would require estimating probabilities for $$2^{100}$$ possible feature combinations per class - an astronomically large number (<small>$$1,2676506002×10^{30}$$</small>). Even for continuous features, estimating a full, multivariate distribution requires computing high-dimensional covariance matrices, and ensuring they are invertible and well-conditioned. These tasks are computationally expensive and statistically fragile unless we have massive datasets.

This is known as the [curse of dimensionality](https://www.datacamp.com/blog/curse-of-dimensionality-machine-learning): as the number of dimensions (features) increases, the amount of data required to reliably estimate densities increases exponentially. In practice, we rarely have enough data to estimate these complex joint distributions without overfitting or introducting heavy regularization.

### The "naive" simplification

The "naive" assumption circumvents this by treating each feature as conditionally independent given the class label:

$$P(e_i\vert H\cap e_1\cap\dots\cap e_{i-1}\cap e_{i+1}\cap\dots\cap e_n) = P(e_i\vert H)$$

, and for all $$i$$ this relation is simplified to:

$$P(H\vert e_1\cap\dots\cap e_n) = \frac{P(H)\cdot\prod\limits_{i=1}^n P(e_i | H)}{P(e_1\cap\dots\cap e_n)}$$

The denominator normalizes the nominator, and **integrates over all possible class labels**, since it does not depend on $$H$$ to be computed - i.e. it doesn't affect which class has the highest probability. So we can rewrite the classification rule using just the numerator:

$$P(H\vert e_1\cap\dots\cap e_n) \varpropto P(H)\cdot\prod\limits_{i=1}^n P(e_i | H)$$

, and we can use $$\arg \max$$ to find the class that **gives the highest posterior probability** for feature vector $$\vec{x}$$:

$$\hat{y}=\arg \max_{y} P(y)\cdot\prod\limits_{i=1}^n P(x_i | y)$$

Estimating the posterior probabilities from training data is done via **Maximum A Posteriori (MAP) estimation**. $$P(y)$$ is estimated as the relative frequency of class $$y$$ in the training set, and each $$P(x_i\vert y)$$ is estimated based on feature distributions in each class.

Under the hood, it computes $$P(y)\cdot\prod\limits_{i=1}^n P(x_i \vert y)$$ for every class $$y$$, and normalizes the results so they sum to $$1$$.

### Why does it work?

Naive Bayes often performs surprisingly well in practice, despite its unrealistic independence assumptions. This is because accurate probability estimates are not always required for good classification. What matters is that the decision boundary induced by comparing class scores remains useful. Even when the probabilities themselves are not well calibrated. As long as the incorrect independence assumption does not lead to systematically biased scores, the final predictions can still be highly effective.

This is why Naive Bayes is often described as

> "wrong, but useful" - A pragmatic trade-off between statistical realism and computational simplicity.

We have to be **cautious with correlations between features** though, since it breaks down our feature independence assumption, and the model exaggerates or underrepresents the true likelihood, and performance degrades.

The naive assumption oversimplifies reality. In most real-world datasets, features are not truly independent. However, the simplification offers practical benefits:

- **Tractability**: We only have to estimate one univariate distribution per feature per class.
- **Sample efficiency**: We need far fewer samples to get reliable estimates of the probability distributions for each class.
- **Speed**: The classifier becomes fast to train and evaluate.

## From assumptions to implementations: Handling continuous features

At the heart of Naive Bayes thus lies a simplifying assumption: All features are conditionally independent given the class label, but in order to **compute** the individual terms $$P(x_i \vert y)$$, we have to make assumptions about each features' distribution. This is where we can use different Naive Bayes variants. The assumptions remain the same, but the difference is the way the individual probabilities for each $$x_i$$ is computed.

- if $$x_i$$ is **categorical**, we use **frequency counts** (Multinominal or Bernoulli NB).
- if $$x_i$$ is **continuous**, we cannot count how often each exact value occurs, since real-valued features are rarely repeated exactly, and this problem grows when the precision (e.g. measurement) increases.

The latter leads us to a natural solution: **Assume a probability distribution over the feature values**. Enter the Gaussian Naive Bayes classifier.

### Gaussian Naive Bayes Classifier

![Gaussian or Normal distribution plot](/assets/images/normal_distribution.png)

<small>Image by [RMIT University](https://learninglab.rmit.edu.au/maths-statistics/statistics/s10-standard-normal-distribution)


The Gaussian distribution is a natural choice, since:

- The Central Limit Theorem since many features tend to be approximately normal.
- Simplicity: A normal distribution is described with only the **mean** and **variance**.
- Mathematical convenience (e.g. closed-form likelihood, stability under transformations, computationally cheap)

It is defined for a **continuous variable $$x$$** with:

- **Mean $$\mu$$**: The center of the distribution
- **Variance $$\sigma^2$$**: The spread of the data - i.e. distance from the mean

#### Probability density function (PDF)

We can use the **probability density function (PDF)** to compute the probability of $$x$$ comes from a Gaussian distribution with a given $$\mu$$, and $$\sigma^2$$:

$$P(x_i\vert y) = \frac{1}{\sqrt{2\pi\sigma_{ic}^2}}\cdot\exp(-\frac{(x_i - \mu_{ic})^2}{2\sigma_{ic}^2})$$


#### First term: the normalization constant

$$\frac{1}{\sqrt{2\pi\sigma_{ic}^2}}$$

In probability theory, a **PDF** describes how likely a continuous random variable is to take on a value in a given range, but, unlike discrete probabilities, where probabilities like $$P(x=3)$$ are directly meaningful, continuous variables don't work that way.

Instead, the probability that a continuous variable fall within a range $$[a,b]$$ is given by **the area under the curve of the PDF between $$a$$ and $$b$$**.

$$P(a \leq x \leq b) = \int_{b}^{a}f(x)dx$$

To be a valid PDF, it must satisfy one crucial condition:

> The **total area under the curve** across the entire real number line must equal 1.

That is

$$\int_{\infty}^{-\infty}f(x)dx=1$$

This renders the distribution **properly normalized** over all real numbers.

#### Second term: exponential decay

$$\exp(-\frac{(x_i - \mu_{ic})^2}{2\sigma_{ic}^2})$$

The exponential term controls **the shape of the bell curve** - i.e. the kurtosis (how _"tall"_ or _"flat"_ is the distribution around $$\mu$$?). It is called **exponential decay**, since, as $$x$$ moves away from $$\mu$$, this term **rapidly decreases toward zero**.

##### Decoding the terms

- $$(x - \mu)$$: Measures the distance of $$x$$ from $$\mu$$
- $$(x - \mu)^2$$: (squared distance): Ensures this value is always positive, and the larger the distance, the larger the value becomes.
- $$(2\sigma^2)$$: Scales the squared distance. The larger $$\sigma$$, the more _"forgiving"_ the distribution is of being far from the mean.
- $$-\frac{(x_i - \mu_{ic})^2}{2\sigma_{ic}^2}$$ (negative sign): This makes the exponent negative, so as the distance increases, the exponent **decays**.
- $$\exp$$: Converts the scaled squared distance into a value between 0 and 1 - i.e. a probability estimate.

The expontential decay models the likelihood $$x$$ appears in a Gaussian-distributed class $$y$$.

- When $$x$$ is **close to the class mean**, the exponent is near 0 - i.e. high probability
- When $$x$$ is **far from the class mean**, the exponent becomes strongly negative - i.e. probability shrinks toward 0.

This makes the model **sensitive to deviations from the class-specific mean**, which is what allows Gaussian Naive Bayes to perform classification based on how well a value fits into the distribution learned for each class.

## Mathematical Optimizations in Gaussian Naive Bayes

When implementing Gaussian Naive Bayes, several mathematical optimizations are employed to **improve both numerical stability and computational efficiency**. The underlying probability theory remains unchanged but the optimizations make the model more robust and scalable (with high-dimensional or sparse data in particular).

### Use log probabilities instead of raw probabilities

In the prediction phase, we compute the product of conditional likelihoods

$$P(y=c\vert x)\varpropto P(y=c)\prod\limits_{i=1}^n P(x_i | y=c)$$

However, multiplying many small probabilities can quickly lead to [numerical underflow](https://en.wikipedia.org/wiki/Arithmetic_underflow), where the product becomes so small that it rounds to zero in floating-point representation. It becomes a number of more precise absolute value than the computer can actually represent in memory on its central processing unit (CPU).

#### Optimization

Apply the natural [logarithm]({% link _posts/math/2025-05-14-logarithms-beginners-guide.md %}) to the entire expression. Since the logarithm is a monotonically increasing function, maximizing the log of the probabilities yields the same result as maximizing the original product:

$$\hat{y}=\arg \max_{c} \biggl( \log P(y=c)+\sum\limits_{i=1}^n \log P(x_i | y=c)\biggl)$$

This transformation avoids numerical underflow and ensures robust computation.

But log-space transformations are not just about numerical safety, they also improve **computational efficiency**, particularly in the context of large-scale or high-dimensional data.

The primary benefit comes from the mathematical identity

$$log(a\cdot b) = log(a) + log(b)$$

Multiplying many identities - i.e. one for each feature variable - can be computationally expensive and slow. By converting the product into a sum of logarithms:

- We **reduce the number of operations** from $$O(n)$$ multiplications to $$O(n)$$ additions. Although modern CPU architectures mitigate the effects, addition is still faster, and less complex than multiplication ($$O(n)$$ vs. $$O(n \log n)$$ time complexity).
- Additions are **faster and more stable** than floating-point multiplications, especially on hardware like GPUs or vectorized CPUs where addition is heavily optimized. It also limits parallelism, since each multiplication depends on the previous result.

### Use vectorization via dot-product form

Terms can be re-arranged to fit the structure of a dot-product by transforming the log-likelihood of a Gaussion PDF into an algebraic form. This allows for efficient batch computation, since modern hardware architectures are highly optimized for such calculations.

Recall the log of a Gaussian PDF

$$log P(x_i \vert y=c) = -\frac{1}{2}log(2\pi\sigma_{ic}^2)-\frac{(x_i - \mu_{ic})^2}{2\sigma_{ic}^2}$$

The total log-probability can be broken down into a **sum of constants**, **quadratic terms**, and a **linear dot-product** between input and model parameters. The class score is represented as

$$log P(y=c \vert x) = log P(y=c)+x+w_c+\text{const}_c$$

Where

- $$w_c=\biggl(\frac{\mu_{ic}}{\sigma_{ic}^2}\biggl)$$,
- The offset $$\text{const}_c=-\sum\limits_i\biggl(\frac{\mu_{ic}^2}{2\sigma_{ic}^2}+\frac{1}{2}log(2\pi\sigma_{ic}^2)\biggl)$$

This transforms inference into a **single dot-product plus offset per class**, which allows for **fast batch prediction** with matrix multiplication ($$X\cdot\mathbf{\theta}^\top+b$$), and allows the constants to be **cached**.

If speed is what you're after, we can assume equal variance $$\sigma_{ic}^2 = \sigma_{c}^2$$ or $$\sigma_{ic}^2 = \sigma^2$$ so that the **quadratic term** cancels out in the equation, and the remaining term becomes linear. This is a trade-off between speed and accuracy, and A/B testing is required to objecticvely compare solutions, and pick the best fitting model for a use-case.

### Use Epsilon Variance Smoothing

The Gaussian PDF breaks down if the estimated variance $$\sigma_{ic}^2=0$$, which can happen when:

- A feature has **zero variance** in a class (e.g. same value across all training samples of a class).
- A feature has **very low variance** making the denominator in the exponent extremely small, leading to numerical instability or [exploding gradients](https://www.machinelearningmastery.com/exploding-gradients-in-neural-networks/) in probabilistic computations.

To guard against these issues, a **small constant $$\epsilon$$ is added** to the variance during computation

$$\tilde{\sigma}_{ic}^2=\sigma_{ic}^2+\epsilon$$

This technique is called **variance smoothing**, and $$\epsilon$$ is typically on the order of $$10^{-9}$$ to $$10^{-5}$$, depending on the scale of the data. By preventing division by zero and avoiding extremely large exponentials, it guarantees well-behaved computation even for pathological features or small training sets.

## Conclusion

Gaussian Naive Bayes might seem simple at first glance, but beneath its straight-forward assumptions lies a wealth of mathematical elegance and engineering nuance. By diving deep into the underlying mathematics, we've seen how classical probability theory, numerical stability optimizations, and linear algebra come together to create an inference engine that's both fast and effective.

Transforming probability products into log-space not only prevents numerical underflow but unlocks powerful computational optimizations: it turns costly multiplications into efficient additions and enables vectorized inference via dot-product formulations. Smoothing tiny variances with epsilon avoids pathological cases where the model becomes overly confident or completely unstable. And through visualizing decision boundaries, we get a tangible sense of how these mathematical choices shape model behavior.

Ultimately, these optimizations aren't just about speed or safety — they reflect a mature understanding of how theoretical models meet the real-world imperfections of data and hardware. Whether you're building a scalable classifier or exploring probabilistic reasoning, mastering these principles ensures your models are not only correct, but robust, interpretable, and production-ready.

In the next and final part, we’ll implement these ideas in Python, visualize decision boundaries, and compare theoretical insights with empirical results.