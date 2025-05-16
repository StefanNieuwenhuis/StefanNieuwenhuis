---
title: "The math behind Naive Bayes, a deep-dive"
description: "Learn what a logarithm is with this detailed, beginner-friendly guide. Understand the math, real-world uses, and why logarithms matter in fields like machine learning and data science."
tags: [math, logarithm, beginner, machine learning, statistics, data science, artificial intelligence, logarithms explained, logarithms definition, logarithms rules, logarithms properties]
published: false
---

## TL;DR

## Introduction

In the ever-growing landscape of machine learning algorithms, Naive Bayes stands out due to its elegant probabilistic foundation and surprising effectiveness in various practical applications, especially in text classification and medical diagnosis. Gaussian Naive Bayes, in particular, is a variant tailored for continuous data, assuming a Gaussian distribution for each feature conditioned on the class.

As part of revisiting foundational concepts in probability and statistics—especially Bayes’ Theorem—I built a machine learning project that implements classifiers from scratch to classify [Iris flower species](https://rcs.chemometrics.ru/Tutorials/classification/Fisher.pdf). The first model I implemented was a [Gaussian Naive Bayes classifier](https://github.com/StefanNieuwenhuis/iris_species_classifier/blob/80686c92ae6166583bd5f3f7b2c355a139e530f4/src/model/gaussian_naive_bayes.py), which served as both a mathematical refresher and a practical coding exercise.

This blog post is a deep mathematical dive into the Gaussian Naive Bayes classifier. We will go beyond the intuition and dissect the mathematical components that underpin it. **This post focuses solely on the mathematics behind Gaussian Naive Bayes—not on implementation or model building**. That will be covered in a dedicated follow-up post.

Whether you're an aspiring data scientist, machine learning engineer, or a curious student, this guide aims to offer clarity and precision in understanding the mathematical machinery that drives Gaussian Naive Bayes.

## The Iris Flower Species dataset

![Iris Flower Species With Labels](/images/iris_species_with_labels.png)

The Iris flower data set or Fisher's Iris data set is a multivariate data set used and made famous by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis.

The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

<small>-- Source: [Wikipedia: Iris Flower Data Set](https://en.wikipedia.org/wiki/Iris_flower_data_set)</small>

## Bayes' Theorem Refresher

Bayes’ Theorem provides a formal method - i.e. mathematical rule for updating probabilities based on new evidence. In other words, with Bayes' Theorem we can compute how much to revise our probabilities (change our minds) when we learn a new fact or observe new evidence.

![Bayes' Theorem formula](/images/Bayes_Theorem.png){: width="500" }
<!-- $$P(H\;\vert\;e) = \frac{P(e\;\vert\;H)\cdot P(H)}{P(e)} \tag{1}$$ -->

In classification terms:

- $$P(H\vert E)$$: **Posterior** - probability of $$H$$ (hypothesis) given $$e$$ (the evidence) is true
- $$P(H)$$: **Prior** - initial belief about $$H$$ before seeing $$e$$
- $$P(e\vert H)$$: **Likelihood** -  probability of observing $$e$$ given that $$H$$ is true
- $$P(e)$$: **Marginal probability** (or evidence) - overall probability of observing $$e$$


When, for example, a certain Iris Flower species is known to have larger petals, Bayes' theorem allows a sample to be assessed more accurately by conditioning it relative to its petal size, rather than assuming the sample is typical of the population as a whole. In other words, we reduce the total number of possibilities to make the propability estimates more likely.

### It is all about information

Bayes' Theorem is, at its core, a tool for updating beliefs in light of new information. It provides a formal definition how evidence should impact our confidence in different hypothesis. The prior, $$P(H)$$, is our belief before seeing any data. The likelihood, $$P(e\vert H)$$, captures how likely the observed data is under each hypothesis, and combining both terms results in the posterior, $$P(H\vert e)$$, our updated belief after seeing the evidence.

Bayes' Theorem is thus about **information flow**: How new data reshapes our expectations and refines our understanding of the world in a formal, structured way.


### Example: Monty Hall Problem

![Monty Hall Problem - Open door](/images/Monty_open_door.png)
<small>Image by [Cepheus](https://commons.wikimedia.org/w/index.php?curid=1234194
) - Own work, Public Domain</small>


The Monty Hall problem is a brain teaser, in the form of a probability puzzle, based nominally on the American television game show Let's Make a Deal and named after its original host, Monty Hall. The problem was originally posed (and solved) in a letter by Steve Selvin to the American Statistician in 1975 It became famous as a question from reader Craig F. Whitaker's letter quoted in Marilyn vos Savant's _"Ask Marilyn"_ column in Parade magazine in 1990:

> Suppose you're on a game show, and you're given the choice of three doors: Behind one door is a car; behind the others, goats. You pick a door, say No. 1, and the host, who knows what's behind the doors, opens another door, say No. 3, which has a goat. He then says to you, "Do you want to pick door No. 2?" Is it to your advantage to switch your choice?

Savant's response was that the contestant should switch to the other door. By the standard assumptions, the switching strategy has a ⁠ $$\frac{2}{3}$$ probability of winning the car, while the strategy of keeping the initial choice has only a ⁠$$\frac{1}{3}$$ probability.

<small>-- Source: [Wikipedia: Monty Hall Problem](https://en.wikipedia.org/wiki/Monty_Hall_problem)</small>

#### To switch or not to switch? Bayes to the rescue!

Bayes’ Theorem helps clarify why **switching increases your odds**.

##### Step 1: Define the hypotheses

Since each door is equally likely to contain the car **before Monty opens any**, [LaPlace definition of probabilty](https://en.wikipedia.org/wiki/Classical_definition_of_probability#:~:text=The%20probability%20of%20an%20event%20is%20the%20ratio%20of%20the%20number%20of%20cases%20favorable%20to%20it%2C%20to%20the%20number%20of%20all%20cases%20possible%20when%20nothing%20leads%20us%20to%20expect%20that%20any%20one%20of%20these%20cases%20should%20occur%20more%20than%20any%20other%2C%20which%20renders%20them%2C%20for%20us%2C%20equally%20possible.) applies.

$$H_1=H_2=H_3=\frac{n_{favorable}}{n_{total}}=\frac{1}{3}$$

We can also define our priors:

- $$H_1$$: The car is behind door No. 1 (our original choice)
- $$H_2$$: The car is behind door No. 2 (we'd win if we switch)
- $$H_3$$: The car is behind door No. 3 (ruled out since Monty just revealed a goat behind this door)


##### Step 2: Use the evidence

Let's denote the evidence, $$e$$, as: "Monty opens door No. 3, revealing a goat".

Next, we want to compute the **posterior probabilities** $$P(H_1\vert e)$$, and $$P(H_2\vert e)$$ - i.e. what is the probability the car is behind door No. 1 or No. 2 given that Monty revealed a goat behind door No. 3?

Applying Bayes' Theorem:

$$P(H_k\;\vert\;e) = \frac{P(e\;\vert\;H_k)\cdot P(H_k)}{P(e)}$$, where $$k \in \{1,2,3\}$$.

And compute the likelihoods:

- If the car is behind door No. 1, $$H_1$$, Monty could randomly choose door No. 2 or 3. $$\Rightarrow P(e\vert H_1)=\frac{1}{2}$$.
- If the car is behind door No. 2, $$H_2$$, Monty could only pick door No. 3, since we picked the first already $$\Rightarrow P(e\vert H_2)=1$$.
- If the car is behind door No. 3, $$H_3$$, Monty could not open the door, since it reveals the car $$\Rightarrow P(e\vert H_3)=0$$

With the likelihoods computed, we now move forward to calculate the marginal probability, $$P(E)$$, or normalization constant.

$$P(E) = P(e\vert H_1)P(H_1)+P(e\vert H_2)P(H_2)+P(e\vert H_3)P(H_3)$$

$$\Rightarrow (\frac{1}{2}\cdot\frac{1}{3})+(1\cdot\frac{1}{3})+(0\cdot\frac{1}{3})$$

$$=\frac{1}{6}+\frac{1}{3}=\frac{3}{6}=\frac{1}{2}$$

Finally, we compute posteriors for hypothesis 1

$$P(H_1\vert e) =\frac{P(e\vert H_1)P(H_1)}{P(e)}$$

$$\Rightarrow \frac{\frac{1}{2}\cdot\frac{1}{3}}{\frac{1}{2}}=\frac{\frac{1}{6}}{\frac{1}{2}}=\frac{1}{3}$$

And hypothesis 2

$$P(H_2\vert e) =\frac{P(e\vert H_2)P(H_2)}{P(e)}$$

$$\Rightarrow \frac{1\cdot\frac{1}{3}}{\frac{1}{2}}=\frac{\frac{1}{3}}{\frac{1}{2}}=\frac{2}{3}$$

##### Conclusion

Switching gives you a $$\frac{2}{3}$$ chance of winning the car, while staying with your initial pick gives you only a $$\frac{1}{3}$$ chance, so we switch.

The Monty Hall Problem demonstrates how Bayesian updating works:

- Start with prior beliefs
- Incorporate new evidence
- Normalize and update beliefs accordingly

It also challenges our intuition and why probabilistic reasoning is essential in making better decisions.

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

![Quarter dollar head and tail](/images/fair_coin.jpg)
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

### Conclusion

The naive assumption clearly oversimplifies reality. In most real-world datasets, features are not truly independent. However, the simplification offers practical benefits:

- **Tractability**: We only have to estimate one univariate distribution per feature per class.
- **Sample efficiency**: We need far fewer samples to get reliable estimates of the probability distributions for each class.
- **Speed**: The classifier becomes fast to train and evaluate.

## From assumptions to implementations: Handling continuous features

At the heart of Naive Bayes lies a simplifying assumption: All features are conditionally independent given the class label, but in order to **compute** the individual terms $$P(x_i \vert y)$$, we have to make assumptions about each features' distribution. This is where we can use different Naive Bayes variants. The assumptions remain the same, but the difference is the way the individual probabilities for each $$x_i$$ is computed.

- if $$x_i$$ is **categorical**, we use **frequency counts** (Multinominal or Bernoulli NB).
- if $$x_i$$ is **continuous**, we cannot count how often each exact value occurs, since real-valued features are rarely repeated exactly, and this problem grows when the precision (e.g. measurement) increases.

The latter leads us to a natural solution: **Assume a probability distribution over the feature values**.

### Gaussian Naive Bayes Classifier

The Gaussian or normal distribution is a natural choice, since:

- The Central Limit Theorem since many features tend to be approximately normal.
- Simplicity: A normal distribution is described with only the **mean** and **variance**.
- Mathematical convenience (e.g. closed-form likelihood, stability under transformations, computationally cheap)

For each class $$y$$, we model each continuous feature $$x_i$$ with a Gaussian distribution:

$$P(x_i\vert y=c)=\frac{1}{\sqrt{2\pi\sigma_{ic}^2}}\exp(-\frac{(x_i - \mu_{ic})^2}{2\sigma_{ic}^2}$$

, where

- $$P(y = c) = \frac{\text{n class c samples}}{\text{total samples}}$$.
- $$\mu_{ic}=\frac{1}{N_c}\sum\limits_{j=1}^{N_c}X_{i}^{(j)}$$, where
    - $$\mu_{ic}$$ is the average value of feature $$x_i$$ across all samples in class $$c$$.
    - $$N_c$$: the number of training samples belonging to class $$c$$
    - $$X_{i}^{(j)}$$: the value of the $$i^{th}$$ feature in the $$j^{th}$$ training sample of class $$c$$
- $$\sigma_{ic}^2=\frac{1}{N_c}\sum\limits_{j=1}^{N_c}(X_{i}^{(j)} - \mu_{ic})^2$$, where
    - $$\sigma_{ic}^2$$ is the variance of feature $$x_i$$ for class $$c$$, and computed with
    - $$N_c$$: the number of training samples belonging to class $$c$$
    - $$X_{i}^{(j)}$$: the value of the $$i^{th}$$ feature in the $$j^{th}$$ training sample of class $$c$$

The formula describes the familiar **bell curve**, centered at $$\mu_{i,y}$$

![Gaussian or Normal distribution plot](/images/normal_distribution.png)

<small>Image by [RMIT University](https://learninglab.rmit.edu.au/maths-statistics/statistics/s10-standard-normal-distribution)


The Gaussian assumption turns Naive Bayes into a [parametric model](https://en.wikipedia.org/wiki/Parametric_model), a family of probability distributions with a finite number of parameters, where we estimate the **mean** and **variance** of each feature within each class using training data.
