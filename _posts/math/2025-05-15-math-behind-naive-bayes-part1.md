---
title: "Gaussian Naive Bayes: Part 1 — Introduction and Bayes Theorem Refresher"
description: "In this first part of the Gaussian Naive Bayes series, we explore the intuition behind Bayes' Theorem, how it applies to classification problems, and introduce the Naive Bayes assumption."
tags: [machine learning, bayes theorem, naive bayes, probability, classification]
series: "Gaussian Naive Bayes Classifier"
series_part: 1
---

{% include _series.html %}

This is the first part of a multi-part series on Gaussian Naive Bayes, a simple yet surprisingly effective probabilistic classifier. In this post, we’ll explore the foundational concepts behind Bayes’ Theorem and how they apply to machine learning. We’ll also introduce the Naive Bayes assumption, setting the stage for a deeper mathematical dive in the next post.

In the ever-growing landscape of machine learning algorithms, Naive Bayes stands out due to its elegant probabilistic foundation and surprising effectiveness in various practical applications, especially in text classification and medical diagnosis. Gaussian Naive Bayes, in particular, is a variant tailored for continuous data, assuming a Gaussian distribution for each feature conditioned on the class.

As part of revisiting foundational concepts in probability and statistics—especially Bayes’ Theorem—I built a machine learning project that implements classifiers from scratch to classify [Iris flower species](https://rcs.chemometrics.ru/Tutorials/classification/Fisher.pdf). The first model I implemented was a [Gaussian Naive Bayes classifier](https://github.com/StefanNieuwenhuis/iris_species_classifier/blob/80686c92ae6166583bd5f3f7b2c355a139e530f4/src/model/gaussian_naive_bayes.py), which served as both a mathematical refresher and a practical coding exercise.

This blog post is a deep mathematical dive into the Gaussian Naive Bayes classifier. We will go beyond the intuition and dissect the mathematical components that underpin it. **This post focuses solely on the mathematics behind Gaussian Naive Bayes—not on implementation or model building**. That will be covered in a dedicated follow-up post.

Whether you're an aspiring data scientist, machine learning engineer, or a curious student, this guide aims to offer clarity and precision in understanding the mathematical machinery that drives Gaussian Naive Bayes.

## The Iris Flower Species dataset

![Iris Flower Species With Labels](/assets/images/iris_species_with_labels.png)

The Iris flower data set or Fisher's Iris data set is a multivariate data set used and made famous by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis.

The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

<small>-- Source: [Wikipedia: Iris Flower Data Set](https://en.wikipedia.org/wiki/Iris_flower_data_set)</small>

## Bayes' Theorem Refresher

Bayes’ Theorem provides a formal method - i.e. mathematical rule for updating probabilities based on new evidence. In other words, with Bayes' Theorem we can compute how much to revise our probabilities (change our minds) when we learn a new fact or observe new evidence.

![Bayes' Theorem formula](/assets/images/Bayes_Theorem.png){: width="500" }
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

![Monty Hall Problem - Open door](/assets/images/Monty_open_door.png)
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

Switching gives you a $$\frac{2}{3}$$ chance of winning the car, while staying with your initial pick gives you only a $$\frac{1}{3}$$ chance, so we switch.

The Monty Hall Problem demonstrates how Bayesian updating works:

- Start with prior beliefs
- Incorporate new evidence
- Normalize and update beliefs accordingly

It also challenges our intuition and why probabilistic reasoning is essential in making better decisions.

## Conclusion

We've seen how Bayes' Theorem provides a probabilistic framework for classification and how the _"naive"_" assumption simplifies modeling joint distributions. In [Part 2]({% link _posts/math/2025-05-19-math-behind-naive-bayes-part2.md %}), we’ll dive deeper into the mathematical formulation of Gaussian Naive Bayes, including how log-probabilities, vectorization, and variance assumptions impact efficiency and accuracy.
