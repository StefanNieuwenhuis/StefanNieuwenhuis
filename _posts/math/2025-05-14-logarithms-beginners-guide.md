---
title: "What is a Logarithm? A Complete Beginner's Guide"
description: "Learn what a logarithm is with this detailed, beginner-friendly guide. Understand the math, real-world uses, and why logarithms matter in fields like machine learning and data science."
tags: [math, logarithm, beginner, machine learning, statistics, data science, artificial intelligence, logarithms explained, logarithms definition, logarithms rules, logarithms properties]
---

If you’ve ever asked yourself, _"What is a logarithm, and why should I care?"_ —you’re in the right place. Whether you're a student, aspiring data scientist, or just curious about the math behind modern technology, this blog post will give you a solid understanding of logarithms.

I’ve recently been revisiting my knowledge of statistics and probability, including foundational concepts like Bayes' Theorem. In the process, I built a machine learning project where I implement different classifiers from scratch to classify [Iris Flower Species](https://rcs.chemometrics.ru/Tutorials/classification/Fisher.pdf). The first one was a Gaussian Naive Bayes classifier. While implementing it, I ran into several mathematical optimizations that piqued my curiosity, especially the use of log transforms for numerical stability and computational efficiency. That journey is what led me to write this post.

This isn’t just a basic introduction—we’ll break everything down from first principles, explaining not just what logarithms are, but why they work, how they’re used, and how they tie into real-world applications like machine learning and data science.

Let’s start at the very beginning.

## What is a logarithm?

A **logarithm** is a mathematical tool to answer this question:

> "To what power must I raise a number to get another number?"

In other words, how many times do I have to multiply a number by itself to get another number?

Like subtraction is the opposite of summation, a logarithm is the **inverse** of exponentiation. If exponentiation is

$$10^2 = 100$$

Then the logarithmic version is

$$\log_{10}(100) = 2$$

**Intuition**: "I have to multiply 10 twice to get 100."

## Breaking it down

![Exponents versus Logarithms](/images/Exponent_vs_Logarithm.png){: width="500" }

{% imagesize /images/Exponent_vs_Logarithm.png:opengraph?width=500&height=100 alt='Exponents versus Logarithms' %}

Here, $$\log$$ stands for logarithm. The right side part of the arrow is read to be _"Logarithm of $$x$$ to the base $$b$$ is equal to $$a$$"_.

- **Base(b)**: The number we are multiplying by itself.
- **Exponent(a)**: How many times we multiply the base.
- **Argument(x)**: The number that we are taking the logarithm of.

### Restrictions

- $$\log_{b}(x) = a \;\text{if and only if}\; b^a = x$$. This is the basic definition of a logarithm.
- The base $$b$$ of a logarithm is always a positive real number ($$b \in \mathbb{R} \;\vert\; x > 0$$)
- The argument $$x$$ is always a positive real number ($$x \in \mathbb{R} \;\vert\; x > 0$$).


## Why do we use logarithms?

Logarithms show up all over math, science, engineering, and computing. Here’s why:

1. They scale down large numbers
1. They convert multiplication into addition
1. They reverse exponentiation

### They scale down large numbers

Many real-world phenomena grow or shrink exponentially (e.g. population growth, viral infections, radioactive decay, financial interest, ...). These changes happen so fast that it's hard to describe them with linear math.

Why? Linear math assumes steady, additive growth. It's like climbing stairs, flight over flight. Exponential growth is like climbing the same stairs but skipping increasingly more and more flights with every step. Each step is **multiplicative** in stead of additive.

Continuous multiplicative growth quickly outpaces our ability to model or interpret it with simple arithmetic, and that's where logarithms come in, since they convert exponential relationships into linear relationships (but do not change the underlying data).

In other words, logarithms **linearize exponential growth**: They _"compress"_ fast-growing data into a scale we can handle more easily.

![Comparing Exponential, Linear, and Logarithmic Curves](/images/Comparing_Exponential_Linear_Logarithmic_Curves.png){: width="1500" }

For an interactive version of this image, see [GeoGebra](https://www.geogebra.org/graphing/ppgejxu5)

What do we see graphed?

- Linear functions grow at a steady, continuous pace
- All logarithmic functions pass through the point $$(1,0)$$
- Logarithmic functions grow slowly
- Exponentials ($$e^x$$) grow rapidly
- Logarithms are the inverse of exponentials

Compared to exponential growth like $$e^x$$, logarithmic functions such as $$log(x)$$ grow very slowly. This _"compression"_ of scale makes it less-complex to analyze, as the curve flattens out and almost behaves linearly across a wide range of input values. This allows us to apply simpler, linear arithmetic to model and work with phenomena that are otherwise exponential.


### They turn multiplication into addition

The **Product Rule** is one of the most powerful features of logarithms:

$$\log(a\cdot b) = \log(a) + \log(b)$$

It tells us that the logarithm of a product is equal to the sum of the logarithms of the individual factors. To put it simply: **Multiplying values inside a logarithm is the same as adding their logarithms**. This might seem a little abstract at first, so let's use a simple analogy and then dive deeper into the math.

Why does the Product Rule work?

Logarithms are the inverse of exponentiation, and exponentiation has the same property: When you multiply numbers in the base of an exponential, you add their exponents, given that the base of all numbers is the same.

$$b^x\cdot b^y = b^{x+y}$$

This is just a basic property of exponents: when you multiply powers of the same base, you add the exponents. Don't worry, I have added examples below. If we take the logarithm of both sides of the equation above, we get:

$$log_{b}(b^x\cdot b^y) = log(b^{x+y})$$

Using the **logarithm rule for powers**, we can simplify the right-hand side:

$$log_{b}(b^x\cdot b^y) = x+y$$

Now, from the **product rule**, we can see that:

$$log_{b}(b^x\cdot b^y) = log_{b}(b^x) + log_{b}(b^y)$$

Which results in:

$$log_{b}(b^x\cdot b^y) = x+y = log_{b}(b^x) + log_{b}(b^y)$$

And this is exactly the product rule.

#### Example

Let's make this more concrete with numbers. We have two numbers, 100 and 10, and we want to compute the logarithm of their product:

$$log_{10}(100\cdot10)=log_{10}(1000)$$

Now. using the **product rule**

$$log_{10}(100) + log_{10}(10)$$

We compute that

$$log_{10}(100) = 2$$

$$log_{10}(10) = 1$$

Hence,

$$log_{10}(100) + log_{10}(10)=3$$


And, indeed:

$$log_{10}(1000) = 3 \Leftrightarrow 10^3=1000$$

#### Why is this useful?

1. **Simplifying Calculations**: The Product Rule makes complex logarithmic expressions easier to simplify and calculate, especially when dealing with large numbers or unknowns.
1. **Breaking Down Multiplications**: If you need to multiply two values but want to avoid directly multiplying them (e.g. to avoid complexity), you can first take their logarithms, add them, and then take the antilog (inverse logarithm) to find the product. This was especially useful before calculators, when logarithmic tables were common.
1. **Dealing with Exponentials in Data**: In real-world applications like data science, physics, or machine learning, many processes involve multiplying values that could be better handled with logarithms. By breaking multiplication into addition, it often becomes computationally simpler and more stable.

### They Reverse Exponentiation

Logarithms **undo** exponentials and this help solve equations where the unknown is an exponent. For example, if you know

$$2^x=16$$

You can solve for $$x$$ with

$$log_{2}(16) = x = 4$$

So, logarithms help you solve for exponents — the exact inverse operation of exponentiation. This is why:

> **A logarithm is the inverse of an exponential function.**

#### Why it matters

In many equations, especially in science and engineering, you know the argument and the base, but you need to find the exponent. That’s when you turn to logarithms.

## The Three Common Types of Logarithms

1. **Common Logarithm (Base 10)**
    - Written as $$\log(x)$$
    - Used in scientific notation and calculators
2. **Natural Logarithm (Base e)**
    - Written as $$ln(x)$$ where $$e \approx 2.718$$
    - Found in continuous growth/decay processes and advanced mathematics
3. **Binary Logarithm (Base 2)**
    - Written as $$\log_{2}(x)$$
    - Used in computer science (e.g. data structures and algorithms)

## Real-World Applications of Logarithms

Logarithms aren’t just abstract math — they're part of everyday technologies:

- **Machine Learning & AI**: Algorithms like Naive Bayes use logarithms to make probability computations more stable and efficient.
- **Search Engines**: [TF-IDF scores](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) for search relevance use logarithmic scaling.
- **Audio & Sound**: Decibels (dB) are a logarithmic measure of sound intensity.
- **Earthquakes**: The Richter scale is logarithmic.
- **Finance**: Compound interest is modeled exponentially; logs help reverse it.
- **Computer Science**: Binary logarithms appear in time complexity, especially in search and sort algorithms (e.g., binary search: O(log n)).

## The 7 laws of logarithms

### 1. Product rule

$$\log(a\cdot b) = \log(a) + \log(b)$$

The logarithm of a product is the sum of the logarithms.

**Example**

$$\log_{10}(100\cdot1000)=\log_{10}$$


### 2. Quotient rule

$$\log_{b}(\frac{x}{y}) = \log_{b}(x) - \log_{b}(y)$$

The logarithm of a quotient is the difference of the logarithms.

**Example**

$$\log_{2}(\frac{4}{32})=\log_{2}(32) - \log_{2}(4) = 5 - 2 = 3$$


### 3. Power rule

$$\log_{b}(x^p)=p\log_{b}(x)$$

The logarithm of a power is the exponent times the logarithm of $$x$$.

**Example**

$$\log_{10}(100^3) = 3\cdot\log_{10}(100) = 3\cdot2=6$$

### 4. Change-of-Base rule

$$\log_{b}(x) = \frac{\log_{k}(x)}{\log_{k}(b)}$$

The Change-of-Base rule allows changing the base of a logarithm to a different base.

**Example**

$$\log_{2}(8)=\frac{\log_{10}(8)}{\log_{10}(2)}\approx\frac{0.903}{0.301}\approx3$$

### 5. Zero Exponent rule

$$\log_{b}(1)=0$$

Regardless of its base, the logarithm of 1 is always 0.

### 6. Log of 1 rule

$$\log_{b}(b)=1$$

The logarithm of the base to itself is always 1.

### 7. Log of base rule

$$b^{\log_{b}(x)}=x$$

The base raised to the log of a number returns the number

**Example**

$$2^{\log_{2}(10)}=10$$


## History of Logarithms

Logarithms were invented by [John Napier](https://en.wikipedia.org/wiki/John_Napier) in the early 1600s to simplify calculations. Before calculators, scientists used logarithm tables to perform long multiplications and divisions. The invention of the slide rule, which used logarithmic scales, revolutionized engineering and astronomy.

## Conclusion

Logarithms are one of the most elegant tools in mathematics. They allow us to simplify multiplication, deal with exponential growth, and make real-world problems manageable—from data science to earthquake measurement. Even if you start with zero math background, understanding logarithms gives you access to a whole new world of problem-solving techniques.

If you're learning machine learning, algorithms, or even just trying to understand scientific scales, logs will be your friend. And now, you know exactly what they are and why they matter.
