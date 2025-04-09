---
title: "From Neural Nets to Hashmaps — Why I’m Relearning the Fundamentals"
categories:
  - dsa
tags:
  - hashmaps
  - data structures
  - fundamentals
  - machine learning
  - recommender systems
  - kubernetes
  - blog
date: 2025-04-09
excerpt: "Hashmaps power everything from feature stores to caching layers. In this post, I explain why I’m revisiting them as a machine learning engineer."
---

> *“You don’t really understand a concept until you’ve taught it.”*

I recently set out on a mission to revisit fundamental data structures—not just to refresh my knowledge, but to better articulate and document them as part of my public learning journey. This post kicks off a new series where I’ll be blogging my notes and learnings as I dive deeper into Data Structures & Algorithms (DSA).

In this first entry, we’ll explore one of the most essential and elegant structures: **hashmaps** (also known as dictionaries in Python).


## 🧠 Why Start with Hashmaps?

As a Machine Learning Engineer, I often interact with complex distributed systems and large-scale data pipelines. But I’ve learned over the years that **deep mastery of foundational data structures** gives you an edge when debugging, optimizing, or explaining complex systems.

Hashmaps are ubiquitous in machine learning codebases, from feature stores and caches to logging metadata, configs, and parameter storage.

Let’s break them down from the ground up.


## 🧩 Why Hashmaps Matter (Even in ML)
You might ask: __Why should someone working on multi-armed bandits or collaborative filtering care about hashmaps?__

Because maps — aka dictionaries or hash tables — are **everywhere** in production ML:

* Counting term frequencies or user-item interactions? → Use a hashmap.
* Storing cached embedding vectors for reuse? → Use a hashmap.
* Implementing a feature store or a key-value-based retrieval backend? → Definitely a hashmap.

And beyond ML, hashmaps are the go-to tool for engineers working on performance-critical backend systems — just like the ones powering Booking.com’s search, availability, and personalization flows.

So instead of just skimming the docs, I sat down and re-implemented the core operations, step by step. What I found was surprisingly delightful.


## 🔧 How a Hashmap Actually Works
At a high level, a hashmap is a **key-value** store that lets you do this:

```python
prices = {'hotel_1': 89, 'hotel_2': 120}
print(prices['hotel_1'])  # → 89
```

But under the hood?

1. The key (`'hotel_1'`) is passed through a **hash function**, turning it into a numeric index.
2. That index maps to a **bucket** in an array.
3. If two keys map to the same index (a **collision**), the hashmap resolves it — often using **chaining**, where multiple key-value pairs are stored in a list at that index.

This simple trick makes **average-case lookup, insert, and delete all O(1)**.

It’s blazing fast. And it’s why modern feature stores, caching layers, and even graph algorithms use them constantly.


## 🔄 When I Relearned It, I Noticed...
Something clicked. Not in a “textbook” way, but in a __real-world recommender system__ kind of way.

Take the ["Group Anagrams"](https://leetcode.com/problems/group-anagrams) problem. It feels abstract at first, but then you realize it's just a question of **key design**. You hash on a sorted string — a clever fingerprint — and group words accordingly.

That’s exactly what we do in ML when creating **hashed feature buckets** or **aggregating click events** by session ID.

> Revisiting these problems now feels like seeing old friends through a new lens: with the eyes of someone who has fought latency bugs and wrangled production data.

## 🧪 My Practice Flow
As part of this DSA refresh, I’ve been solving classic hashmap-related LeetCode problems, including:

* ✅ [Two Sum](https://leetcode.com/problems/two-sum)
* ✅ [Group Anagrams](https://leetcode.com/problems/group-anagrams)
* ✅ [Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence)
* ✅ [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k)

I’m not just solving them to get green ticks. I’m solving them to **understand what makes solutions elegant, efficient, and robust enough to scale**.


## 📈 Complexity Recap

|  Operation  |	 Average Case  |  Worst Case  |
|-------------|----------------|--------------|
|  Insert     |  O(1)          |  O(n)        |
|  Lookup     |  O(1)          |  O(n)        |
|  Delete     |  O(1)          |  O(n)        |

Yes, worst case is linear due to collisions — but with good hash functions and low load factors, you rarely hit it. Modern languages (Python, Java, Go, etc.) optimize aggressively here.

## 🛠️ Common Use Cases in ML Engineering

Here’s where I see hashmaps in practice:
* **Feature lookup** in online prediction services
* **Embedding tables** (backed by a key-value store)
* **Experiment tracking configs** (e.g., logging which variant a user saw)
* **Hyperparameter tuning frameworks**

Understanding how they behave under the hood helps me reason about performance bottlenecks—especially when dealing with large-scale or high-throughput services.


## 📈 Final Thoughts

Revisiting the hashmap wasn’t just a refresher for me—it was a reminder of how much power lies in simplicity. As I continue this DSA refresher series, I’ll keep connecting the dots between textbook knowledge and real-world machine learning engineering.


## 🚀 Why I’m Sharing This Publicly

I’m building this blog as a transparent record of my learning process — not just to prep for interviews, but to sharpen my engineering instincts.

It’s part of a larger project where I’m also:

* Building a real-time recommender system from scratch with **FastAPI + Redis + Spark**
* Deploying everything to my bare-metal **Kubernetes homelab**
* Measuring end-to-end performance with **Prometheus + Grafana**
* Open-sourcing the entire thing

So whether you're a fellow ML engineer, a systems-minded developer, or a recruiter curious about my thought process — welcome aboard.


## 📣 Let’s Learn Together

If you're also revisiting the fundamentals — or if you're deep in the weeds of ranking models and want to get more hands-on with infra — I’d love to connect.

This is just the first in a series of DSA posts. Next up: Sliding windows, prefix sums, and graph traversal for recommender systems, and much more.

* 👉 Check out the [DSA series here](https://stefannieuwenhuis.github.io/categories/#dsa)
* 👉 Follow the full [Recommender From Scratch project here](https://stefannieuwenhuis.github.io/categories/#recommender-system)
* 👉 Let’s connect on [LinkedIn](https://www.linkedin.com/in/stefannhs) and [GitHub](https://github.com/StefanNieuwenhuis)

See you in the next post.