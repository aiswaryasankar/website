---
layout: post
title:  "No More Tea"
date:   2019-12-10 11:43:09 -0800
topic: probability
image: /stock_images/binomial_distribution.png
---

It's past midnight yet there's still work to be finished. :/ To help her get through the night, she reaches for her preferred tea brand and realizes there are no more. There are only two different types of tea boxes and each time she wants to drink tea, she chooses a tea bag from her preferred brand with probability p. Slightly annoyed, she looks to the box with the other tea brand.  What's the probability there are at least k more tea bags from the other brand to get him through the night? 

<br>
<hr>
<br>

Thinking about choosing from two different boxes, one with probability $$p$$ and the other with corresponding probability $$1-p$$ brings to mind the token problem of trying to figure out the probability of flipping $$k$$ heads out of $$n$$ coin flips where the coin lands heads up with probability $$p$$. The order in which we get heads and tails doesn't matter here, as long as we get $$x$$ heads and $$n-k$$ tails. Thus we can model this probability using the following

$$P(X = k) = \binom{n}{k} (p)^k (1-p)^{(n-k)}$$

Essentially we have the binomial distribution. Now in this case we are trying to determine the probability that she has chosen all $$n$$ tea bags from one box before choosing the last $$k$$ tea bags from the less preferred box. Thus we can compute the following probability

$$P(X = k) = \binom{2n - k}{n} (p)^n (1-p)^{(n-k)}$$

Thus we note that there is only a meager 12% probability that she has picked n from her preferred box leaving k left in the other.

Now given that there are 