---
layout: page
title: Poisson Derivation
permalink: /poisson
---

We want to model the number of people that arrive in lab during lunch hour. We want to find the probability that there are 2, 5 or 10 people in lab during that time period. We assume that people arrive randomly and walk in on their own. Let's also say that over the sequence of general observation I estimate that there are on average 4 people in lab during lunch hour on any given day.  
- Expected value of random variable is lambda
- expected value of a random var is n x p
- model this as a binomial distribution
- we can divide up the hour into minutes so there are n minutes with probability lambda/60 that a person arrives in that minute
- now using this, how can we compute that there are k people that arrive during the hour?
- binomial distribution 

$$P(X = k) =  \binom{60}{k} (\frac{\lambda }{60})^k (1-\frac{\lambda }{60})^{(60-k)}$$

- This isn't completely accurate though because you are counting a success if a person arrives in a given minute. However you might have 2 people arriving within the same minute and they wouldn't be distinguished or reflected in the count k. 
- Thus we can continue to make smaller and smaller time intervals.  
- If we take the limit as the time interval goes to infinity we have the following

$$P(X = k) =  \lim_{n \to \infty } \binom{n}{k} (\frac{\lambda }{n})^k (1-\frac{\lambda }{n})^{(n-k)}$$

In order to compute this limit we know this holds from calculus:

$$e = \lim_{n \to \infty } (1 + \frac{1}{n})^n$$

We can now simplify the limit:

$$P(X=k) = \lim_{n \to \infty } \left ( \frac{n!}{(n-k)! k!} \right ) \left ( \frac{\lambda}{n} \right )^k \left ( 1- \frac{\lambda}{n} \right )^n \left ( 1- \frac{\lambda}{n} \right )^{(-k)}$$

We pull out terms that don't depend on n


$$P(X=k) = \left ( \frac{\lambda^k}{k!} \right )\lim_{n \to \infty }  \left ( 1 - \frac{\lambda}{n} \right )^n \left ( 1- \frac{\lambda}{n} \right )^{(-k)}$$


We know that since k is a constant the last term will go to 0 and thus we have 

$$P(X=k) = \left ( \frac{\lambda^k}{k!} \right )\lim_{n \to \infty }  \left ( 1 - \frac{\lambda}{n} \right )^n $$

Here by the definition of e we have the formula for a Poisson random variable

$$P(X=k) = \left ( \frac{\lambda^k}{k!} \right ) e^{-\lambda}$$

Essentially all we have done here is to break up our rate into increasingly smaller time segments. Then taking the limit as these time intervals go to infinity we can now compute the probability of any number of intervals in the given time period.


