---
layout: post
title: Pick up hobbies?
date:   2019-1-8 11:43:09 -0800
categories: jekyll update
topic: probability
image: /stock_images/compound_poisson.png
---

Who needs hobbies when you can work all day? :) Unfortunately, perhaps, we aren't yet machines. Let's say you pick up a new hobby whenever you meet someone interested in it. If interesting, new people appear in your life randomly and independently and you spend a random amount of time on each hobby, by the end of a year how long would you have spent on hobbies? Also as fun as hobbies sound, you're slightly concerned you might get carried away - what's the probability you spend over 1000 hours in a year?

<br>
<hr>
<br>

The first thing I see in trying to figure this out is that there seems to be 2 random factors. The people arrive randomly and independently which leads me to belive we can model their arrivals as a Poisson process, but then the amount of time spent doing the hobby together is also random. Given both of these factors, we want to find the total expected amount of time spent.  <i><b>How can we compute the expected value of a random variable that depends on the number of events from a Poisson process?</b></i>

<br>
![Compound Poisson Process](/stock_images/compound_poisson.png){:style="float: center;margin-left: 150px; margin-top: 7px :height="550px" width="450px" :class="img-responsive"}

<br><br>

Well whenever the value of one variable depends on the value of another we can use conditional expectation. This would be the expected amount of money spent given that we know $$n$$ people have arrived. So for example given that you meet 6 new people this year, what's the expected amount of time you spend on hobbies?  Now evidently we don't actually know the number of new people you meet since its a Poisson process.  How do we account for that? We can condition the amount of time spent for every possible number of arrivals $$n$$.  This is simply summing up over all the conditional expectations. Let's let $$Y(t)$$ represent the total number of hours spent on hobbies as a function of time and N(t) represent the Poisson process for the arrival of new people.

$$E[Y(t)] = \sum_{n=0}^{\infty} E[Y(t) \mid N(t) = n] P(N(t) = n)$$

Great. Now we only have to break this down. We know the number of people that have arrived is modeled by the Poisson process $$N(t)$$.  Thus $$P(N(t) = n)$$ is the Poisson with parameters $$(\lambda, n)$$.  

$$P(N(t) = n) = \frac{(\lambda t)^n}{n!} e^{-\lambda t}$$

Then we have to compute the term $$E(Y(t) \mid N(t) = n)$$.  We know the amount of time spent is a random variable with a normal distribution.  Thus the expected value of this random variable is the mean itself or $$\mu$$.  We have $$n$$ such random variables - since they are all independent and identically distributed, the total expected value is $$n \mu$$.  Combining these terms we have 

$$E[Y(t)] = \sum_{n=0}^{\infty} \mu n \frac{(\lambda t)^n}{n!} e^{-\lambda t}$$

Now to further simplify this expression.  We can take out all terms not connected to $$n$$.

$$E[Y(t)] = \mu e^{-\lambda t} \sum_{n=0}^{\infty} n \frac{(\lambda t)^n}{n!}$$

We pull out one $$\lambda t$$ term and divide out $$n$$.

$$E[Y(t)] = \mu \lambda t (e^{-\lambda t}) \sum_{n=0}^{\infty} \frac{(\lambda t)^{n-1}}{(n-1)!}$$

Now I don't exactly remember all the series expansions from calculus any more but this one is useful - if we substitue $$y=n-1$$ the summation term turns into the Maclaurin series of the exponential.  Thus we have 

$$ 
\begin{align}
E[Y(t)] &= \mu \lambda t (e^{-\lambda t}) \sum_{n=0}^{\infty} \frac{(\lambda t)^y}{y!} \\
        &= \mu \lambda t (e^{-\lambda t}) (e^{\lambda t}) \\
        &= \mu \lambda t \\
\end{align}
$$


What a simple result! Its simply the product of the mean of the random variables and the expected value of the poisson process.  
Now for the actual result - clearly interesting new people appear in your life as a poisson process with rate 3 people a year.  Hobbies done right are a time drain, let's say mean 8 hours a week, 360 hours for the college year. Therefore based on these estimates, the expected number of hours we spend on hobbies for the year is 

$$E[Y(t)] = 3 \cdot 360 \cdot 1 = 1,080 \text{hours} !!!$$

<br>

<h3>Too Much of Something Good?</h3>
<br>

An expected average of over a thousand hours on hobbies? Woah. You thought you could keep it to under 1000 - <i><b>can we compute the probability that you spend under 1000 hours for the year? How about the chance you splurge and spend over 1500 hours?</b></i>

<br>

So we are working with the random variable $$Y(t)$$ here.  $$Y(t)$$ is the sum of all the time spent on each individual hobby. We found the expected value of $$Y(t)$$ to be 1080 hours. What else do we need in order to determine the probability that you spend less than some amount of time? First we know that each of the $$X_i$$ are normally distributed random variables and the sum of normally distributed random variables is also normally distributed.  Computing the probability that this variable takes on a value smaller than a bound sounds a lot like computing the CDF. So the CDF of a normal distribution? Luckily since the normal distribution is quite annoying to integrate, we can first convert the distribution to the $$N(0,1)$$ - normal mean 0, standard deviation 1 - and then look up the value in a normal distribution table. Much easier.

<br>

Thus how do we compute the variance of this compound poisson random variable?  We know from prior derivation that variance of a random variable can be computed as $$V(X) = E[X^2] - E[X]^2$$.  We already know $$E[Y(t)]^2$$ thus we need to compute $$E[Y(t)^2]$$. This derivation is fairly computation heavy - here's the derivation.

$$E[Y(t)^2] = \sum_{n=0}^{\infty} E[Y(t)^2 \mid N(t) = n] P(N(t) = n)$$

Let's first compute $$E[Y(t)^2 \mid N(t) = n]$$.  We know that $$Y(t)^2 = (X_1 + X_2 + ... + X_n)^2$$. In the following steps we expand out this product, take the expectation of each term, and make substituions for the variance and mean.


$$ \begin{align}
Y(t)^2 &= (X_1 + X_2 + ... + X_n)^2 \\
	   &= X_1^2 + X_2^2 + ... + 2X_1X_2 + 2X_1X_3 + ...\\

\end{align}
$$ 

$$E[Y(t)^2] = \sum_{i=0}^n E(X_i^2) + \sum_{(i, j \neq j)}^n 2E(X_i)E(X_{j}) $$


Now we want to simplify both of these terms.  From above we know that $$Var(X) = E[X^2] - E[X]^2 $$. Thus we know $$E[X^2] = Var(X) + E[X]^2 $$. The variance is $$\sigma^2$$.

$$E[Y(t)^2] = n(\sigma^2 + \mu^2) + n^2\mu^2 - n\mu^2$$

Simplifying terms we get 

$$E[Y(t)^2] = n\sigma^2 + n^2\mu^2$$

Finally we can plug this back into the original equation

$$E[Y(t)^2] = \sum_{n=0}^{\infty} (n\sigma^2 + n^2\mu^2) \frac{(\lambda t)^n}{n!} e^{-\lambda t}$$

We can factor this expression out to better understand what the result means

$$E[Y(t)^2] = \sigma^2 \sum_{n=0}^{\infty} n \frac{(\lambda t)^n}{n!} e^{-\lambda t} + \mu^2 \sum_{n=0}^{\infty}n^2 \frac{(\lambda t)^n}{n!} e^{-\lambda t}$$

Here the first term is the number of events $$n$$ times the probability of having $$n$$ events - the expectation of the poisson process. The second term likewise is $$n^2$$ times the expectation of the poisson process or the expectation of the square of the poisson process. Remember $$N(t)$$ represents the Poisson process and thus this can be expressed as 

$$E[Y(t^2)] = \sigma^2 E[N(t)] + \mu^2 E[N(t)^2] $$

Finally we can use this term in our formula for the variance of Y(t).

$$\begin{align}
Var(Y(t)) &= E[Y(t)^2] - (E[Y(t)])^2 \\
	      &= \sigma^2 E[N(t)] + \mu^2 E[N(t)^2] - (\mu \lambda t)^2\\
	      &= \sigma^2 E[N(t)] + \mu^2 E[N(t)^2] - \mu^2 E[N(t)]^2\\
	      &= \sigma^2 E[N(t)] + \mu^2 ([E[N(t)]^2 - [E[N(t)]]^2)\\
\end{align}$$ 

Now once again we can simplify this last term using the definition of the variance

$$Var(Y(t)) = \sigma^2 E[N(t)] + \mu^2 var(N(t)) $$

We know that the variance of a Poisson process and the expectation of a Poisson process is $$\lambda t$$.
Thus we finally have

$$Var(Y(t)) = \lambda t (\sigma^2 + \mu^2)$$

<br>
<h3>Buckle Down or Splurge?</h3>
<br>

We have everything we need to finally compute the probability that the number of hours you spend on hobbies is greater than or less than a given amount. We can first normalize the distribution by subtracting out the mean and dividing by the standard deviation 

$$P(X \leq a) = P(\frac{Y(t) - \mu}{ \sqrt(Var(Y(t)))} \leq a)$$

If we plug in the following values - $$\lambda$$ = 3 for the rate of the poisson process, and each of the random variables $$X$$ have mean $$\mu$$ = 360 and standard deviation $$\sigma$$ = 30. As computed earlier the expected value of the compound possion process is thus $$ \lambda \mu t = 3 \cdot 360 = 1080$$.  From the equation just derived, the variance of the compound poisson distribution is $$\lambda t (\sigma^2 + \mu^2)$$ or $$3 \cdot (30^2 + 360^2) = 391500$$.  

$$P(X \leq 1000) = P(\frac{Y(t) - 1080}{ \sqrt(391500)} \leq \frac{1000 - 1080}{\sqrt(391500)}) = -.128$$

Now since we have converted this distribution to the standard normal distribution we can look up this value -0.128 (the number of standard deviations we are from the mean) in a z table which gives us 44%! Thus there is a 44% chance that we will be able to stay below the 1000 hour mark! For the case where we want to splurge and spend over 1500 hours on hobbies we can do the same computation, we get

$$P(X \geq 1500) = 1-P(X \leq 1500) = 1-P(\frac{Y(t) - 1080}{ \sqrt(391500)} \geq \frac{1500 - 1080}{\sqrt(391500)}) = 1 - 0.67$$

Looking up 0.67 in a Z table gives us 74.8% chance we are less than 1500 which translates to 25.1% chance that we do manage to supress the inner workaholoic and splurge on new hobbies :D


