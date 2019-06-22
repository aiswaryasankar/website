---
layout: post
title: The Probability Bowl
date:   2019-3-9 11:43:09 -0800
categories: jekyll update
topic: probability
image: /stock_images/erlang.png
---

The EECS + Math departments decide its time to hold a competition to see who knows their probability chops the best. Exciting!! The first round is divided into 3 groups. The top 3 finishers in each group are awarded finalist medals - the top finisher moves on to the final round. The final round consists of one 'Problem of the Gods'. The time it takes you to finish the problem (if you can) determines the winners.  Two of my friends and I are jokesters and make a bet that we'll sweep the competition placing 1st, 2nd and 3rd in a certain order. Can we compute the probability that we do? 

<br>
<hr>
<br>

So exciting. 
<h3>Becoming a Finalist</h3>
Before we can realize our true goals of sweeping the contest, my friends and I must first place in the top 3 in each of our respective groups. Let's say the groups are evenly split and there are $$n$$ people in each group. The stakes are high in this exam - you aren't allowed to submit your test until you get every question correct. Thus the only determinant in who wins is the time taken to complete the exam. The time it takes for any person to finish the exam can be modeled as an exponential distribution. Each of us thinks that we have worked the hardest, yet the truth is that we all have the same rate of completion for the exam. <b><i>How can we now determine the probability that amongst the entire pool, we will individually place in the top 3?</i></b>

<br><br>
We can first write out random variables $$X_1$$, $$X_2$$,... $$X_{n}$$ representing the time it takes each person to complete the exam. Each can be represented as an exponential distribution $$X_1 = exp(\lambda) $$, $$X_2 = exp(\lambda)$$. Now once everyone finishes and submits the exam we can write down the ordering of these finish times - for example the finish times can be ordered as the following from shortest to greatest $$X_4, X_7, X_2, X_9, X_1, ... $$.  Thus in this case person 4 finished the quickest and won the competition with participants 4, 7, and 2 being awarded finalist medals. 

<br><br>
Thus the probability that we are in the top 3 can be computed as the sum of the probability we place 3rd, place 2nd and place 1st. In order for my friend to place 3rd in her group there must be 2 people who have completion times less than her and the rest of the group must take longer. Let's say that she takes $$a$$ minutes to finish the exam. What's the probability that any person, let's say $$X_1$$, took less than $$a$$ minutes to finish the exam?  It would be $$P(X_1 \leq a) = F(x)$$ where $$F(x)$$ represents the CDF of each $$X_i$$. Likewise the probability that someone takes more time would be $$P(X_1 > a) = 1 - F(x)$$.  

<br><br>
Now we know that each person has probability $$F(x)$$ of taking less time and probability $$1 - F(x)$$ of taking more time. This sounds like the binomial! It doesn't matter which person takes more or less time and thus we can write out the probability that $$k$$ people take less time as 

$$P(X_k \leq a) = \sum_{j=k}^{n} F(x)^k (1 - F(x))^{n-k}$$

We can formalize the probability she placed 3rd out of the 10 people in her group as 

<br>

$$P(X_3 \leq a) = \sum_{j=k}^{9} F(x)^2 (1 - F(x))^{7}$$ 

<br>

Similarly we can do the same to compute the probability that she took the least time to complete the exam. In this case everyone else would have taken more time giving us

$$P(X_0 \leq a) = \sum_{j=k}^{n} F(x)^0 (1 - F(x))^{n}$$

<br>
<h3>Operation: Competition Sweep</h3>


Against all odds (since everyone in the competition had the same rate of completing the exam this truly is quite a feat) my two friends and I do manage to each win first place in our batches. Now its time "to sweep" :D.  Here the stakes are high though and we each are honest about how quickly we think we can solve very complex problems. We each put forth our estimated rates and to preserve anonymity, let's say I take 65 minutes per hard problem, $$\lambda$$, Alice takes 50 minutes, $$\mu$$, and Eve takes 45 minutes $$\rho$$. 

I bet we'll place in the order of our rates with 
