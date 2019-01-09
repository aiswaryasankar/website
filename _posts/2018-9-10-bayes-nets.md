---
layout: post
title:  Bayes Nets
date:   2018-12-15 11:43:09 -0800
categories: jekyll update
topic: AI
---


How can I find the probability I have a certain medical condition given that I have a set of symptoms?

<h3>Background Story</h3>
<p style="margin-left:25px">
Now thinking through which treatment was best for me, I realized that my condition was unique.  My age, the level of calcium in my diet, and my exercise level decreased my chances of osteonecrosis but my family history of osteoperosis, relatively higher levels of stress hormones generally increased my chances.  How could I take all of these factors into account when determining the probability I had osteonecrosis?  Is there some way I could factor in all of these interrelated determinants into my calculation?  Now I don’t yet know what these probabilities are for my condition, so until I figure them out I’ll explore this problem using a related medical query.  Let’s consider the following.  You notice that you have 2 of the symptoms of one ailment and 5 of the symptoms of another.  However you fit more of the precursors for the first ailment.  You want to take a shot at diagnosing yourself or at least have an educated guess at how your doctor will diagnose you.  You really want to know the likelihood that you have each of the ailments.
</p>

Let's say you know the following probabilities, grouped as the probability of a medical condition and the probability one has a symptom given he or she has the medical condition.

P(Smokes)​\\
P(lung disease | smokes)\\
P(shortness of breath | lung disease)\\
P(chest pain | lung disease)\\
P(cold)\\
P(cough | cold and lung disease)\\
P(fever | cold) \\

<b>Using these known probabilities can we find P(Cold | Shortness of breath, Cough)?</b>

<br>
<hr>
<br>

<p style="margin-left:25px">
There are some key probability concepts underlying this query.  Yet before really thinking about these concepts, how could you approach this intuitively?  We know the probability of having a cough given that you have a cold.  Its often the reverse question that’s more interesting though.  Given the symptom or symptoms what is the probability you have a certain condition?  This is the fundamental question we will soon be able to answer.
</p>

<h4>So, how do we model this formally?</h4>
<p style="margin-left:25px">We first need to turn all of these probability tables into a better representation.  Here we have the general structure and properties of a Bayes Net.  We create a graphical representation where each variable is a node.  Then for each conditional probability that we have, we draw an arc.  The arc would represent the fact that knowing one variable gives you additional information about the other.  Thus P(A|B) would be modeled with with an arc from B to A since given B you have more information about A.
</p> ​

<h4>Can we pick out any specific probability from this structure?</h4>

<p style="margin-left:25px"> Essentially can you determine any conditional probability you want?  Yes.  It doesn't matter which order or exactly which conditional probability tables you know initially.  Let's look at the following variations of this Bayes Net.  
</p>

![Bayes](/stock_images/bayes.png){:style="float: center;margin-left: 100px; margin-top: 7px :height="650px" width="550px" :class="img-responsive"}

<p style="margin-left:25px">
Perhaps you don't care about Shortness of Breath or Fever.  As in Bayes Net 1, we can simply remove them.  Or perhaps you have additional information about P(cough | pneumoia), we can add that in.  Perhaps you don't actually know   This seems so powerful and with so much flexibility.  Are there things we have to guarantee hold though?  Are there any limits to this flexibility?  Really digging into why this works, I’ve understood this entire structure and its properties stem from very simple and intuitive facts of probability.  There are two key facts that you need to show regarding why this holds, the chain rule for probability and the conditional independence assumption for Bayes Nets.  
</p>

<h4>What is the chain rule of probability?</h4>
<p style="margin-left:25px">
This simply is nothing more than a manipulation of the definition of conditional probability.  When I say probability a person has lung disease given he or she smokes, we are asking of all the people that smoke, what percentage of them have lung disease?  First you figure out the percentage of people that smoke.  Then of these people you find the percentage of people that have lung disease.  Take the fraction and you have the probability of lung disease given that one smokes.  This can be written in the following way:
</p>

$$P(\text{lung disease | smoke}) = \frac{\text{P(lung disease and smoke)}}{\text{P(lung disease)}} $$

Now the chain rule of probability simply solves for the joint probability instead by moving P(smoke) over

$$P(\text{lung disease and smoke}) = P(\text{lung disease | smoke} P(\text{smoke}))$$

We can follow the same logic and expand this to multiple variables, thus:

$$P(x_1, x_2,.. x_n) = P(x_1)P(x_2|x_1)...P(x_n|x_1,x_2...x_{n-1}) $$

<br>
<h4>What is the independence assumption of a Bayes Net?</h4>

<p style="margin-left:25px">Now just one more thing.  You want to know the probability a person has chest pain.  If you know whether or not the person has lung disease, does knowing whether the person smokes give you more information?  Essentially you can think of it this way.  You know event A occurs and you know the probability B occurs given A has occurred.  Do you need to know what caused A to know the probability of B?  Nope.  Thus in Bayes nets we assume a node is independent of its grandparents and above given its parent nodes.
</p>  
<br>
<h4>How do these properties prove we can find any conditional probability?</h4>

<p style="margin-left:25px">The two things we know so far are that you can compute the probability of multiple events occuring as the product of conditonal probabilities and that a node is independent of all nodes given its parent.  

We know $$P(A,B) = P(A)P(B|A)$$ and if C is parent of B is parent of A, $$P(A|B) = P(A|B,C)$$
     
Now let's use the following three facts as well:
<ul>
<li>The sum of a joint probability = 1</li>
<li>A Bayes Net can be split into multiple conditional probabilities</li>
<li>A joint probability represents all the possible variable combinations</li>
</ul>
<p style="margin-left:25px">
From here, what is the joint probability of multiplying together all the conditional probability tables in our Bayes net?  Let's look at each one individually.  Therefore we can multiply all the CPTs and get a full joint distribution.  In the next parts we will go on to show that once you have a full joint distribution you can evaluate any conditional probability through the process of marginalization and normalization.  Essentially getting rid of variables you don't care about and then reversing a joint probability into a conditional probability if desired.</p>
<br>
<h4>How can we make this process more efficient?</h4>

<p style="margin-left:25px">Now all one needs is a quick glance at the following Bayes net describing factors that influence liver disorder to realize that run time and efficiency will be essential considerations in making inference using Bayes Nets a reality.  Once again here’s a section for those curious to know more about the actual computer science implementation details of this algorithm.

There's the fairly obvious question now, are all of these nodes actually related?  For example if I were to ask which nodes are independent?  Does knowing cough actually tell me anything about knowing whether I have a fever?  These are all very interesting questions and there are certain structures in a Bayes Net that can guarantee you the answers to these queries.  Look further into a technique called 'd-separation' if you are interested.

Regarding my situation, had I been able to accumulate all the necessary probabilities to create this Bayes Net I would have been much more informed regarding the chance of having osteonecrosis.  Generally attaining these probabilities may be challenging however I do believe that due to the large increase in all the data that is being stored regarding patients, more and more correlations between variables and symptoms will become apparent.
</p>
