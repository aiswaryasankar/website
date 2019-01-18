---
layout: post
title:  Medical Markov Decision Process
date:   2018-10-15 11:43:09 -0800
categories: jekyll update
topic: AI
---


How can a doctor choose the best treatment option for a patient at each time step?

<h3>Background Story</h3>
<p style="margin-left:25px">
This situation comes from a personal experience.  I had a broken sesamoid bone in my right foot and my doctor initially told me to rest and stay off physical activity for a week.  A month later though when it wasn't healing we decided to take an MRI to see if something else was going on.  The results weren't good - the MRI showed osteonecrosis or bone death.  I was left with one of two options; either I could go through an experimental bone growth therapy or undergo a surgery.  My doctor reasoned since I was young we should try the bone regrowth since surgery could leave me with future complications.  I agreed, yet I was very weary of waiting longer if it ended up I would need the surgery anyway.  Ultimately I did need the surgery and it was only a whole 9 months later that I was healed with that bone removed.  This experience was fraught with me thinking of the likelihood of success for each treatment option, potential reward and how the action would set me up for the future.  Why did we put off the surgery so long? 
</p>
<br>
<b>Could we have designed a more optimal treatment plan especially since my doctor claimed she had data regarding the success rates and adverse effects of each of the treatments?</b>

<hr>
<br><br>
<h4><b>How can we model this problem?</b></h4>
     
This is where I started becoming highly interested in how Markov decision processes (MDP) could be used to tackle this question.  There are key components to any medical situation as there are with MDPs.
<ul>
    <li>States - where an actor is at a given time (patient condition)</li>
    <li>Actions - something an actor does to move from state to state (treatment)</li>
    <li>Transition probability - probability if you take an action from a certain state you will land in another state </li>
    <li>Reward function - reward or utility an actor gains from being at a particular state</li>
</ul>

Each of these components appear to match my medical situation well.  Possible states and actions would include:
<ul>
    <li>State space: (high pain, broken bone), (no pain, broken bone), (no pain, bone healing), (no pain, bone removed)</li>
    <li>Actions: take aspirin, experimental ultrasound, no treatment, surgery</li>
</ul>

Each of these actions have different states they can result in with different probabilities of each.  What actions would take me from (Broken bone, High pain) to (Healing Bone, No Pain)?  No Treatment I estimate would have a 0.3% chance of ending in state (Broken bone, No Pain), a 0.1% chance of transitioning to (Healing Bone, No Pain) and a 0.6% chance of ending in state (Broken bone, Pain).  A diagram seems helpful to illustrate this situation.  I can model similarly for the remaining states and actions.

![MDP](/stock_images/mdp.png){:style="float: center;margin-left: 150px; margin-top: 7px :height="550px" width="450px" :class="img-responsive"}

<h4><b>How can we determine rewards?</b></h4>
<p style="margin-left:25px">
Here is where a really fascinating and perhaps the most valuable feature of a MDP comes into play.  I'll preface it with a scenario.  Let's say one treatment has a high probability of success and will greatly improve my condition.  Another treatment may only improve my condition slightly with similar probability.  Given only this information one would choose the first option.  However let's say the follow up treatment from the first treatment has a high probability of a ver bad side effect while the second treatment doesn't.  Given this additional information of future possible options, option two may very well be better.  Now this high chance of an adverse state may not be in the immediate next action but 4 or 5 subsequent treatments later!  You want to take all these potential future states into consideration when determining you action from the current state.  How?
</p>

<h4><b>How can we take future states into consideration?</b></h4>
<p style="margin-left:25px">
Simply put we need to keep track of a fraction of the reward attained at future states in our state reward.  I want to choose the action, No Treatment or Aspirin that will give me the highest reward - not just for the current state but in the scope of my entire treatment.  First we know that we can only get the value from one action so of all the possible treatment options, let's choose the one that will give us the maximum value.  Then how do we compute the value for having taken a specific action, let's say No Treatment?

Well I have a .6% chance of getting the reward from (Broken bone, Pain), a .3% chance of reward from (No pain, Broken bone) and a .1% chance of reward from (No pain, Healing bone).  Thus the expected value would be sum over all the products of the transition probabilities and their values.  

Now we can translate this into an equation.  First I'll describe it in words and then we can put it into symbols.  From a state we consider all actions one can take.  For a given action then its value is the sum of the values of all the states you can reach with that action weighted by the probability of reaching that state.  The value at that state is the reward from moving to that state plus the value at that state.  This leaves us with the following equation:
</p>

$$V(s) = max_a \sum_{s'} [T(s,a,s')[R(s,a,s') + \gamma V(s')]] $$

<h4><b>Now lastly the big question: how do we know the value of the future state when computing the current state value?</b></h4>
<p style="margin-left:25px">
Well we know the final state value which is the terminal reward I expect to get once I am completely healed.  Thus once we know this value we can use it to compute the value of all states one time step prior to the terminal state.  From here we can continue to propagate these values all the way until we know the value of the first state.
</p>

<h4><b>Have we found the optimal treatment?</b></h4>

<p style="margin-left:25px">   
​Great.  Let's say we perform this process on my medical case and determine that the value of (No pain, Broken bone) is 35 and (No pain, Healing bone) is 45.  Has this solved the initial question?  What treatment should I take at this time?

Thinking about this makes me realize that I don't actually know which treatment to take just from the state values.  Did I take No Treatment to end up at (No pain, Healing bone) or did I take Aspirin to get there?  We can solve this issue by using the given values we've determined, plugging them into our formula and determining which action gave us the max value.  Essentially we are computing the values for each given action and then choosing which action gave us the maximum.  As an equation this can be written with a small tweak to the above equation:
</p>

$$V(s) = argmax_a \sum_{s'} [T(s,a,s')[R(s,a,s') + \gamma V(s')]] $$

<p style="margin-left:25px">
​Thus we now know which treatment action will be optimal while accounting for all the future treatments, the probability they will succeed and the utility I will get from undergoing that treatment!  
</p>
