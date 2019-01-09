---
layout: post
title:  "No More Tea"
date:   2018-12-10 11:43:09 -0800
topic: probability
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

<style>
body {font-family: Arial, Helvetica, sans-serif;}
.modal {
  display: none; /* Hidden by default */
  position: fixed; /* Stay in place */
  z-index: 1; /* Sit on top */
  padding-top: 100px; /* Location of the box */
  left: 0;
  top: 0;
  width: 100%; /* Full width */
  height: 100%; /* Full height */
  overflow: auto; /* Enable scroll if needed */
  background-color: rgb(0,0,0); /* Fallback color */
  background-color: rgba(0,0,0,0.4); /* Black w/ opacity */
}

/* Modal Content */
.modal-content {
  background-color: #fefefe;
  margin: auto;
  padding: 20px;
  border: 1px solid #888;
  width: 80%;
}

/* The Close Button */
.close {
  color: #aaaaaa;
  float: right;
  font-size: 28px;
  font-weight: bold;
}

.close:hover,
.close:focus {
  color: #000;
  text-decoration: none;
  cursor: pointer;
}
</style>
<body>

<!-- Trigger/Open The Modal -->
<button id="myBtn">Binomial Distribution</button>

<!-- The Modal -->
<div id="myModal" class="modal">
  <div class="modal-content">
    <span class="close">&times;</span>
    <p>Here's the derivation!</p>
  </div>

</div>

<script>
// Get the modal
var modal = document.getElementById('myModal');

// Get the button that opens the modal
var btn = document.getElementById("myBtn");

// Get the <span> element that closes the modal
var span = document.getElementsByClassName("close")[0];

// When the user clicks the button, open the modal 
btn.onclick = function() {
  modal.style.display = "block";
}

// When the user clicks on <span> (x), close the modal
span.onclick = function() {
  modal.style.display = "none";
}

// When the user clicks anywhere outside of the modal, close it
window.onclick = function(event) {
  if (event.target == modal) {
    modal.style.display = "none";
  }
}
</script>



