---
layout: post
title: Neural Machine Translation
date:   2021-2-12 11:43:09 -0800
categories: jekyll update
topic: nlp
image: /stock_images/NMT.png
---

<br>
<h1> Neural Machine Translation</h1>

How can you train a baseline model to perform language translation? One of the primary tasks of natural language processing that I first interacted with was Google Translate, a machine translation model that converts text in one language to another. What seemed insanely arcane to me at first has actually gotten to the point where one can now setup and train a model to do this from scratch. Let's try to build it up from a set of input sentences in one language to output sentences in another. Over the next few articles we'll make it even better. In this case, we're translating one sequence of text to another, but you'll soon come to see that it is the same underlying question with other tasks such as summarization - taking a long sequence and translating it to a much more concise version. This series is meant to cover first the conceptual underpinning's of these tasks and the key building blocks that it is built upon, then the mathematical representations of these concepts, then the building blocks for translating this conceptual understanding to a trained model and finally the nuts and bolts of making the research process smooth with the latest tools and flows to really get you up to speed with evaluating these models. Each step as important as the last :).




<h1> The Conceptual Model</h1>

For language translation, let's start off with a basic dataset which is a set of sentences in English with the corresponding translated sentences in Spanish. This is your training data. Your goal is to best predict the translated sentence given the input language sentence. First a lot of initial ideas to throw out the window - a precomputed mapping of word translations, passing a sentence as input and predicting an entire sentence as output, or even learning to directly map (think dictionary lookup) each word in the source language to another word in the target language. The last one is poor because you have no context from the rest of the sentence. Instead we will learn how to use all the information from the source sentence to predict each word individually in the output language. To get a holistic view of any problem, the first few pieces to put in place are the training data, evaluation metrics, and loss function - how do you measure differences between the predicted outputs and the actual outputs and then train the model to account for these mismatches. Since in this problem, one is trying to predict the output word by computing a probability distribution over all the possible words in the target language, the intuitive loss function here would be softmax - selecting the word with the highest probability and then checking if the predicted output word matches with the output word in the dataset. Common evaluation metrics for these tasks include ROUGE, BLEU and perplexity. We'll discuss them further towards the end of the article. But generally think computing a score of how many words were predicted accurately.



Great now to set up the model. The most common initial approach to model a sequential prediction task would be with an RNN. This essentially allows you to predict an output for each input while remembering the previous input states as well. This is set up by sharing the weights between each of the update steps. There are a few challenges with this approach however - the first is that we exponentially forget the past time steps - for more reference on this issue read here. The second is that you can only use the past in order to predict the next output. To address the first issue, several years ago an improvement to the RNN was developed - the LSTM and GRU. An LSTM addresses the issue of exponential forgetting by training both cell states and hidden states. The cell states allow for information to pass through relatively undisturbed allowing you to remember key elements from the past. For more information read here. For the second problem, essentially for a given sentence you want to be able to take both the future and previous words into consideration when predicting the output. This can be solved using a bi-directional LSTM. Now however we must keep in mind that we aren't directly predicting an output word given an input word. This would be word by word translation which is not ideal given that we want to translate with context of the entire sentence. Thus we will train an encoder LSTM that will encode the entire input sentence and create a representation that will then be fed into the decoder LSTM to perform the translation step by step. The encoder LSTM will thus have to essentially compress the entire input sequence into one cell and hidden layer. This loses a large amount of information. The solution here? We will still initialize the decoder LSTM using the last output of the encoder LSTM however when predicting the output of each time step in the decoder LSTM we want to be able to take into consideration each of the encoder states. This is the main idea captured by 'attention'. Along with the encoding that is passed in from the last state of the encoder sequence, we also take the dot product of the current decoder step with each of the encoder states.




<h1>The Mathematical Model</h1>



Let's break this all down into dimensions and equations, then implement, test and find out where we can do better. First we need to translate each of the words in the sentences into vector representations to work with. This by itself is a whole research area. Here however we have two options to work with. Either we can initialize these embeddings randomly and then train the model to learn the best word embeddings or we can go ahead and initialize the word embeddings to be from a pre-trained word embedding model such as GLoVE or global word vectors. We'll start off with a random initialization.



Great so we will have an embedding layer and a dictionary that maps each of the input words in the sentence to the corresponding word embedding for that word. Thus we have
$$x_1, ..., x_m | x_i \in \R ^{ex1}$$
Here $e$ represents the dimensions of the word embedding. With this sequence of word embeddings, we then want to pass them through the encoder to get the hidden and cell states. As discussed above, we will be using a bidirectional LSTM as our encoder. Thus we will have both the initial hidden and cell state as well as the final hidden and cell states to use as inputs for the decoder layer. These values can be concatenated to create
$$h_i^{enc} = [\overrightarrow{h_i^{enc}}; \overleftarrow{h_i^{enc}}]  \text{  where }  {h_i^{enc} \in \R^{2hx1}}$$
The same for the cell states
$$c_i^{enc} = [\overrightarrow{c_i^{enc}}; \overleftarrow{c_i^{enc}}]  \text{  where }  {c_i^{enc} \in \R^{2hx1}}$$



Now we want to pass these vectors into the decoder. The main reason we are writing this out in vector notation here is to pay attention to the dimensions before writing the code. Here since the inputs are of dimension $2h \text{ x } 1$ due to the concatenation of the forwards and backwards representations, we will have to add a linear projection layer to reduce the dimension to $h\text{x}1$ to pass into the decoder. This can be represented as

$$h_0^{dec} = W_h[\overrightarrow{h_m^{enc}}; \overleftarrow{h_1^{enc}}] \text{ where } W_h \in \R^{hx2h} $$
Similarly for the cell states we have

$$c_0^{dec} = W_c[\overrightarrow{c_m^{enc}}; \overleftarrow{c_1^{enc}}]  \text{ where }  W_c \in \R^{hx2h} $$

 This takes care of initializing the decoder with the condensed representation generated by the encoder layer. However as discussed above, the main improvement that attention brings is being able to focus on each of the encoder states again when running the decoder.

Now unlike the encoder, the decoder needs to process each of the target words one at a time so that it can compute attention as well and use the output from the previous time step to process the next word. Thus after computing the word embeddings for each word in the target sentence, we will go ahead and step through each decoding step individually. The decoder LSTM will take in the hidden state, cell state from the previous time step as well as the predicted word from the previous timestamp.

  Finally with the probability distribution over the output words generated, we will compute the overlap with the gold standard summary produced.



<h1>The Code</h1>



The implementation of this will bring that conceptual understanding to life. Now let's structure this problem in the classical machine learning framework of create your dataset, initialize your model, then set up functions for training and evaluation. For clarity we will create a NMT (neural machine translation) Class, a Model Embedding Class and a Vocab class then handle the training and evaluation.




<h3>Model Embedding Class</h3>

This will handle converting input words to their embeddings.

```

class ModelEmbeddings(nn.Module):


def __init__(self, embed_size, vocab):

	super(ModelEmbeddings, self).__init__()
	self.source_language = nn.Embedding(len(vocab.src), self.embed_size, padding_idx=pad_token)
	self.tgt_language = nn.Embedding(len(vocab.src), self.embed_size, padding_idx=pad_token)

```
<h3>NMT Model Class</h3>

Here we want to initialize the machine translation model. This is done by first defining all the pieces and layers of the model in the init function and then defining a forward function. When you call model.train() it automatically links to run the forward function defined in the model. In the init, we will define both the encoder and decoder and the necessary projection layers. Forward will handle all the logic to take a batch of input sentences in the source language, convert them to indices, get the model embeddings, and then pass them through the encoder and decoder modules. We'll then sum the number of matches between the output and the gold standard summary.


With that overview let's look at an implementation of init.

```

class NMTModel(nn.Module):

	def __init__(self, embed_size, hidden_size, vocab, dropout_rate):

		# Model Embeddings as defined in the class is a simple lookup table that stores embeddings of a fixed dictionary and size
		self.model_embeddings = ModelEmbeddings(embed_size, vocab)
		self.hidden_size = hidden_size
		self.dropout_rate = dropout_rate
		self.vocab = vocab

		# Bidirectional LSTM for the encoder to take in an entire sequence
		self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.hidden_size, bias=True, bidirectional=True)

		# LSTM Cell for the decoder to process each word
		self.decoder = nn.LSTMCell(input_size=embed_size+hidden_size, hidden_size=self.hidden_size, bias=True)

		# Projection from encoder hidden output to decoder input linear layer from hidden_size*2 to hidden_size
		self.enc_to_dec_hidden_proj = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)

		# Projection from encoder cell output to decoder input linear layer from hidden_size*2 to hidden_size
		self.enc_to_dec_cell_proj = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)

		# Attention projection is a linear layer that computes attention on the hidden encoder input. This is then multiplied with the decoder outputs so it serves to learn weights from the encoder.
		self.att_projection = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)

		# This takes the attention vector joined with the decoder hidden states and combines them through a layer. Since the attention vector is h and the hidden states are 2h this is 3hx1
		self.combined_output_projection = nn.Linear(self.hidden_size*3, self.hidden_size, bias=False)

		# Given the output vector you want to compute softmax over all the output options. This is thus size of vocabxh and u_t is hx1. Thus the final output is vocabx1.
		self.target_vocab_projection = nn.Linear(self.hidden_size, len(vocab.tgt), bias=False)

		self.dropout = nn.Dropout(p=dropout_rate)
```



Now that the model is defined, the forward function will handle all the logic you want to occur when calling model.train(). Given a mini-batch of source and target sentences, it will compute the log-likelihood of target sentences under the language models learned by the NMT system. First given the raw src and target sentence in batches you need to convert these into tensors of the ids in the vocab dictionary and pad them to all be the same length. Then we can call encode with the padded source sentences. Encode will return the hidden state of the encoder (both the cell and hidden state) as well as the decoder initial state. We can then call decode with the initial states of the decoder, the hidden state of the encoder, encoder masks (will discuss the need for masking at the end) and the target sequence. Decode will return the decoder outputs which we can then compute a softmax over in order to get the normalized probability distribution over all words in the target language. Summing this up will be the log probability of generating true target words.



```

def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:

	# Convert the word sentences into their id representations and convert to tensor
	source_padded = self.vocab.src.to_input_tensor(source, device=self.device) # Tensor: (src_len, b)
	target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device) # Tensor: (tgt_len, b)

	# Compute the encoding of the inputs
	enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)

	# Pass the dec_init state and the encoder hidden states, masks with the the target to decode
	combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)

	# Softmax over the outputs
	P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

	# Compute log probability of generating true target words
	scores = target_gold_words_log_prob.sum(dim=0)

	return scores

```

What actually happens in encode? Essentially a few operations - we want to get the model embeddings for the word indices, pad the sequences so they are all equal lengths, call the Bidirectional LSTM cell on the output, uncompress the padding, and then prepare the inputs to the decoder layer. This is done by combining the first and last hidden and cell states and then computing the projections to get the decoder initial state. We want to return both the encoder hidden state and this projection to pass in as the decoder initial state. An important implementation detail to point out here is the use of torch.nn.pack_padded_sequence and torch.nn.pad_packed_sequence. This is important because the current representation of the sentences is as a batch of equal length inputs where the empty tokens are pad tokens. When training an LSTM, we went ahead and padded the input since the length of sequences in a size 8 batch for example is [2,6,7,3,4,7,5,3]. This results in 64 computations (8x8), but you needed to do much fewer not including the padded tokens. Pytorch packs the sequence into a tuple of two lists. One contains the elements of sequences and other contains the size of each sequence the batch size at each step. Pad_packed_sequence does the reverse operation. Think of this as a useful optimization hack.

```

def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

	enc_hiddens, dec_init_state = None, None

	# Model_embeddings implements the torch.nn.Embedding layer which learns word vectors for a vocabulary
	X = self.model_embeddings.source(source_padded)

	# Pack_padded will essentially compress the padding before calling the encoder
	X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths=source_lengths)

	# An LSTM returns output, (h_n, c_n) where h_n is the hidden layer of the nth step and c_n is the cell state
	enc_hiddens, (last_hidden, last_cell) = self.encoder(X)

	# Pack_padded_sequence will readd the padding. We want to transpose to get (batch, hidden_size)
	enc_hiddens = torch.nn.utils.rnn.pack_padded_sequence(enc_hiddens)[0].transpose(0,1)

	# The encoder is a bidirectional LSTM, thus the last_hidden layer will have shape (2, batch_size, hidden_size). We concatenate the outputs of the forwards and backwards layer along the 1st dimension to stack vertically.
	last_hidden = torch.cat((last_hidden[0], last_hidden[1]), 1)

	# Compute a projection on this last_hidden tensor to reduce the dimension to pass into the decoder
	init_decoder_hidden = self.enc_to_dec_hidden_proj(last_hidden)

	# Stack the cell states similar to the hidden states
	last_cell = torch.cat((last_cell[0], last_cell[1]), 1)

	# Compute a projection to reduce the dimension
	init_decoder_cell = self.enc_to_dec_cell_proj(last_cell)

	# The decoder init state is a vector of the hidden and cell states computed above
	dec_init_state = (init_decoder_hidden, init_decoder_cell)

	return enc_hiddens, dec_init_state

```

From this encode function we get back the hidden state as well as the initial states for the decoder. The decoder then has to take this entire representation of the source sentence and use it to predict a target word at each time step. The decoder will return the probability distribution across the batch of words for time step to the forward function which is responsible for taking that and comparing against the actual target word in the training dataset. It uses this comparison to compute the loss. Since unlike the encoder the decoder has to predict the output word by word, we will break this function up into decode and a step function. We want to call the LSTMCell on each value Y in the target sentence. What are the appropriate inputs to the decode function? What do you have to pass into an LSTMCell?  What other operations does decode have to perform? It takes in Ybar_t which is a concatenation of y_t with o_t which is the output of the previous state. Then you pass in dec_state which is the previous time step's decoder hidden and cell states. Along with the encoder hidden states and the encoder hidden state projection. The encoder hidden state and projection state remain the same for all time steps. At each time step, the step function will return the output of the LSTMCell as well as the decoder cell and hidden states. Each of the output tokens will be stored and returned to the forward function in order to compute the loss with the target words. With that general framework of how we'll be stepping through each word and passing it into the decoder through the step function, let's write this function to essentially loop through all the words in the target sentence Y and pass each word, the decoder's previous state and the encoder hidden state and proj to the step function.



```

def decode(self, enc_hiddens, enc_masks, dec_init_state, target_padded) -> torch.Tensor:

	# Initialize the decoder state (hidden and cell) with the input passed in from the encoder
	dec_state = dec_init_state
	# For the first time step, the previous combined output vector o_{t-1} will be the zero vector. Since we are working over a batch, the batch size is the size of enc_hiddens
	batch_size = enc_hiddens.size(0)

	# The prev output vector should be size batch_size, hidden_size which is the length of the hidden state
	o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

	# Initialize a list of predictions
	predictions = []

	# First we want to compute the attention projection over the entire enc_hiddens
	enc_hiddens_proj = self.att_proj(enc_hiddens)

	# Need to convert the target words into tensors by passing it through model_embeddings trained on the target language
	Y = self.model_embeddings.target(target_padded)

	# We iterate over a tensor using torch.split():
	for y_t in torch.split(Y, 1, dim=0):

		# The dimensions here are (1, batch_size, embed_size) so we need to sqeeze to get it to a 2-d vector. This is because we are taking the first word across each sentence in the entire batch
		y_t = torch.squeeze(y_t)

		# We want to pass both the output y_t and the previous prediction o_t as the decoder input so we concatenate. Dim=1 because they should be stacked not appended horizontally
		decoder_input = torch.cat((y_t, o_t), dim=1)

		# The step function will take in the inputs to the LSTM cell (decoder_input, dec_state) as well as enc_hiddens, enc_hiddens_proj and enc_masks in order to compute attention at each time step. It returns the output word and dec_hidden_state
		output_word, dec_hidden_state, e_t = self.step(decoder_input, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)

		# Reset the dec_state to the output of step dec_hidden_state
		dec_state = dec_hidden_state

		# Add the output word to the prediction
		predictions.append(output_word)

	# Stack all the individual tensors of shape(batch_size, hidden_size) into one tensor of shape (tgt_len, batch_size, hidden_size)
	torch.stack(predictions, dim=0)

return predictions

```


Great, so that handled computing attention and iterating over the batch of target sentences one word at a time in order to pass those inputs into the step function. Here we want to compute one forward step of the LSTM decoder and compute attention. The output of this step will be the distribution over output states as well as the decoder hidden and cell states.

```

def step(self, dec_input, dec_state, enc_hiddens, enc_hiddens_proj):
	# First apply the LSTMCell to the decoder_init and dec_state from the previous time step
	dec_hidden, dec_cell = self.decoder(dec_input, dec_state)

	# Using the encoder_hidden_proj, compute the attention scores by multiplying the enc_hiddens_proj with dec_hiddens. Essentially adding a dimension to allow this computation (batch_size x max_len x embed_size) x (batch_size x embed_size).
	attn_weights = enc_hiddens_proj.matmul(dec_hidden.unsqueeze(2)).squeeze(2)

	# Using the attention weights you want to softmax to normalize and then multiply with dec_hiddens to project the weights on the decoder hidden state
	alpha_t = torch.softmax(e_t, 1)

	# Looking at matrix sizes alpha_t is (b, src_len), enc_hiddens is (b, src_len, 2h) thus, unsqueeze to add dimension
	alpha_t = alpha_t.unsqueeze(1)

	# Multiply the normalized softmax scores with the encoding layers and remove the extra dimension
	a_t = alpha_t.matmul(enc_hiddens).squeeze(1)

	# Stack the decoder hidden layers and the attention tensor
	U_t = torch.cat((dec_hidden, a_t), dim=1)

	# Linear projection on this combined input
	V_t = self.combined_output_projection(U_t)

	# Compute the output prediction
	O_t = self.dropout(torch.tanh(V_t))

	combined_output = O_t

return dec_state, combined_output, e_t

```

That's all for setting up the NMT Model!

The last step here is to set up a default pipeline to train and evaluate this code.  Essentially that involves loading the sentences through the Vocab class to convert it to the ids and then calling model.train() once the model is initialized.  Then for epoch, loop through all batches, sum over the losses until the loss doesn't change.  Then run evaluate to see how the model performs!   In the next article we'll look more into how to deploy this on a GPU and train and evaluate. There are several approaches if you don't have access to your own GPU - Google Colab or dockerizing your code and deploying to a VM such as AWS or Google Cloud.