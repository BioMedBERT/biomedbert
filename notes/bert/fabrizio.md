# BERT Notes 

# 1. Introduction
- Bidirectional Models with Masked Language Models, reach the state of the art on 12 NLP tasks
- When training the objective is to predict the original vocabulary id of t he masked word based only on its context.

# 2. Related work
## 2.1 Unsupervised Feature-base approaches 
- objectives to discriminate correct from incorrect words in left and right context (Mikolov 2013)
- objectives generalized to coarser embeddings such as sentence embeddings or paragraph embeddings
- other objectives: rank candidate next sentences, left to right generation of next sentence words give a representation of the previous sentence, denoising autoencoder derived objectives.
- ELMo extracts *context-sensitive* features from a left-to-right and a right -to-left language model. concatenate the two contexts to get the word
- The Cloze task:
	
	 A cloze test (also cloze deletion test) is an exercise, test, or assessment consisting of a portion of language with certain items, words, or signs removed (cloze text), where the participant is asked to replace the missing language item. Cloze tests require the ability to understand context and vocabulary in order to identify the correct language or part of speech that belongs in the deleted passages. This exercise is commonly administered for the assessment of native and second language learning and instruction.
	
## 2.2 Unsupervised Fine-tuning Approaches
- More recently, sentence or document encoders which produce contextual token representations have been *pre-trained* from *unlabeled text* and fine-tuned for a supervised downstream task
## 2.3 Transfer Learning from Supervised Data
- Transfer learning is important

# 3. BERT
- Two steps process:
	- Pre-training
	- Fine-tuning
- Each downstream task has separate fine-tuned models, even though they are initialized with the same pre-trained parameters
- Unified Architecture across Tasks
### Model Architecture

Multi-layer bidirectional transformer encoder based on the original implementation.

- Transformer Blocks (Layers): **L**
- Hidden size: **H**
- self-attention heads as **A**

BERT *Base* (L=12, H=768, A=12) 110M Param
BERT *Large* (L=24, H=1024, A=16) 340M Param

feed forward filter size is set tot 4*H
- H = 768 x 4 = 3072
- H = 1024 x 4 = 4096

### Input / Output representations
- able to represent both single sentence and pair of sentences (i.e. Question, Answer)
- WordPiece embeddings
	- 30k vocabulary
	- [CLS] as always the first token
	- The final hidden state corresponding to this token (the [CLS] token) is used as the aggregate sequence representation for *classification* tasks.
- [SEP] Token to separate the sequence + a Mask. (Learnable Embedding??)
- $E$: input Embeddings
- $C \in \R^{H}$: the special `[CLS]` token 
- $T_i \in \R^{H}$ the final vector for the $i^{th}$ input token

- For a given token, its input representation is constructed by *summing* the corresponding `token` + `segment` + `position` embeddings
	- segment embedding?
	- position embedding?

## 3.1 Pre-training BERT
Two unsupervised Tasks:

### Task#1 Masked LM:
- standard conditional models can only be trained left to right or right to left, since bidirectional conditioning would allow each word to indirectly see itself, and the model could trivially predict the target word in a multi layered context. 
- **Masked LM** (**MLM**)we simply mask some percentage of the input tokens at random and then predict those masked tokens. 
- The final hidden vectors corresponding to the mask tokens are fed into an output softmax over the vocabulary as in standard LM.
- In all of our experiments, we mask 15% of all *WordPiece* tokens in each sequence at **random** 
#### The case for `[MASK]` tokens:
The `[MASK]` tokens do not appear during fine-tuning.
To mitigate this problem, the training data generator choses 15% of the token positions at *random* for prediction.

If the `i`-th token is chosen we replace it with:
	- The `[MASK]` token 80% of the time
	- a *random* token 10% of the time
	- the same token again 10% of the time
$T_i$ will be used to predict the original token with cross entropy loss.

### Task #2: Next sentence Prediction 

- Many NLP tasks are based on understanding the *relationship* between two sentences, which is not directly captures by language modeling
- we pre-train a *binarized* *next sentence prediction* task that can be trivially generated from any monolingual corpus.
	- when choosing the sentences *A* and *B* for each pre-training example, 50% of the time *B* is the actual next sentence that follows *A* (IsNext)
	- NotNext: 50% of the time it is a random sentence from the corpus
- BERT transfers **all** parameters to downstream tasks.

### Pre Training data
- BookCorpus (800M words)
- English Wikipedia (2,500M words)
	- Extract only the text passages
	- Ignore lists, tables and headers
- Is critical to use a document-level corpus rather than a shuffled sentence-level corpus.

## 3.2 Fine tuning BERT
- encoding a concatenated text pair with self attention effectively includes bidirectional cross attention between two sentences (??)
- Relatively cheaper to fine-tune


## 4 Experiments
11 NLP tasks:
### 4.1 GLUE
- batch size of 32
- fine-tune for 3 epochs
- learning rates: 
	- 5e-5, 4e-5, 3e-5, 2e-5
- $BERT_{LARGE}$ is unstable on small dataset
	- use random restarts

on fine tuning:
	- $S \in \R^{H}$ start Vector 
	- $E \in \R^{H}$ end Vector 

Training objective is the sum of the log-likelihoods of the correct start and end positions.


## 4.3 SQUAD
TODO
## 4.4 SWAG
TODO

# 5 Ablation Studies
## 5.1 Effect of Pre-training Tasks
- No NSP
- LTR & No NSP
they both fail miserably

## 5.2 Effect of Model Size
- Increasing the embeddings from 200 to 600 helps but from 600 to 1000 did not bring further improvements

## 5.3 Feature based approach with BERT
- Case preserving WordPiece model
- Maximal document context provided by the data

- we use the representation of the first sub-token as the input to the token level classifier over the NER label set.



# Appendix

## A.1 Illustration of Pre-training Tasks

### Masked LM and and the Masking procedure
- the 80%, 10%, 10% strategy is to bias the representation towards the actual observed word

The Transformer encoder does not know which words it will e asked to predict or which have been replaced by random words, so it is forced to keep a distributional contextual representation of **every** input token.


Ideas:
# should we encode the space too? it will allow the model to group the word pieces together.
[CLS]THE[SPACE]MAN[SPACE]went[SPACE]to[SPACE]

### Pre-Training procedure
- sample two sentences from the corpus
- less than 512 tokens combined
- LM masking is applied AFTER WordPiece Tokenization

- batch size: 256 sequences * 512 tokens = 128k token/batch
- steps: 1M
- epochs: ~40 
- optimizer: Adam, 1e-4, beta_1 = 0.8, beta_2 = 0.999
- L2 weight decay of 0.01
- lr: 
	- 10k warm up 
	- linear decay
- dropout:
	- 0.1 on *ALL* layers (?)
- activation: GELU
- loss: mean masked LM likelihood + mean next sentence prediction likelihood
- each pre-training took 4 days

- longer sequences are very expensive because attention is quadratic to the sequence length.
- sequence lengths of 128 for 90% of train 
- 10% on 512 sequence length.

 ## A.3 Fine tuning Procedure
 we change only:
 - batch size: 16, 32
 - learning rate: 5e5, 3e-5, 2e-5
 - number of training epochs: 2, 3, 4
 - do grid search on the best Hyper parameters for small datasets.

## B. TODO

# C Additional Ablation Studies
## C.1 Effect of Number of Training Steps
- Does BERT really need such a large amount of pre-training to achieve high fine-tuning accuracy?
- YES: +1.0 accuracy between the 500k and the 1M steps

## C.2 Ablation for Different Masking Procedures


## Ideas:
- increase the dropout at the start?

