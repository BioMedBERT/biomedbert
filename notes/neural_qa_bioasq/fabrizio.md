# Neural Question Answering at BioASQ5B
- [ArxiV](https://arxiv.org/pdf/1706.08568.pdf)
- Extractive Models
- Pre-trained the model on a large scale open domain QA dataset (SQUAD) and then fine-tuned on BioASQ training set.
- State of the Art (at the time) on FACTOID questions
# 1 Introduction
BioASQ is a Semantic Indexing, question answering and extraction challenge.

For Factoid Questions the system's responses are interpreted as a ranked list of an answer candidates. 
They are evaluated using mean-reciprocal-rank (MRR).

Approach:
	- Train FastQA on the SQUAD dataset and then fine tune on BioASQ
# 2 Model
## 2.1 Network architecture

- The Embedding is a concatenation of various elements:
	- Question Type Features:
		- (see Fast QA)
	- Character Embedding (this is important)
	- GloVe Embedding (\ref?)
		- 300d general english
	- Biomedical Embedding
		- (from: Pavlopoulos et alt 2014)
		- Word2Vec 200d (Mikolov et al 2013)
	- one-hot encoding of question type
- These embedding composition is used for both The Context and The Answer
- Inside Trick: Compute the start probability via the sigmoid rather than softmax function to be able to output multiple spans as likely answer spans.
	- This generalizes the factoid QA network to list questions

## 2.2. Training and decoding
LOSS: We define our loss as the cross-entropy of the correct start and end indices.

### Dataset Preparation
- extract answer spans by lookup.
	- this is IMPRECISE and can lead to BAD input data on a SMALL dataset.
### Decoding
- Beam Search to return the top 20 answers span.
- remove all duplicate strings
- output the top five answer strings as our ranked list of answer candidates
- *PROBABILITY CUTOFF THRESHOLD $t$*
	- WE set $t$ to be the threshold for which the list F1 score on the development set is optimized.

### Ensemble
- 5 fold cross-validation of the entire training set

### Implementation
- Tensorflow (\ref{abadi et al 2016})

# 3 Results & discussion
- wining 3 out of the 5 batches
- just answering yes to the yes/no challenge provides a strong baseline because the dataset is SKEWED.
