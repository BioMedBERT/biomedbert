# [Pre-trained Language Model for Biomedical Question Answering](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506)

BioBERT pre-trained on SQuAD or SQuAD 2.0 easily 

Notes status: [DRAFT]

# 1. 
- Language models have been shown to be highly effective in question Answering (QA)
- Language models are mostly pre-trained on general domain corpora, they cannot be generalized to biomedical corpora.
- A LM pre-trained on biomedical corpora is needed for building effective biomedical QA models.
- In this paper we investigate the effectiveness of BioBERT in biomedical question answering and report our results from the 7th BioASQ Challenge.
- Challenges
	- dataset is very small (few thousands samples)
	- very expensive to create
- There are various types of questions:
	- factoids
	- lists
	- yes/no

- BIO-BERT
	- trained in other large scale extractive question answering  datasets, and then fine-tune it on BioASQ.
	- Modify last layer to answer all the possible questions types.
- Contributions:
	- PreTrain on SQUAD and SQUAD 2.0 largely improves the results
	- Achieve best overall performances in BioASQ
	- Achieves SOTA in BioASQ 6b phase B.
	- Analyze pre-post processing strategies and their different results.
# 2 Methods
## 2.1
- BioBERT  is better than BERT in bioNLP
	- uses WordPiece embeddings (the original BERT ones)
	- add [CLS] token at the start of the sequences
## 2.2 Task-specific layer
- list and factoids uses the output
- yes/no uses the [CLS]

## 2.3 
- list/factoid:
	- softmax for the list / factoid on the E output
	- cross-entropy
- yes/no:
	- sigmoid
	- binary cross-entropy

## 2.3 PRE-Processing
According to the reals of the BioASQ Challenge, all the factoid and list types.
- Converted to the SQUAD dataset format
	- PASSAGES
		- length varies from a sentences to a paragraph.
	- CONTEXT: passage in a dataset that contains answers or clues for answers.
	- ANSWERS: 
		- exact answer
		- starting position

- Use various sources as PASSAGES:
	- snippets: lines of text with information in them. 
	- PubMed abstracts as
	- Multiple passages attached to a question were divided to form a QUESTION:PASSAGE pair. increasing the number of overall examples. (data augmentation)
- YES/NO:
	- The distribution is skewed
		- under-sample the training data to balance the yes / no
			- TODO: use other strategies (weighted loss, re-sampling)

- Strategies for developing the datasets:
	- Snippets as-is Strategy
		- 
	- Full Abstract Strategy
		- Title + abstract as a PASSAGE
	- Appended Snippet Strategy
		- Add N sentences BEFORE and AFTER the original snippet.

## 2.4 Post-Processing
- Passage pairs with multiple augmented answers were given a list of confidence values. The highest one was used as final answer
- List Answers were given from a threshold value
- If the list answer contained a number $N$, the top $N$ answers were returned
- incomplete answers were removed:
	- non-paired parenthesis
	- non-paired brackets
	- commas at the beginning and end of an answer 
# 3 Experimental Setup
## 3.1 Dataset
- factoid and list type questions contain the `exact` answer.
- yes/no:
	- only binary answers
- multiple passages are provided as corresponding passages.

- 3,722 question-context pairs were made from the 779 factoid questions in the BioASQ7b
- Missing answers:
	- factoid: 28.2% 
	- list: 5.6% 
- Exclude unanswerable questions

## 3.2 Training

- Start by training on SQUAD 1.1
	- factoid and list type 
- SQUAD 2.0
	- yes/no
- Hyperparameter Tuning
- list 
	- 0.42 threshold

# 4 Results & Discussion
- BioASQ 7b Leaderboard 
- Mean Reciprocal Rank (MRR)
- Mean average F-measure (F1)
- Factoid:
	- Strict Accuracy (`SAcc`)
	- Lenient Accuracy (`LA`)
- List:
	- Mean Average Precision
	- Mean Average Call
	- Mean Average F1
- Yes/No:
	- Macro Average F1

| Yes/No | Factoid | List |
| --- | --- | --- |
| 67.12 | 46.37 | 32.76 |
| 83.31 | 56.67 | 47.32 |
| 74.73 | 51.15 | 32.98 |
| 82.08 | 69.12 | 46.04 |
| 82.50 | 36.38 | 46.19 |


- Our system obtained a 20% to 60% performance improvement over the best system.

- Pre-training BioBERT on the specific tasks helps.

- The performance of our system is largely affected by how the data is pre-processed.
- The effectiveness of the pre-processing strategy varies depending on the type of question.

- factoid questions:
	- Appended Snippets and Full abstract
- Yes/NO:
	- As Is
- List:
	- As Is