# Notes on the BIOBERT Paper - May 29 2020 - fabrizio.milo@gmail.com
[Arxiv](https://arxiv.org/pdf/1901.08746.pdf)

## 1 Introduction
While ELMo and BERT have proven the effectiveness of contextualized word representations, they **cannot** obtain high performance  on biomedical corpora because they are pre-trained on only *general domain* corpora.

Adapting BERT for the biomedical domain could potentially benefit numerous biomedical NLP researcher.

## 2 Approach
- Initialized from bert-wikipedia-books-corpus
- Trained on PubMed abstracts + PMC full-text articles
- Trained and fine-tuned on:
	- Name Entity Recognition (NER)
	- Relationship Extractions (RE)
	- Question Answering (QA)
- Contributions
	- *first* domain specific BERT 
	- 23 days on eight NVIDIA v100 GPUs (TFlops? compute!)
	- Biomedical NER: 0.62
	- Biomedical RE: 2.80
	- Biomedical QA: score: 12.24
## 3 Materials and Methods
### 3.1 BERT
- Architecture: Same as BERT
- Previous Architectures:
	- NON Contextualized
		- Word2Vec
		- GloVe
	- Contextualized
		- ELMo: Bidirectional Language Model
		- CoVe: Machine Translation to embed Context into word
		- BERT: Use Masked Language, Pre-Train using bidirectional Transformers.
	- BERT
		- SOTA on most NLP benchmarks with min modifications

### 3.2 Pre-training BioBERT
- Corpus:
	- PubMed Abstracts: 4.5B words
	- PMC Full-text articles: 13.5B words
	- English Wikipedia: 2.5 B
	- Book Corpus: 0.8B
- Domain specific terms:
	- Proper Nouns: BRCA1, c.248T>C
	- Terms: transcriptional, antimicrobial 
- Tokenization:
	- WordPiece: Mitigate Out of Vocabulary Issue 
- Cased vocabulary results in better performances
- They did *NOT* fine tuned the WordPiece
	- They just added 

### 3.3 Fine-tuning BioBERT
- NER:
	- Previous: LSTMs, CRFs
	- BERT: BIO2 probabilities?
	- Evaluation Metrics:
		- Entity Level Precision
		- Recall
		- F1 Score
- RE:
	- Anonymized target named entities using predefined tags such as:
		- @GENES
		- @DISEASE$
		- "Serine at position 986 of @GENE$ may be an independent genetic predictor for angiographic @DISEASE$.
	- Evaluation Metrics:
		- Precision
		- Recall
		- F1 Score
- QA:
	- Use BioSquad (similar to SQuAD)
	- Token level probabilities for the start/end location of answer phrases are computed using a **single output layer**.
	- However we observed that about 30% (i.e. 3 out of 10) of the BioASQ factoid questions were unanswerable in an extractive QA setting as the exact answers **DID NOT** appear in the given passages. 
	- Like Wiese et al (2017) we excluded the samples with unanswerable questions from the training sets. 
	- Same Pre-Training Process, using SQUAD
		- Improves the performances of both BERT and BioBERT
	- Evaluation Metrics:
		- Strict Accuracy (SA?)
		- Lenient Accuracy (LA?)
		- Mean Reciprocal Rank (MRR)
# 4 Results
## 4.1 Datasets
### NER: Pre Processed versions of all the NER datasets provided by (BioSQUAD)
- exceptions:
	- 2010 i2b2/VA and JNLPBA use
		- https://github.com/spyysalo/standoff2conll
	- Species-800:
		- https://github.com/spyysalo/s800
- BC2GM dataset:
	- no alternate annotations (?)

| Dataset | Entity Type | Number of annotations |
| --- | --- | --- |
| NCBI Disease | Disease | 6,881 |
| 2010 ib2b2/VA | Disease | 19,665 |
| BC5CDR | Disease | 12,694 |
| BC5CDR | Drug/Chem | 15,411 |
| BC4CHEMD | Drug/Chem | 79,842 |
| BC2GM | Gene/Protein | 20,703 |
| JNLPBA | Gene/Protein | 35,460 |
| LINNAEUS | Species | 4,077 |
| Species-800 | Species | 3,708 |


### Relation Extractions: Pre-Processing

- gene-disease relations and protein chemical relations
- Preprocessed GAD and EU-ADR datasets
- CHEMPROT: Lim and Kang 2018
- BioASQ factoid dataset, converted to BioASQ
- PMIDs: full abstracts and QA as in the challenge
	- BioASQ pre-processed fully available
- SAME DATASET SPLIT
- 10 fold cross-validation for 
	- GAD
	- EU-ADR

| Dataset | Entity Type | Number of relations |
| --- | --- | --- |
| GAD | Gene-Disease | 5330 |
| EU-ADR | Gene-Disease | 355 |
| CHEMPROT | Protein-Chemical | 10,031 |


### Question Answer 

| Dataset | Number of Train | Number of Test |
| --- | --- | --- |
| BioASQ 4b-factoid | 327 | 161 |
| BioASQ 5b-factoid | 486 | 150 |
| BioASQ 6b-factoid | 618 | 161 |

- Other Datasets (!):
	- MedMentions: A Large Biomedical Corpus Annotated with UMLS Concepts
	- They give excuses on why they didn't use this one.

## 4.2 Experimental Setups
- Bert Base Wikipedia+BookCorpus 1M Steps
- BioBERT (+ PubMed + PMC) trained for **470K** steps
- Pre training steps:
	- 200K: PubMed (? define optimal)
	- 270K: PMC (? same)
- learning rate scheduling and Batch size are the same (? this smells as 8 V100 shouldn't be able to handle the same batch size of the TPUv3-128)
- Hyper Parameters:
	- Max sequence length: 512
	- Mini Batch Size: 192
	- 98, 304 words per iteration ( i.e 512 * 192 ) 
	- 23 days to train. (lol)
- They used **ONLY BERT BASE** ($$$)
- Fine tuning on batch size of 10, 16, 32 or 64
- learning rate: 5e-5, 3e-5 or 1e-5
- QA & RE are fast (~1h on a Titan X). NER takes more time

## 4.3 Experimental Results
- TODO: write table

## 5 Discussion
- Save the steps of the training at different steps
- 1B Words is quite effective up to 4.5B words (I guess that is the capacity of BERT BASE, I wonder with BERT LARGE how this changes)
- The results *clearly* show that the performance of each dataset improves as the number of pre-training steps increases.
- F1 scores for NER & RE
- MRR for QA
- TODO: write table