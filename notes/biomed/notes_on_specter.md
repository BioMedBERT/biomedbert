# Notes on the SPECTER : Document-level Representation Learning using Citation-informed Transformers paper - May 31 2010 - suzanne.repellin@gmail.com
[Arxiv](https://arxiv.org/pdf/2004.07180.pdf)

**SPECTER**: Scientific Paper Embeddings using Citation-informed TransformERs

## 1 Introduction
While word embedding documents are widely used, whole-document embeddings are relatively underexplored.
Unsatisfying results when using “off-the-shelf” pretrained language models to protuce reprensentaiton for classification or recommandation of papers with only their title and abstract.
Specter incorporates inter-document context into the Transformer language models.
Goal: document representations effective for multiple tasks with no nees for fine-tuning the pre-traned model for the specific taks.
Model uses citations but does not require any citation information.
Example of tasks where the model outpower state-of-the-art : topic classification, citation prediction, and recommendation.

In addition, release of **Scidoc**: Collection of data sets and an evaluation suite for documentlevel embeddings in the scientific domain. Covers seven document-level tasks ranging from citation prediction, to document classification and recommendation.

## 2 Model
- Transformer model architecture as basis of encoding the input paper
  - Uses *SciBERT** from **BERT**: multiple-layers of Transformers to encode tokens in a sequence.
  - **Document representation**: SPECTER builds embedding from the concatenated title and abstract.
  - Incorporate citations into the SciBERT model as a signal of inter-document relatedness.
  
- Pretrain on corpus of citations, used as an inter-document relatedness signal, formulated as a triplet loss learning objective.
  - Loss function that trains the Transformer model to learn closer representations for papers when one cites the other, and more distant representations otherwise
  - Use of triplets : two papers of which one quotes the other, a third unrelated paper, the first twos should be closer to each other that to the third one
  - Selection of the negative (unrelated) examples: one random set, one "harder" set where their are two degrees of citation separation (cited by a paper cited by the paper)
  - **Inference** only needs abstract and title, not citations, so it is possible to infer recent papers not yet cited
  
- Embeddings that can be applied to downstream tasks in a “feature-based” fashion (no fine tuning)

## 3 SciDocs Evaluation Framework
- A new comprehensive evaluation framework to measure the effectiveness of scientific paper embedding, covering seven document-level tasks ranging from citation prediction, to document classification and recommendation. 
- Training data and associated dataset are released.
### 3.1 **Document classification**:
  - **MeSH classification**: Class documents according to their Medical Subject Headings. Dataset of 23K academic medical papers each assigned one of the 11 top-level disease classes.
  - **Paper Topic Classification**: Predefined topic categories of the Microsoft Academic Graph (MAG). The topics are organized in a hierarchy of 5 levels, where level 1 is the most general and level 5 is the most specific. Dataset of 25K papers split in 19 categories.
### 3.2 **Citation prediction**:
  - **Direct citations**: predict which papers are cited by a given query paper from a given set of candidate papers (5 cited, 25 uncited). 25k total papers and 1k query papers
  - **Co-citations**: predict a highly co-cited paper with a given paper (two papers often cited together).30k
### 3.3 **User activity**: Using logs of user sessions from a major academic search engine
  - **Co-views**: 30K papers + 1K random paper out of training set. (1 paper is submitted for first batch 5 cited and 25 uncited from second batch and rank them)
  - **Co-reads**: When user clicks view pdf on description page -> stronger interest. Approx. 30K papers.
 ### 3.4 **Recommandation**
Evaluate the ability of paper embeddings to boost performance in a production recommendation system = recommand a paper according to another paper the user consulted. User clickthrough data: 22K clickthrough events from a public scholarly search engine.

## 4 Experiments
- **Training Data**: 
  - Semantic Scholar corpus (Ammar et al., 2018) = 146K query papers (around 26.7M tokens) with their corresponding outgoing citations + 32K papers for validation
  - Per paper: 5 training triples comprised of a query, a positive, and a negative paper
  - 2 hard negatives and 3 easy negatives per paper
  -  684K training triples and 145K validation triples
- **Training and Implementation**:
  - NLP: AllenNLP
  - Initialization: SciBERT pretrained weight
  - Training: Adam optimizer following the suggested hyperparameters in Devlin et al.
  - Each training epoch takes approximately 1-2 days to complete on the full datase
- **Task-Specific Model Details**: TODO
- **Baseline Methods**:
  - Work at intersection of textual representation, citation mining, and graph learning
  - SIF a method for learning document representations by removing the first principal component of aggregated word-level embeddings which we pretrain on scientific text
  - SciBERT, a state-of-the-art pretrained Transformer LM for scientific text + Sent-BERT
  - Citeomatic, paper representation model for citation prediction
  - SGC, state-of-the-art graph-convolutional approach
  
## 5 Results
Substantial improvements across all tasks with average performance of 80.0 across all metrics on all tasks which is a 3.1 point absolute improvement over the next-best baseline
- Classifier performance when trained on SPECTER representations is better than when trained on any other baseline
- User activity prediction: improving over the best baseline (Citeomatic in this case) by 2.7 and 4.0 points
- Similar trends for the “citation” and “co-citation” tasks: model outperforming virtually all other baselines except for SGC (but SGC cannot be used in real-world setting to embed new papers that are not cited yet)
- **recommendation task**: SPECTER outperforms all other models on this task with nDCG of 53.9.

