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
- **Document classification**: 
