
# Notes on the Report:SQuAD to BioASQ: analysis of general to specific

[Stanford Report PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/default/15759479.pdf)

## Abstract 

As biomedical information in the form of publications and electronic health records
(EHR) increases at an increasingly fast pace, there is clear utility in having systems that can automatically handle information extraction, summarization, and
question answering tasks. While there have been significant strides in improving
language tasks for general language, addressing domain-specific contexts still
remains challenging. In this project, I apply and fine-tune models to the SQuAD
dataset and further modify/adapt for biomedical domain-specific question answering. I evaluated and compared performance on the SQuAD dataset and BioASQ,
a biomedical literature QA dataset, with the goal of analyzing and developing
approaches to leverage unsupervised language models for domain-specific applications. Upon generating various fine-tuned models, the best performance for general
language SQuAD QA achieved an F1 score of 76.717, EM score of 73.379, and for
biomedical-specific BioASQ QA achieved an F1 score of 70.348 and EM score of
49.902.


# Notes:

## Fabrizio's
Another experiment (run on Azure) that confirms that fine tuning on the Task data helps improving the results. Interesting how the author categorizes this models as Pre-TRained Contextual Embedding (PCE).

Another note on the experiment is the increase of performance in using a bigger window for answering and overall sequence length.

They don't do unsupervised re-train at the domain level. They don't re-use the SQUAD trained model to further turn their BioASQ task.