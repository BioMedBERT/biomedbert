# Don't Stop Pretraining: Adapt Language Models to Domains and Tasks


 The paper proposes various conceptually simple yet effective tecniques to leverages pre-trained language models on general domain to achieve state of the art results for domain specific tasks.

 Two Main Strategies:
 - Domain Adaptation: Which consist of simply fine-tune on unlabeled text on the specific domain dataset
 - Task Adaptation: which further refine the language model with unlabled data on the task specific text before passing it to training the specific labeled task


 # Results
 - Domain adaptation + Task Adaptation gives the best results on RoBERTA

# Insights (Fabrizio):
I think this is where we could focus to get the most immediate results. 
- 1. Train the base english corpus using RoBERTa's english corpus base.
    - 1.1 run standard SQUAD tests using BERT on Wiki+Book vs the new dataset
- 2. Apply domain adaptation from 1. on our refined corpus. 
    - 2.1 we need to understand a bit better the vocabulary effect. Can we domain adapt a vocabulary? How can we "augment" an existing embedding space with new words?
- 3. Apply Task adaptation on BioASQ
    - 3.1 try to beat the baseline of BERT
