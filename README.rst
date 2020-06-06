===============================
BioMedBERT
===============================

.. image:: https://badge.fury.io/py/biomedbert.png
    :target: http://badge.fury.io/py/biomedbert

.. image:: https://travis-ci.org/dvdbisong/biomedbert.png?branch=master
        :target: https://travis-ci.org/dvdbisong/biomedbert

.. image:: https://pypip.in/d/biomedbert/badge.png
        :target: https://crate.io/packages/biomedbert?version=latest


A Natural Language Processing and Understanding (NLP/NLU) tool for biomedical research

Table of Contents
-----------------
.. contents:: :depth: 2

Usage
-----

:Clone repo: ``git clone https://github.com/aivscovid19/covid-19_research_collaboration.git``
:Install BioMedBERT:
   ``pip install .``

Train the BioMedBERT Model
--------------------------
Syntax for training BioMedBERT-Large model:
  ``biomedbert code train model <model_type> <model_dir> <pretrain_dir> <bucket_name> <tpu_name> <train_steps> <train_bs> <eval_bs> <tpu_cores>``

Train BioMedBERT-Large from BERT weights:
  ``biomedbert code train model large biomedbert_large_breathe_wikipedia_bert_weights_vocab pre_trained_biomed_wikipedia_data_bert_vocab ekaba-assets biomedbert 100000 128b``

Models repository
-----------------
BioMedBERT-Large from BERT weights:
  ``gs://ekaba-assets/biomedbert_base_bert_weights_and_vocab``

BioMedBERT-Large from scratch:
  ``gs://ekaba-assets/biomedbert_base_scratch_breathe_bert_vocab``

On TPUs v3-128
--------------
First set the GCP zone to `europe-west4-a`
  ``biomedbert gcp project set ai-vs-covid19 europe-west4-a``

Syntax:
  Create TPU:
    ``biomedbert gcp vm create tpu <vm-instance> [preemptible]``
  Delete TPU:
    ``biomedbert gcp vm delete tpu <vm-instance>``

Create TPUs:
  ``biomedbert gcp vm create tpu biomedbert``
Create Preemptible TPUs:
  ``biomedbert gcp vm create tpu biomedbert-preempt preemptible``

Delete TPUs:
  ``biomedbert gcp vm delete tpu <vm-instance>``


Fine-tune and Evaluate SQuAD
----------------------------
Fine-tune SQuAD Syntax:
  ``biomedbert squad finetune (v1|v2) <model_type> <bucket_name> <model_dir> <train_file> <predict_file> <tpu_name> <tpu_cores>``
Evaluate SQuAD Syntax:
  ``biomedbert squad evaluate (v1|v2) <bucket_name> <model_dir> <evaluate_file> <predict_file>``

Finetune SQuAD
^^^^^^^^^^^^^^^
with BERT weights:
""""""""""""""""""
v1:
  ``biomedbert squad finetune v1 large ekaba-assets biomedbert_base_bert_weights_and_vocab train-v1.1.json dev-v1.1.json biomedbert 128``
v2:
  ``biomedbert squad finetune v2 large ekaba-assets biomedbert_base_bert_weights_and_vocab train-v2.0.json dev-v2.0.json biomedbert 128``


From Scratch:
"""""""""""""
v1:
  ``biomedbert squad finetune v1 large ekaba-assets biomedbert_base_scratch_breathe_bert_vocab train-v1.1.json dev-v1.1.json biomedbert-preempt 128``
v2:
  ``biomedbert squad finetune v2 large ekaba-assets biomedbert_base_scratch_breathe_bert_vocab train-v2.0.json dev-v2.0.json biomedbert-preempt 128``

Evaluate SQuAD
^^^^^^^^^^^^^^
with BERT weights:
""""""""""""""""""
v1:
  ``biomedbert squad evaluate v1 ekaba-assets biomedbert_base_bert_weights_and_vocab evaluate-v1.1.py dev-v1.1.json predictions.json``
v2:
  ``biomedbert squad evaluate v2 ekaba-assets biomedbert_base_bert_weights_and_vocab evaluate-v2.0.py dev-v2.0.json``

From Scratch:
"""""""""""""
v1:
  ``biomedbert squad evaluate v1 ekaba-assets biomedbert_base_scratch_breathe_bert_vocab evaluate-v1.1.py dev-v1.1.json``
v2:
  ``biomedbert squad evaluate v2 ekaba-assets biomedbert_base_scratch_breathe_bert_vocab evaluate-v2.0.py dev-v2.0.json``


Fine-tune and Evaluate BioASQ
-----------------------------
Fine-tune BioASQ Syntax:
  ``biomedbert bioasq finetune <model_type> <train_file> <predict_file> <bucket_name> <model_dir> <squad_folder> [<tpu_name> <tpu_cores>]``
Evaluate BioASQ Syntax:
  ``biomedbert bioasq evaluate <bucket_name> <model_dir> <train_file> <eval_file> <squad_folder>``

Finetune BioASQ
^^^^^^^^^^^^^^^
Change the ``<train_file>`` (BioASQ-train-factoid-4b.json)  and ``<predict_file>`` (BioASQ-test-factoid-4b-1.json) accordingly.

with BERT weights:
""""""""""""""""""
From SQuAD v1:
  ``biomedbert bioasq finetune large BioASQ-train-factoid-4b.json BioASQ-test-factoid-4b-1.json ekaba-assets biomedbert_base_bert_weights_and_vocab squad_v1 biomebert 128``
From SQuAD v2:
  ``biomedbert bioasq finetune large BioASQ-train-factoid-4b.json BioASQ-test-factoid-4b-1.json ekaba-assets biomedbert_base_bert_weights_and_vocab squad_v2 biomedbert-preempt 128``


From Scratch:
"""""""""""""
From SQuAD v1:
  ``biomedbert bioasq finetune large BioASQ-train-factoid-4b.json BioASQ-test-factoid-4b-1.json ekaba-assets biomedbert_base_scratch_breathe_bert_vocab squad_v1 biomebert 128``
From SQuAD v2:
  ``biomedbert bioasq finetune large BioASQ-train-factoid-4b.json BioASQ-test-factoid-4b-1.json ekaba-assets biomedbert_base_scratch_breathe_bert_vocab squad_v2 biomedbert-preempt 128``

Evaluate BioASQ
^^^^^^^^^^^^^^^
with BERT weights:
""""""""""""""""""
From SQuAD v1:
  ``biomedbert bioasq evaluate ekaba-assets biomedbert_base_bert_weights_and_vocab BioASQ-train-factoid-4b.json 4B1_golden.json squad_v1``
From SQuAD v2:
  ``biomedbert bioasq evaluate ekaba-assets biomedbert_base_bert_weights_and_vocab BioASQ-train-factoid-4b.json 4B1_golden.json squad_v2``

From Scratch:
"""""""""""""
From SQuAD v1:
  ``biomedbert bioasq evaluate ekaba-assets biomedbert_base_scratch_breathe_bert_vocab BioASQ-train-factoid-4b.json 4B1_golden.json squad_v1``
From SQuAD v2:
  ``biomedbert bioasq evaluate ekaba-assets biomedbert_base_scratch_breathe_bert_vocab BioASQ-train-factoid-4b.json 4B1_golden.json squad_v2``

Fine-tune and Evaluate RE
--------------------------
Fine-tune RE Syntax:
  ``biomedbert re finetune <model_type> <re_dataset> <re_dataset_no> <model_dir> <bucket_name> <tpu_name> <tpu_cores>``
Evaluate RE Syntax:
  ``biomedbert re evaluate <re_dataset> <re_dataset_no> <model_dir> <bucket_name>``

Finetune RE
^^^^^^^^^^^^
with BERT weights:
""""""""""""""""""
GAD 1:
  ``biomedbert re finetune large GAD 1 biomedbert_base_bert_weights_and_vocab ekaba-assets biomedbert-preempt 128``
EU-ADR 1:
  ``biomedbert re finetune large euadr 1 biomedbert_base_bert_weights_and_vocab ekaba-assets biomedbert-preempt 128``


From Scratch:
"""""""""""""
GAD 1:
  ``biomedbert re finetune large GAD 1 biomedbert_base_scratch_breathe_bert_vocab ekaba-assets biomedbert 128``
EU-ADR 1:
  ``biomedbert re finetune large euadr 1 biomedbert_base_scratch_breathe_bert_vocab ekaba-assets biomedbert 128``

Evaluate RE
^^^^^^^^^^^^
with BERT weights:
""""""""""""""""""
GAD 1:
  ``biomedbert re evaluate GAD 1 biomedbert_base_bert_weights_and_vocab ekaba-assets``
EU-ADR 1:
  ``biomedbert re evaluate euadr 1 biomedbert_base_bert_weights_and_vocab ekaba-assets``

From Scratch:
"""""""""""""
GAD 1:
  ``biomedbert re evaluate GAD 1 biomedbert_base_scratch_breathe_bert_vocab ekaba-assets``
EU-ADR 1:
  ``biomedbert re evaluate euadr 1 biomedbert_base_scratch_breathe_bert_vocab ekaba-assets``


Requirements
------------

- Python >= 2.6 or >= 3.3

License
-------

MIT licensed. See the bundled `LICENSE <https://github.com/aivscovid19/covid-19_research_collaboration/blob/master/LICENSE>`_ file for more details.
