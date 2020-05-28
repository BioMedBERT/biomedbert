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
  ``biomedbert gcp vm create tpu <vm-instance> [preemptible]``
  ``biomedbert gcp vm delete tpu <vm-instance>``

Creating TPUs:
  ``biomedbert gcp vm create tpu biomedbert``
  ``biomedbert gcp vm create tpu biomedbert-preempt``

Deleting TPUs:


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
  biomedbert squad evaluate v1 ekaba-assets biomedbert_base_bert_weights_and_vocab evaluate-v1.1.py dev-v1.1.json predictions.json``
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
Change the ``<train_file>`` and ``<predict_file>`` accordingly:

- 4b: BioASQ-train-factoid-4b.json
      BioASQ-test-factoid-4b-1.json
- 5b: BioASQ-train-factoid-5b.json
- 6b: BioASQ-train-factoid-6b.json

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
v1:
  biomedbert squad evaluate v1 ekaba-assets biomedbert_base_bert_weights_and_vocab evaluate-v1.1.py dev-v1.1.json predictions.json``
v2:
  ``biomedbert squad evaluate v2 ekaba-assets biomedbert_base_bert_weights_and_vocab evaluate-v2.0.py dev-v2.0.json``

From Scratch:
"""""""""""""
v1:
  ``biomedbert squad evaluate v1 ekaba-assets biomedbert_base_scratch_breathe_bert_vocab evaluate-v1.1.py dev-v1.1.json``
v2:
  ``biomedbert squad evaluate v2 ekaba-assets biomedbert_base_scratch_breathe_bert_vocab evaluate-v2.0.py dev-v2.0.json``


* biomedbert glue finetune MRPC gs://ekaba-assets/biomedbert_base_bert_weights_and_vocab model.ckpt-68000 vocab.txt biomedbert-tpu

Requirements
------------

- Python >= 2.6 or >= 3.3

License
-------

MIT licensed. See the bundled `LICENSE <https://github.com/aivscovid19/covid-19_research_collaboration/blob/master/LICENSE>`_ file for more details.
