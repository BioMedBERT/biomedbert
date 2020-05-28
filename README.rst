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

Usage
-----

:Clone repo: ``git clone https://github.com/aivscovid19/covid-19_research_collaboration.git``
:Install BioMedBERT:
   ``pip install .``

Models repository
-----------------
BioMedBERT-Large from BERT weights:
  ```gs://ekaba-assets/biomedbert_base_bert_weights_and_vocab```

BioMedBERT-Large from scratch:
  ```gs://ekaba-assets/biomedbert_base_scratch_breathe_bert_vocab```

Fine-tune SQuAD
-----------------
Syntax:
  ```biomedbert squad finetune (v1|v2) <model_type> <bucket_name> <model_dir> <train_file> <predict_file> <tpu_name> <tpu_cores>```

how
  The term is a one-line phrase, and the
  definition is one or more paragraphs or
  body elements, indented relative to the
  term. Blank lines are not allowed
  between term and definition.



* biomedbert glue finetune MRPC gs://ekaba-assets/biomedbert_base_bert_weights_and_vocab model.ckpt-68000 vocab.txt biomedbert-tpu

Requirements
------------

- Python >= 2.6 or >= 3.3

License
-------

MIT licensed. See the bundled `LICENSE <https://github.com/aivscovid19/covid-19_research_collaboration/blob/master/LICENSE>`_ file for more details.
