"""Code to train and eval a BERT passage re-ranker on the TREC CAR dataset."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

# local modules
import metrics
import modeling
import optimization

flags = tf.flags

FLAGS = flags.FLAGS


## Required parameters
flags.DEFINE_string(
    "data_dir", 
    "./data/tfrecord/",
    "The input data dir. Should contain the .tfrecord files and the supporting "
    "query-docids mapping files.")

flags.DEFINE_string(
    "bert_config_file",
    "./data/bert/pretrained_models/uncased_L-24_H-1024_A-16/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "output_dir", "./data/output",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_boolean(
    "trec_output", True,
    "Whether to write the predictions to a TREC-formatted 'run' file..")

flags.DEFINE_string(
    "init_checkpoint",
    "/path_to_bert_pretrained_on_treccar/model.ckpt-1000000",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 3e-6, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 400000,
                     "Total number of training steps to perform.")

flags.DEFINE_integer(
    "max_dev_examples", None,
    "Maximum number of dev examples to be evaluated. If None, evaluate all "
    "examples in the dev set.")

flags.DEFINE_integer("num_dev_docs", 10,
                     "Number of docs per query in the dev files.")

flags.DEFINE_integer(
    "max_test_examples", None,
    "Maximum number of test examples to be evaluated. If None, evaluate all "
    "examples in the test set.")

flags.DEFINE_integer("num_test_docs", 1000,
                     "Number of docs per query in the dev files.")

flags.DEFINE_integer(
    "num_warmup_steps", 40000,
    "Number of training steps to perform linear learning rate warmup.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


METRICS_MAP = ['MAP', 'RPrec', 'MRR', 'NDCG', 'MRR@10']
FAKE_DOC_ID = "00000000"  # Fake doc id used to fill queries with less than num_eval_docs.


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  output_layer = model.get_pooled_output()
  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, log_probs)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    len_gt_titles = features["len_gt_titles"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    (total_loss, per_example_loss, log_probs) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    scaffold_fn = None
    initialized_variable_names = []
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:
        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.PREDICT:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={
              "log_probs": log_probs,
              "label_ids": label_ids,
              "len_gt_titles": len_gt_titles,
          },
          scaffold_fn=scaffold_fn)

    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def input_fn_builder(dataset_path, seq_length, is_training,
                     max_eval_examples=None):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""

    batch_size = params["batch_size"]
    output_buffer_size = batch_size * 1000

    def extract_fn(data_record):
      features = {
          "query_ids": tf.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
          "doc_ids": tf.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
          "label": tf.FixedLenFeature([], tf.int64),
          "len_gt_titles": tf.FixedLenFeature([], tf.int64),
      }
      sample = tf.parse_single_example(data_record, features)

      query_ids = tf.cast(sample["query_ids"], tf.int32)
      doc_ids = tf.cast(sample["doc_ids"], tf.int32)
      label_ids = tf.cast(sample["label"], tf.int32)
      #if "len_gt_titles" in sample:
      len_gt_titles = tf.cast(sample["len_gt_titles"], tf.int32)
      #else:
      #  len_gt_titles = tf.constant(-1, shape=[1], dtype=tf.int32)
      input_ids = tf.concat((query_ids, doc_ids), 0)

      query_segment_id = tf.zeros_like(query_ids)
      doc_segment_id = tf.ones_like(doc_ids)
      segment_ids = tf.concat((query_segment_id, doc_segment_id), 0)

      input_mask = tf.ones_like(input_ids)

      features = {
          "input_ids": input_ids,
          "segment_ids": segment_ids,
          "input_mask": input_mask,
          "label_ids": label_ids,
          "len_gt_titles": len_gt_titles,
      }
      return features

    dataset = tf.data.TFRecordDataset([dataset_path])
    dataset = dataset.map(
        extract_fn, num_parallel_calls=4).prefetch(output_buffer_size)

    if is_training:
      dataset = dataset.repeat()
      dataset = dataset.shuffle(buffer_size=1000)
    else:
      if max_eval_examples:
        dataset = dataset.take(max_eval_examples)

    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes={
            "input_ids": [seq_length],
            "segment_ids": [seq_length],
            "input_mask": [seq_length],
            "label_ids": [],
            "len_gt_titles": [],
        },
        padding_values={
            "input_ids": 0,
            "segment_ids": 0,
            "input_mask": 0,
            "label_ids": 0,
            "len_gt_titles": 0,
        },
        drop_remainder=True)

    return dataset
  return input_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  
  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `FLAGS.do_train` or `FLAGS.do_eval` must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tpu_cluster_resolver = None
  if FLAGS.use_tpu:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        TPU_ADDRESS)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=2,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.eval_batch_size)

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", FLAGS.num_train_steps)
    train_input_fn = input_fn_builder(
        dataset_path=os.path.join(FLAGS.data_dir, "dataset_train.tf"),
        seq_length=FLAGS.max_seq_length,
        is_training=True)
    estimator.train(input_fn=train_input_fn,
                    max_steps=FLAGS.num_train_steps)
    tf.logging.info("Done Training!")

  if FLAGS.do_eval:
    for set_name in ["dev", "test"]:
      tf.logging.info("***** Running evaluation on the {} set*****".format(
          set_name))
      tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
      
      max_eval_examples = None
      
      if set_name == "dev":
        num_eval_docs = FLAGS.num_dev_docs
        if FLAGS.max_dev_examples:
          max_eval_examples = FLAGS.max_dev_examples * FLAGS.num_dev_docs
          
      elif set_name == "test":
        num_eval_docs = FLAGS.num_test_docs
        if FLAGS.max_test_examples:
          max_eval_examples = FLAGS.max_test_examples * FLAGS.num_test_docs
          
      eval_input_fn = input_fn_builder(
          dataset_path=os.path.join(FLAGS.data_dir, "dataset_" + set_name + ".tf"),
          seq_length=FLAGS.max_seq_length,
          is_training=False,
          max_eval_examples=max_eval_examples)

      if FLAGS.trec_output:
        trec_file = tf.gfile.Open(
            os.path.join(
                FLAGS.output_dir, "bert_predictions_" + set_name + ".run"), "w")
        query_docids_map = []
        docs_per_query = 0  # Counter of docs per query
        with tf.gfile.Open(
            os.path.join(FLAGS.data_dir, set_name + ".run")) as ref_file:
          for line in ref_file:
            query, _, doc_id, _, _, _ = line.strip().split(" ")
            
            # We add fake docs so the number of docs per query is always
            # num_eval_docs.
            if len(query_docids_map) > 0:
              if query != last_query:
                if docs_per_query < num_eval_docs:
                  fake_pairs = (num_eval_docs - docs_per_query) * [
                      (last_query, FAKE_DOC_ID)]
                  query_docids_map.extend(fake_pairs)
                docs_per_query = 0

            query_docids_map.append((query, doc_id))
            last_query = query
            docs_per_query += 1
            
      # ***IMPORTANT NOTE***
      # The logging output produced by the feed queues during evaluation is very 
      # large (~14M lines for the dev set), which causes the tab to crash if you
      # don't have enough memory on your local machine. We suppress this 
      # frequent logging by setting the verbosity to WARN during the evaluation
      # phase.
      tf.logging.set_verbosity(tf.logging.WARN)
      
      result = estimator.predict(input_fn=eval_input_fn,
                                 yield_single_examples=True)
      start_time = time.time()
      results = []
      all_metrics = np.zeros(len(METRICS_MAP))
      example_idx = 0
      total_count = 0
      for item in result:
        results.append(
            (item["log_probs"], item["label_ids"], item["len_gt_titles"]))

        if len(results) == num_eval_docs:

          log_probs, labels, len_gt_titles = zip(*results)
          log_probs = np.stack(log_probs).reshape(-1, 2)
          labels = np.stack(labels)
          len_gt_titles = np.stack(len_gt_titles)
          assert len(set(list(len_gt_titles))) == 1, (
              "all ground truth lengths must be the same for a given query.")   
          
          scores = log_probs[:, 1]
          pred_docs = scores.argsort()[::-1]
          
          gt = set(list(np.where(labels > 0)[0]))

          # Metrics like NDCG and MAP require the total number of relevant docs.
          # The code below adds missing number of relevant docs to gt so the 
          # metrics are the same as if we had used all ground-truths.
          # The extra_gts have all negative ids so they don't interfere with the
          # predicted ids, which are all equal or greater than zero.
          extra_gts = list(-(np.arange(max(0, len_gt_titles[0] - len(gt))) + 1))
          gt.update(extra_gts)
          
          all_metrics += metrics.metrics(
              gt=gt, pred=pred_docs, metrics_map=METRICS_MAP)

          if FLAGS.trec_output:
            start_idx = example_idx * num_eval_docs
            end_idx = (example_idx + 1) * num_eval_docs
            queries, doc_ids = zip(*query_docids_map[start_idx:end_idx])
            assert len(set(queries)) == 1, "Queries must be all the same."
            query = queries[0]
            rank = 1
            for doc_idx in pred_docs:
              doc_id = doc_ids[doc_idx]
              score = scores[doc_idx]
              # Skip fake docs, as they are only used to ensure that each query
              # has 1000 docs.
              if doc_id != FAKE_DOC_ID:
                output_line = " ".join(
                    (query, "Q0", doc_id, str(rank), str(score), "BERT"))
                trec_file.write(output_line + "\n")
                rank += 1

          example_idx += 1
          results = []
          
        total_count += 1
        
        if total_count % 10000 == 0:
          tf.logging.warn("Read {} examples in {} secs. Metrics so far:".format(
              total_count, int(time.time() - start_time)))
          tf.logging.warn("  ".join(METRICS_MAP))
          tf.logging.warn(all_metrics / example_idx)
          
      # Once the feed queues are finished, we can set the verbosity back to 
      # INFO.
      tf.logging.set_verbosity(tf.logging.INFO)
      
      if FLAGS.trec_output:
        trec_file.close()

      all_metrics /= example_idx

      tf.logging.info("Eval {}:".format(set_name))
      tf.logging.info("  ".join(METRICS_MAP))
      tf.logging.info(all_metrics)


if __name__ == "__main__":
  tf.app.run()

