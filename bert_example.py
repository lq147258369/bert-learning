import tensorflow as tf
import pandas as pd
import numpy as np
import codecs, os, random, datetime
import sys, pickle
import keras

from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import json
import base64
from bert4keras.models import build_transformer_model


def parse_model_output_path():
    return os.environ['ENV_MODEL_OUTPUT']


def check_is_chief():
    if 'TF_CONFIG' not in os.environ:
        print("Run in single machine mode...")
        return True

    tf_config_obj = json.loads(os.environ.get("TF_CONFIG"))
    task_type = tf_config_obj["task"]["type"]
    return task_type == "chief"


def create_cls_model(num_labels):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for layer in bert_model.layers:
        layer.trainable = True

    x_token_in = Input(shape=(None,))
    x_segment_in = Input(shape=(None,))
    tf.keras.layers.Lambda(lambda x: x[:, 0], name='CLS-token1')(x1_emb)
    keras.layers.Lambda(lambda x: x[:, 0], name='CLS-token1')(x1_emb)
    x = bert_model([x_token_in, x_segment_in])      #[batch,128,768]
    pool_layer_ave = tf.keras.layers.GlobalAvgPool1D(name='pooling_ave')(x)
    p = tf.keras.layers.Dense(num_labels, activation='softmax')(pool_layer_ave)  # 二分类

    model = Model([x_token_in, x_segment_in], p)
    model.compile(
        optimizer=Adam(1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.SparseCategoricalCrossentropy()]
    )
    return model


def parse_tfrecord_inputs():
    if 'HIVE_INPUT_TBL_PART_INFO' not in os.environ:
        raise Exception("Can't find env.HIVE_INPUT_TBL_PART_INFO, maybe set -tbl and -pt in barathum.")
    raw_hive_input_tbl_part_info = os.environ["HIVE_INPUT_TBL_PART_INFO"]
    print("RAW_HIVE_INPUT_TBL_PART_INFO: ", raw_hive_input_tbl_part_info)
    parsed_hive_input_tbl_part_info = base64.b64decode(raw_hive_input_tbl_part_info)
    print("parsed_hive_input_tbl_part_info: ", parsed_hive_input_tbl_part_info)
    hive_input_tbl_part_info = json.loads(parsed_hive_input_tbl_part_info)
    input_train_files = []
    train_total_count = 0
    input_test_files = []
    test_total_count = 0
    for env_input_tbl_part in hive_input_tbl_part_info:
        location = env_input_tbl_part['location']
        part_date, part_split = env_input_tbl_part['parts']
        num_rows = env_input_tbl_part['numRows']
        if part_split == "train":
            input_train_files.append(location + "/*")
            train_total_count += num_rows
        elif part_split == "test":
            input_test_files.append(location + "/*")
            test_total_count += num_rows
        else:
            raise Exception("Unknown split type")

    dataset_train_filenames = []
    for filename in input_train_files:
        dataset_train_filenames += tf.data.Dataset.list_files(filename)

    dataset_test_filenames = []
    for filename in input_test_files:
        dataset_test_filenames += tf.data.Dataset.list_files(filename)

    return dataset_train_filenames, train_total_count, dataset_test_filenames, test_total_count


def preprocess_dataset(raw_dataset, tokenizer, label2id):
    def seq_padding(l, padding=0, maxlen=128):
        return np.array(
            np.concatenate([l,
                            [padding] * (maxlen - len(l))
                            ]) if len(l) < maxlen else l
        )

    def map_encode(content, module):
        text_decoded = content.numpy().decode()[:maxlen - 2]
        label_decoded = module.numpy().decode()
        x_t, x_s = tokenizer.encode(first=text_decoded)
        return seq_padding(x_t), seq_padding(x_s), label2id[label_decoded]

    processed = raw_dataset.unbatch() \
        .map(
        lambda x: tf.py_function(func=map_encode, inp=[x['content'], x['module']], Tout=[tf.int32, tf.int32, tf.int32])) \
        .map(lambda x_t, x_s, label: ((x_t, x_s), label))
    return processed


# 添加一个callback，在on_epoch_end的时候计算f1
from sklearn.metrics import f1_score, recall_score, precision_score


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return


if __name__ == "__main__":

    print("开始任务")

    is_chief = check_is_chief()

    print("开始创建策略")
    print("is_chief: ", is_chief)

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    print("开始下载bert权重")
    os.system(
        "hdfs dfs -get -p hdfs://pdd-data-ns/apps/nothive/warehouse/merchant/home_page_card_fg/sms_clf/chinese_L-12_H-768_A-12")
    print("下载bert权重完成")

    maxlen = 128
    BATCH_SIZE = 64
    config_path = './chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = './chinese_L-12_H-768_A-12/vocab.txt'
    # label2id_file = './OMS/label2id.pkl'

    # 读取hive表数据
    dataset_train_filenames, train_total_count, dataset_test_filenames, test_total_count = parse_tfrecord_inputs()
    print("train_total_count: ", train_total_count)
    print("test_total_count: ", test_total_count)

    column_specs = [
        tf.TensorSpec(shape=[None], dtype=tf.string, name="content"),
        tf.TensorSpec(shape=[None], dtype=tf.string, name="module")
    ]

    # 读取原始训练数据
    print("num_replicas_in_sync", strategy.num_replicas_in_sync)
    print("BATCH_SIZE", BATCH_SIZE)
    train_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync
    test_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync
    raw_dataset_train = tf.data.DiOrcDataset(dataset_train_filenames, train_batch_size, column_specs)
    raw_dataset_test = tf.data.DiOrcDataset(dataset_test_filenames, test_batch_size, column_specs)

    # 创建 tockenizer
    with strategy.scope():
        token_dict = {}
        with codecs.open(dict_path, 'r', 'utf-8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        tokenizer = Tokenizer(token_dict)
        # 读取 label 映射关系
        # with open(label2id_file,'rb') as fb:
        # label2id = pickle.load(fb)
    label2id = {'T': 1, 'F': 0}
    # 数据预处理
    train_data = preprocess_dataset(raw_dataset_train, tokenizer, label2id).shuffle(train_batch_size).batch(
        train_batch_size).prefetch(1)
    test_data = preprocess_dataset(raw_dataset_test, tokenizer, label2id).take(10000).batch(test_batch_size).prefetch(1)
    # 自动分区
    # options = tf.data.Options()
    # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    # train_data = train_data.with_options(options)
    # test_data = test_data.with_options(options)

    with strategy.scope():
        model = create_cls_model(len(label2id))
    # metrics = Metrics(test_data)
    for i in range(1):
        print("开始%d轮训练" % i)
        model.fit(
            train_data,
            epochs=1,
            validation_data=test_data
        )
        print("%d轮训练结束" % i)
        if is_chief:
            save_path_final = "hdfs://yiran-data-ns/user/xianqiushi/share/udf/jiyun/model/epoch_%d" % i
        else:
            save_path_final = '/tmp/model/barathum_keras_model_%d' % i
        print("Start export model: " + save_path_final)
        model.save(save_path_final, save_format='tf')

    if is_chief:
        save_path_final = parse_model_output_path()
    else:
        save_path_final = '/tmp/model/' + 'barathum_keras_model'
    print("Start export model: " + save_path_final)
    model.save(save_path_final, save_format='tf')
