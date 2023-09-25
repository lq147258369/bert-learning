import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
    """

    Args:
        input_ids: 输入字token id.int32 Tensor of shape [batch_size, seq_length] containing word ids.
        vocab_size: 词表大小
        embedding_size: embedding维度
        initializer_range: embedding 值范围
        word_embedding_name:
        use_one_hot_embeddings:

    Returns:

    """
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])  # [batch_size, seq_length, 1]
    embedding_table = tf.Variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initial_value=tf.random.truncated_normal(shape=(vocab_size, embedding_size), stddev=initializer_range)
    )  # [vocab_size, embedding_size]
    flat_input_ids = tf.reshape(input_ids, shape=[-1])  # [batch_size * seq_length]
    if use_one_hot_embeddings:
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)  # [sequence_length, vocab_size]
        out_put = tf.matmul(one_hot_input_ids, embedding_table)  # [batch_size * seq_length,embedding_size]
    else:
        out_put = tf.gather(embedding_table, flat_input_ids)  # [batch_size * seq_length,embedding_size]
    input_shape = get_shape_list(input_ids)

    # [batch_size , seq_length, embedding_size]
    out_put = tf.reshape(out_put, shape=input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return out_put, embedding_table


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1
                            ):
    """Performs various post-processing on a word embedding tensor.
        token embedding+segment embedding+position embedding
        在 Transformer论文中的position_embeddings是 由 sin/cos函数生成的固定的值，而在这里代码实现中是跟普
        通word embedding一样随机生成的，可以训练的。作者这里这样选择的原因可能是BERT
        训练的数据比Transformer那篇大很多，完全可以让模型自己去学习
      Args:
        input_tensor: float Tensor of shape [batch_size, seq_length,
          embedding_size].
        use_token_type: bool. Whether to add embeddings for `token_type_ids`. 论文中的segment embedding
        token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
          Must be specified if `use_token_type` is True.
        token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
        token_type_embedding_name: string. The name of the embedding table variable
          for token type ids.
        use_position_embeddings: bool. Whether to add position embeddings for the
          position of each token in the sequence.
        position_embedding_name: string. The name of the embedding table variable
          for positional embeddings.
        initializer_range: float. Range of the weight initialization.
        max_position_embeddings: int. Maximum sequence length that might ever be
          used with this model. This can be longer than the sequence length of
          input_tensor, but cannot be shorter.
        dropout_prob: float. Dropout probability applied to the final output tensor.

      Returns:
        float tensor with same shape as `input_tensor`.

      Raises:
        ValueError: One of the tensor shapes or input values is invalid.
      """
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    sequence_length = input_shape[1]
    width = input_shape[2]
    output = input_tensor  # [batch_size, seq_length,embedding_size]

    if use_token_type:
        if token_type_ids is None:  # [batch_size, seq_length]
            raise ValueError("`token_type_ids` must be specified if"
                             "`use_token_type` is True.")
        token_type_table = tf.Variable(
            initial_value=tf.random.truncated_normal(shape=[token_type_vocab_size, width], stddev=0.02),
            name=token_type_embedding_name,
            shape=[token_type_vocab_size, width]
        )
        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])  # [batch_size* seq_length]
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)  # [batch_size* seq_length,
        # token_type_vocab_size]
        token_type_embedding = tf.matmul(one_hot_ids, token_type_table)
        token_type_embedding = tf.reshape(token_type_embedding, shape=[batch_size,sequence_length,width])  # [
        # batch_size,sequence_length,width]
        output += token_type_embedding

    if use_position_embeddings:
        assert_op = tf.debugging.assert_less_equal(sequence_length, max_position_embeddings)  # # 如果seq_length
        # >max_position_embeddings就抛出异常
        # tf.control_dependencies是一个上下文管理器,确保assert_op先计算完成
        with tf.control_dependencies([assert_op]):
            full_position_embedding = tf.Variable(
                initial_value=tf.random.truncated_normal(shape=[max_position_embeddings,width],stddev=0.02),
                shape=[max_position_embeddings,width],
                name=position_embedding_name
            )

    return


def get_shape_list(tensor, expected_rank=None, name=None):
    """返回tensor的shape, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []  # [None,32]:None是动态维度，32是静态维度
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(dim)

    # 没有动态维度
    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]

    return shape


def assert_rank(tensor, expected_rank, name=None):
    return


variable = tf.Variable([[1, 2], [3, 4], [5, 6]])
print(tf.expand_dims(variable, axis=[-1]))
print(tf.one_hot(tf.reshape(tf.expand_dims(variable, axis=[-1]), shape=[-1]), depth=10))
print(variable.shape.as_list())
