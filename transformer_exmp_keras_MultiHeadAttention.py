import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,Input
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as Callback

# sentences = [['我 是 学 生 喜 欢 学 习', 'S I am a student like learning', 'I am a student like learning E'],  # S: 开始符号
#              ['我 是 男 生 P P P P', 'S I am a boy P P', 'I am a boy P P E'],
#              ['我 喜 欢 学 习 P P P', 'S I like learning P P P', 'I like learning P P P E'],  # E: 结束符号
#              ]  # P: 占位符号，如果当前句子不足固定长度用P占位
#
# src_vocab = {'P': 0, '我': 1, '是': 2, '学': 3, '生': 4, '喜': 5, '欢': 6, '习': 7, '男': 8}  # 词源字典  字：索引
# src_idx2word = {src_vocab[key]: key for key in
#                 src_vocab}  # {0: 'P', 1: '我', 2: '是', 3: '学', 4: '生', 5: '喜', 6: '欢', 7: '习', 8: '男'}
# src_vocab_size = len(src_vocab)  # 字典字的个数
# tgt_vocab = {'P': 0, 'S': 1, 'E': 2, 'I': 3, 'am': 4, 'a': 5, 'student': 6, 'like': 7, 'learning': 8, 'boy': 9}
# idx2word = {tgt_vocab[key]: key for key in tgt_vocab}  # 把目标字典转换成 索引：字的形式
# tgt_vocab_size = len(tgt_vocab)  # 目标字典尺寸
# src_len = len(sentences[0][0].split(" "))  # Encoder输入的最大长度：8
# tgt_len = len(sentences[0][1].split(" "))  # Decoder输入输出最大长度:7
#
#
# # 把sentences 转换成字典索引
# def make_data(sentences):
#     enc_inputs, dec_inputs, dec_outputs = [], [], []
#     for i in range(len(sentences)):
#         enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
#         dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
#         dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]
#         enc_inputs.extend(enc_input)
#         dec_inputs.extend(dec_input)
#         dec_outputs.extend(dec_output)
#     return enc_inputs, dec_inputs, dec_outputs
#
#
# enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
#
#
# # 自定义数据集函数
# dataset = tf.data.Dataset.from_tensor_slices([enc_inputs, dec_inputs, dec_outputs])
#
# d_model = 512   # 字 Embedding 的维度
# d_ff = 2048     # 前向传播隐藏层维度
# d_k = d_v = 64  # K(=Q), V的维度
# n_layers = 6    # 有多少个encoder和decoder
# n_heads = 8     # Multi-Head Attention设置为8




'''
@tf.keras.utils.register_keras_serializable使用它将函数注册到keras序列化框架中
'''
@tf.keras.utils.register_keras_serializable()
class Embedding(tf.keras.layers.Layer):
    def __init__(self, src_vocab_size, d_model, **kwargs):
        self._src_vocab_size = src_vocab_size
        self._d_model = d_model                             #字 Embedding 的维度
        super(Embedding, self).__init__()

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape = (self._src_vocab_size, self._d_model),
            initializer='glorot_uniform',
            name='embeddings'
        )
        super(Embedding, self).build(input_shape)           # 一定要在最后调用它.这个方法必须设 self.built = True，可以通过调用super完成

    def call(self, inputs, **kwargs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        embeddings = K.gather(self.embeddings, inputs)      #查表
        embeddings *= self._d_model ** 0.5                  #scale
        return embeddings


@tf.keras.utils.register_keras_serializable()
class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, masking=True, future=False, dropout_rate=0, **kwargs):
        self._masking = masking                                                             # Padding Mask
        self._future = future                                                               # Attention Mask 生成上三角矩阵
        self._dropout_rate = dropout_rate
        self._masking_num = -2**32+1
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def mask(self, inputs, masks):                                                             # inputs: [batch_size, seq_len]；masks: [batch_size, seq_len]
        '''
        在训练时将补全的位置给Mask掉，也就是在这些位置上补一些无穷小（负无穷）的值，经过softmax操作，这些值就成了0，就不在影响全局概率的预测
        Args:
            inputs:
            masks:

        Returns:

        '''
        masks = K.cast(masks, 'float32')
        masks = K.tile(masks, [K.shape(inputs)[0] // K.shape(masks)[0], 1])                    #对masks进行复制，维度和masks一样是2维，这里只对行复制，列复制1倍
        masks = K.expand_dims(masks, 1)                                                        # [batch_size, 1, len_masks],对数据维度进行扩充，原始数据维度 3x6，扩充后变为 3x6x1 ，如果设置axis=0，则扩充为 1x3x6，同理 axis=1，扩充至 3x1x6，axis=2 作用与 axis=-1一致，都是在最后追加一维。
        outputs = inputs + masks * self._masking_num
        return outputs

    def future_mask(self, inputs):                                                          #[batch_size, seq_len, seq_len]
        '''
        sequence MASK是只存在decoder的第一个mutil_head_self_attention里，为什么这样做？是因为在测试验证阶段，模型并不知道当前时刻的输入和未来时刻的单词信息。
        也就是对于一个序列中的第i个token解码的时候只能够依靠i时刻之前(包括i)的的输出，而不能依赖于i时刻之后的输出。
        因此我们要采取一个遮盖的方法(Mask)使得其在计算self-attention的时候只用i个时刻之前的token进行计算
        Args:
            inputs:

        Returns:

        '''
        diag_vals = tf.ones_like(inputs[0, :, :])                                           #[]创建一个和输入参数（tensor）维度一样，元素都为1的张量。
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()                # 生成上三角矩阵,[batch_size, tgt_len, tgt_len]
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])
        paddings = tf.ones_like(future_masks) * self._masking_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
        return outputs

    def call(self, inputs, **kwargs):
        if self._masking:                                                                   # Q: [batch_size, n_heads, len_q, d_k]
            assert len(inputs) == 4, "inputs 是[queries, keys, values, masks]"           # K: [batch_size, n_heads, len_k, d_k]
            queries, keys, values, masks = inputs                                           # V: [batch_size, n_heads, len_v(=len_k), d_v]
        else:
            assert len(inputs) == 3, "inputs 是[queries, keys, values]"                   # assert 如果条件返回 False，则会引发 表达式2的AssertionError,
            queries, keys, values = inputs

        if K.dtype(queries) != 'float32':   queries = K.cast(queries, 'float32')
        if K.dtype(keys) != 'float32':    keys = K.cast(keys, 'float32')
        if K.dtype(values) != 'float32':    values = K.cast(values, 'float32')

        matmul = K.batch_dot(queries, tf.transpose(keys, [0,2,1]))                          #tf.transpose(keys, [0,2,1]->keys[n_heads,d_k, len_k]
        scaled_matmul = matmul / int(queries.shape[-1]) ** 0.5                  # scale
        if self._masking:
            scaled_matmul = self.mask(scaled_matmul, masks)
        if self._future:
            scaled_matmul = self.future_mask(scaled_matmul)

        softmax_out = K.softmax(scaled_matmul)
        out = K.dropout(softmax_out, self._dropout_rate)
        outputs = K.batch_dot(out, values)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


@tf.keras.utils.register_keras_serializable()
class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, n_heads, head_dim, dropout_rate=.1, masking=True, future=False, trainable=True, **kwargs):
        self._n_heads = n_heads                     #Multi-Head Attention设置为8
        self._head_dim = head_dim                   # _model_dim // _n_heads
        self._dropout_rate = dropout_rate
        self._masking = masking
        self._future = future
        self._trainable = trainable
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self._weights_queries = self.add_weight(                                #(d_model, d_k * n_heads)
            shape=(input_shape[0][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_queries"
        )
        self._weights_keys = self.add_weight(
            shape=(input_shape[1][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_keys"
        )
        self._weights_values = self.add_weight(
            shape=(input_shape[2][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_values"
        )
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self._masking:
            assert len(inputs)==4, "inputs 应该是[queries, keys, values, masks]"
            queries, keys, values, masks = inputs
        else:
            assert len(inputs) == 3, "inputs 应该是[queries, keys, values]"
            queries, keys, values = inputs

        queries_linear = K.dot(queries, self._weights_queries)
        keys_linear = K.dot(keys, self._weights_keys)
        values_linear = K.dot(values, self._weights_values)

        queries_multi_heads = tf.concat(tf.split(queries_linear, self._n_heads, axis=2), axis=0)
        keys_multi_heads = tf.concat(tf.split(keys_linear, self._n_heads, axis=2), axis=0)
        values_multi_heads = tf.concat(tf.split(values_linear, self._n_heads, axis=2), axis=0)

        if self._masking:
            att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads, masks]
        else:
            att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads]

        attention = ScaledDotProductAttention(
            masking=self._masking, future=self._future, dropout_rate=self._dropout_rate
        )
        att_out = attention(att_inputs)
        outputs = tf.concat(tf.split(att_out, self._n_heads, axis=0), axis=2)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape