import tempfile
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,Input
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as Callback
from transformer_exmp_keras_MultiHeadAttention import MultiHeadAttention

tf.config.experimental_run_functions_eagerly(True)

@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(Layer):
    def __init__(self, model_dim, **kwargs):
        self._model_dim = model_dim                 #字 Embedding 的维度
        super(PositionalEncoding, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """

        Args:
            inputs: Input tensor, or list/tuple of input tensors.[batch_size, max_len, model_dim]
            max_len:
            **kwargs:

        Returns:Input tensor, or list/tuple of input tensors.

        """
        max_len = inputs.shape[1]
        pos_table = np.zeros((max_len, self._model_dim)) #一行代表某字的position embedding
        for pos in range(max_len):
            for i in range(self._model_dim):
                pos_table[pos, i] = pos / np.power(10000, 2 * i / self._model_dim)
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])           # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])           # 字嵌入维度为奇数时
        pos_table = K.cast(pos_table, 'float32')                     # [max_len, model_dim]
        return pos_table

    def compute_output_shape(self, input_shape):
        """
        function is used to determine the output shape.
        Args:
            input_shape:

        Returns:

        """
        return input_shape


@tf.keras.utils.register_keras_serializable()
class Add(Layer):
    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        input_a, input_b = inputs
        return input_a + input_b

    def compute_output_shape(self, input_shape):
        return input_shape[0]


@tf.keras.utils.register_keras_serializable()
class PositionWiseFeedForward(Layer):
    def __init__(self, model_dim, inner_dim, trainable=True, **kwargs):
        self._model_dim = model_dim
        self._inner_dim = inner_dim
        self._trainable = trainable
        super(PositionWiseFeedForward, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weights_inner = self.add_weight(
            shape=(input_shape[-1], self._inner_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_inner'
        )
        self.weights_out = self.add_weight(
            shape=(self._inner_dim, self._model_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_out'
        )
        self.bias_inner = self.add_weight(
            shape=(self._inner_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name='bias_inner'
        )
        self.bias_out = self.add_weight(
            shape=(self._model_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name='bias_out'
        )
        super(PositionWiseFeedForward, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.dtype != 'float32':
            inputs = K.cast(inputs, 'float32')
        inner_out = K.relu(K.dot(inputs, self.weights_inner) + self.bias_inner)
        outputs = K.dot(inner_out, self.weights_out) + self.bias_out
        return  outputs

    def compute_output_shape(self, input_shape):
        return self._model_dim

@tf.keras.utils.register_keras_serializable()
class LayerNormalization(Layer):
    def __init__(self, epsilon=1e-8, **kwargs):
        self._epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zero',
            name='beta'
        )
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='one',
            name = 'gamma'
        )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        nomalized = (inputs - mean) / ((variance + self._epsilon) ** 0.5)
        outputs = self.gamma * nomalized + self. beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


@tf.keras.utils.register_keras_serializable()
class Transformer(Layer):
    def __init__(self,
                 vocab_size,
                 model_dim,
                 n_heads = 8,
                 encoder_stack = 6,
                 decoder_stack = 6,
                 feed_forward_size = 2048,
                 dropout_rate = 0.1,
                 **kwargs):
        self._vocab_size = vocab_size
        self._model_dim = model_dim
        self._n_heads = n_heads
        self._encoder_stack = encoder_stack
        self._decoder_stack = decoder_stack
        self._feed_forward_size = feed_forward_size
        self._dropout_rate = dropout_rate
        super(Transformer,self).__init__(**kwargs)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self._vocab_size, self._model_dim),
            initializer='glorot_uniform',
            name='embeddings'
        )
        '''
        Encoder部分
        '''
        self.EncoderPositionEncoding = PositionalEncoding(self._model_dim)
        self.EncoderMultiHeadAttention = [
            MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads)
            for _ in range(self._encoder_stack)
        ]
        self.EncoderLayerNorm0 = [
            LayerNormalization()
            for _ in range(self._encoder_stack)
        ]
        self.EncoderPositionWiseFeedFowards = [
            PositionWiseFeedForward(self._model_dim, self._feed_forward_size)
            for _ in range(self._encoder_stack)
        ]
        self.EncoderLayerNorm1 = [
            LayerNormalization()
            for _ in range(self._encoder_stack)
        ]
        '''
        Decoder部分
        '''
        self.DecoderPositionEncoding = PositionalEncoding(self._model_dim)
        self.DecoderMultiHeadAttention0 = [
            MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads, future=True)
            for _ in range(self._decoder_stack)
        ]
        self.DecoderLayerNorm0 = [
            LayerNormalization()
            for _ in range(self._decoder_stack)
        ]
        self.DecoderMultiHeadAttention1 = [
            MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads)
            for _ in range(self._decoder_stack)
        ]
        self.DecoderLayerNorm1 = [
            LayerNormalization()
            for _ in range(self._decoder_stack)
        ]
        self.DecoderPositionWiseFeedFowards = [
            PositionWiseFeedForward(self._model_dim, self._feed_forward_size)
            for _ in range(self._decoder_stack)
        ]
        self.DecoderLayerNorm2 = [
            LayerNormalization()
            for _ in range(self._decoder_stack)
        ]

    def encoder(self, inputs):
        '''

        Args:
            inputs: [batch_size, max_len]

        Returns:

        '''
        if K.dtype != 'int32':
            inputs = K.cast(inputs, 'int32')
        masks = K.equal(inputs, 0)                          #[batch_size, max_len]
        # embeddings
        embeddings = K.gather(self.embeddings, inputs)
        embeddings *= self._model_dim ** 0.5                # embeddings: [batch_size, max_len, model_dim]
        # position encoding
        position_encodings = self.EncoderPositionEncoding(embeddings)       # position_encodings: [max_len, model_dim]
        '''
        维度不同的 Tensor 如何相加呢？但是是自动填充对应维度，使维度相同后再相加。
        如 torch.Size([1, 2, 3, 4]) 与 torch.Size([3, 4]) 相加，先将后者填充为 torch.Size([1, 2, 3, 4]) ，二者相加后 shape 为 [1, 2, 3, 4]
        torch.Size([3, 2, 1, 4]) 与 torch.Size([3, 4]) 相加，则先将两者填充为 torch.Size([3, 2, 3, 4]) ，二者相加后 shape 为 [3, 2, 3, 4]，有种类似取最小公倍数的感觉。

        '''
        encodings = embeddings + position_encodings                         #encodings:[batch_size, max_len, model_dim]
        encodings = K.dropout(encodings, self._dropout_rate)

        for i in range(self._encoder_stack):
            #多头注意力
            attention = self.EncoderMultiHeadAttention[i]
            attention_input = [encodings, encodings, encodings, masks]      #masks:[batch_size, max_len]
            attention_out = attention(attention_input)                      #attention_out:[batch_size, max_len, model_dim]
            # add & norm
            attention_out += encodings
            attention_out = self.EncoderLayerNorm0[i](attention_out)
            # feed forward
            ff = self.EncoderPositionWiseFeedFowards[i]
            ff_out = ff(attention_out)
            # add & norm
            ff_out += attention_out
            encodings = self.EncoderLayerNorm1[i](ff_out)
        return encodings, masks

    def decoder(self, inputs):
        decoder_inputs, encoder_encodings, encoder_masks = inputs
        if K.dtype(decoder_inputs) != 'int32':
            decoder_inputs = K.cast(decoder_inputs, 'int32')

        decoder_masks = K.equal(decoder_inputs, 0)
        # embedding
        embeddings = K.gather(self.embeddings, decoder_inputs)
        embeddings *= self._model_dim ** 0.5
        # position encoding
        position_encodings = self.DecoderPositionEncoding(embeddings)
        encodings = embeddings + position_encodings
        encodings = K.dropout(encodings, self._dropout_rate)

        for i in range(self._decoder_stack):
            # masked multi-head attention
            masked_attention = self.DecoderMultiHeadAttention0[i]
            masked_attention_input = [encodings, encodings, encodings, decoder_masks]
            masked_attention_out = masked_attention(masked_attention_input)

            # add & norm
            masked_attention_out += encodings
            masked_attention_out = self.DecoderLayerNorm0[i](masked_attention_out)

            # multi-head attention
            attention = self.DecoderMultiHeadAttention1[i]
            attention_input = [masked_attention_out, encoder_encodings, encoder_encodings, encoder_masks]
            attention_out = attention(attention_input)

            # add & norm
            attention_out += masked_attention_out
            attention_out = self.DecoderLayerNorm1[i](attention_out)

            # feed forward
            ff = self.DecoderPositionWiseFeedFowards[i]
            ff_out = ff(attention_out)

            # add & norm
            ff_out += attention_out
            encodings = self.DecoderLayerNorm2[i](ff_out)

        # pre-softmax 和 embeddings 共享参数
        linear_projection = K.dot(encodings, K.transpose(self.embeddings))
        outputs = K.softmax(linear_projection)
        return  outputs

    def call(self, encoder_inputs, decoder_inputs, **kwargs):
        '''

        Args:
            encoder_inputs: [batch_size, max_len]
            decoder_inputs: [batch_size, max_len]，是encoder_inputs的mask 0或1，不知道TRUE FALSE 怎么转换成了0,1。可能是文本分类不需要decoder，所以decoder_inputs设置成了encoder_inputs的mask0,1
            **kwargs:

        Returns:

        '''
        encoder_encodings, encoder_masks = self.encoder(encoder_inputs)         #encoder_encodings:[batch_size, max_len,model_dim]
        encoder_outputs = self.decoder([decoder_inputs, encoder_encodings, encoder_masks])
        return encoder_outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self._vocab_size

    def get_config(self):
        config = {
            "vocab_size" : self._vocab_size,
            "model_dim": self._model_dim,
            "n_heads": self._n_heads,
            "encoder_stack": self._encoder_stack,
            "decoder_stack": self._decoder_stack,
            "dropout_rate": self._dropout_rate
        }
        base_config = super(Transformer, self).get_config()
        return {**base_config, **config}


def load_dataset(vocab_size, max_len):
    '''

    Args:
        vocab_size:
        max_len: 超过max_len会被截断
    x_train:[batch_size, src_len]
    Returns:

    '''
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(maxlen=max_len, num_words=vocab_size)
    # padding补齐为统一长度，默认从前面开始补
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)
    # 等于返回TRUE
    x_train_masks = tf.equal(x_train, 0)
    x_test_masks = tf.equal(x_test, 0)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    return (x_train, x_train_masks, y_train), (x_test, x_test_masks, y_test)

def build_model(vocab_size, max_len, model_dim=8, n_heads=2, encoder_stack=2, decoder_stack=2, ff_size=50):
    '''

    Args:
        vocab_size:
        max_len:
        model_dim:  字 Embedding 的维度
        n_heads:
        encoder_stack:
        decoder_stack:
        ff_size:  前向传播隐藏层维度

    Returns:

    '''
    encoder_inputs = tf.keras.Input(shape=(max_len,),name='encoder_inputs')
    decoder_inputs = tf.keras.Input(shape=(max_len,),name='decoder_inputs')
    outputs = Transformer(
        vocab_size,
        model_dim,
        n_heads=n_heads,
        encoder_stack=encoder_stack,
        decoder_stack=decoder_stack,
        feed_forward_size=ff_size
    )(encoder_inputs, decoder_inputs)
    outputs = tf.keras.layers.GlobalAveragePooling1D()(outputs)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(outputs)
    return  tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)


def train_model(vocab_size=500, max_len=128, batch_size=256, epochs=10):
    train, test = load_dataset(vocab_size=vocab_size, max_len=max_len)
    x_train, x_train_masks, y_train = train
    x_test, x_test_masks, y_test = test
    model = build_model(vocab_size=vocab_size, max_len=max_len)
    model.compile(optimizer=tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    es = tf.keras.callbacks.EarlyStopping(patience=3)
    model.fit([x_train, x_train_masks], y_train,
              batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[es])
    test_metrics = model.evaluate([x_test, x_test_masks], y_test, batch_size=batch_size, verbose=0)
    print("loss on Test:%.4f" % test_metrics[0])

if __name__=='__main__':
    train_model()



