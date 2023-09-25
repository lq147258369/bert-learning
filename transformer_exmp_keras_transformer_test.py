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
from transformer_exmp_keras_transformer import Transformer


class TestTransformer(tf.test.TestCase):
    def test_save_model(self):
        def get_model():
            encoder_inputs = tf.keras.Input(shape=(256,), name='encoder_inputs')
            decoder_inputs = tf.keras.Input(shape=(256,), name='decoder_inputs')
            out_puts = Transformer(5000,
                                model_dim=8,
                                n_heads=6,
                                encoder_stack=2,
                                decoder_stack=2,
                                feed_forward_size=50)(encoder_inputs, decoder_inputs)
            out_puts = tf.keras.layers.GlobalAveragePooling1D(out_puts)
            out_puts = tf.keras.layers.Dense(out_puts)
            return tf.keras.Model(inputs = [encoder_inputs, decoder_inputs], out_puts = out_puts)
        model = get_model()
        encoder_random_input = np.random.randint(size=(10,256), low=0, high=5000)
        decoder_random_input = np.random.randint(size=(10, 256), low=0, high=5000)
        model_pred = model.predict([encoder_random_input, decoder_random_input])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "transformer_model")
            model.save(path)
            loaded_model = tf.keras.models.load_model(path)
            loaded_pred = loaded_model.predict([encoder_random_input, decoder_random_input])
        for i in range(len(model.layers)):
            assert model.layers[i].get_config() == loaded_model.layers[i].get_config()
        self.assertAllClose(model_pred, loaded_pred)
