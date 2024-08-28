import tensorflow as tf
from keras.regularizers import l1, l2
from keras.layers import GRU, MultiHeadAttention, Dense, Concatenate, TimeDistributed, LeakyReLU, Layer, MaxPooling1D
class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, channels, reduction):
        super(ChannelAttention, self).__init__()
        self.avg_pool = tf.keras.layers.GlobalAveragePooling3D()
        self.max_pool = tf.keras.layers.GlobalMaxPooling3D()
        self.fc_avg = tf.keras.Sequential([
            tf.keras.layers.Dense(channels // reduction, kernel_regularizer=l2(0.01), use_bias=True, bias_initializer='zeros'),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(channels, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

        ])
        self.fc_max = tf.keras.Sequential([
            tf.keras.layers.Dense(channels // reduction, kernel_regularizer=l2(0.01), use_bias=True, bias_initializer='zeros'),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(channels, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        ])
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)

        channel_attention_avg = self.fc_avg(avg_pool)
        channel_attention_max = self.fc_max(max_pool)
        channel_attention = self.sigmoid(0.5*(channel_attention_max - channel_attention_avg))


        return tf.expand_dims(tf.expand_dims(tf.expand_dims(channel_attention, 1), 1), 1)


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 3), padding='same', activation='sigmoid')

    def call(self, x):
        max_pool = tf.reduce_max(x, axis=4)
        avg_pool = tf.reduce_mean(x, axis=4)
        max_pool = tf.expand_dims(max_pool, axis=-1)
        avg_pool = tf.expand_dims(avg_pool, axis=-1)
        combined = tf.concat([max_pool, avg_pool], axis=4)
        spatial_attention = self.conv(combined)
        return spatial_attention

# 定義CBAM模塊
class CBAMModule(tf.keras.layers.Layer):
    def __init__(self, channels, reduction, **kwargs):
        super(CBAMModule, self).__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def call(self, x):
        channel_attention = self.channel_attention(x)
        spatial_attention = self.spatial_attention(x)

        result = tf.multiply(x, channel_attention)
        result = tf.multiply(result, spatial_attention)

        return result

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "reduction": self.reduction,
        })
        return config

class SingleHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(SingleHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # 定義Q、K、V的全連接層
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        # 定義最終輸出的全連接層
        self.dense = Dense(d_model)
        self.sqrt_key_dim = tf.sqrt(tf.cast(d_model, tf.float32))

    def call(self, v, k, q):
        # 通過全連接層轉換Q、K、V
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # 計算注意力分數並應用softmax
        attention_scores = tf.matmul(q, k, transpose_b=True) / self.sqrt_key_dim
        attention_scores = MaxPooling1D(pool_size=10, padding='same')(attention_scores)
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        # 計算加權和
        output = tf.matmul(attention_scores, v)
        # 通過全連接層輸出結果
        output = self.dense(output)

        return output

    def get_config(self):
        config = super(SingleHeadAttention, self).get_config()
        config.update({'d_model': self.d_model})

        return config


class MultiScaleMultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads, scale_factors):
        super(MultiScaleMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.scale_factors = scale_factors

        self.heads = []
        for scale_factor in scale_factors:
            scale_d_model = d_model // scale_factor
            self.heads.append(SingleHeadAttention(scale_d_model, num_heads))

        self.concat = Concatenate()

    def call(self, v, k, q):
        all_attention_outputs = []
        for head in self.heads:
            attention_output = head(v, k, q)
            all_attention_outputs.append(attention_output)

        concatenated_output = self.concat(all_attention_outputs)
        return concatenated_output

    def get_config(self):
        config = super(MultiScaleMultiHeadAttention, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'scale_factors': self.scale_factors
        })
        return config