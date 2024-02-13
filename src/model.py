import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D, Conv2DTranspose, LeakyReLU, concatenate, MaxPool2D, ReLU, Add, \
    Lambda
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D, Multiply, Dense, \
    Activation
from conv import Conv2D as RQConv2D
from conv import Conv2DTranspose as RQConv2DTranspose
from dense import Dense as RQDense
from tensorflow import matmul, reshape, reduce_sum, transpose



class ImageUpgradingBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=1, padding="same")
        self.conv2 = tf.keras.layers.Conv2D(filters=6, kernel_size=1, padding="same")
        self.conv3 = tf.keras.layers.Conv2D(filters=6, kernel_size=1, padding="same")
        self.conv4 = tf.keras.layers.Conv2D(filters=6, kernel_size=1, padding="same")
        self.conv_residual = tf.keras.layers.Conv2D(filters=4, kernel_size=1, padding="same")  # 升维补偿卷积块

    def call(self, inputs, *args, **kwargs):
        maps_x_origin = inputs[:, :, :, :3]  # B*H*W*3
        maps_y_origin = inputs[:, :, :, 3:]
        b_, h_, w_, c = maps_x_origin.shape
        maps_x = transpose(reshape(maps_x_origin, [-1, h_ * w_, c]), perm=[0, 2, 1])  # B*3*HW
        maps_y = transpose(reshape(maps_y_origin, [-1, h_ * w_, c]), perm=[0, 2, 1])
        maps_x_h = hilbert_transform(maps_x, axis=1)  # B*3*HW
        maps_y_h = hilbert_transform(maps_y, axis=1)
        cat_x = reshape(transpose(concatenate([maps_x, maps_x_h], axis=1), perm=[0, 2, 1]), [-1, h_, w_, 6])  # B*H*W*6
        cat_y = reshape(transpose(concatenate([maps_y, maps_y_h], axis=1), perm=[0, 2, 1]), [-1, h_, w_, 6])

        one = tf.ones(shape=(1, 1, 1, 1))
        v1 = self.conv1(one)
        v2 = self.conv2(one)
        v3 = self.conv3(one)
        v4 = self.conv4(one)
        v1 = reshape(v1, [-1, 6, 1])  # basis vector with shape(-1,3,1)
        v2 = reshape(v2, [-1, 6, 1])
        v3 = reshape(v3, [-1, 6, 1])
        v4 = reshape(v4, [-1, 6, 1])
        V = concatenate([v1, v2, v3, v4], axis=-1)  # the 4-D column subspace with shape(_,6,4)
        V = V / (1e-6 + reduce_sum(abs(V), axis=1, keepdims=True))
        maps_x_t = reshape(cat_x, [-1, h_ * w_, 6])  # B*HW*6
        maps_y_t = reshape(cat_y, [-1, h_ * w_, 6])
        V_t = transpose(V, perm=[0, 2, 1])  # B*4*6
        mat = matmul(V_t, V)  # B*4*4
        mat_inv = tf.linalg.inv(mat)
        # Projection --->  Y = V×(V_t×V)_(-1)×V_t×(X)
        reconstruct_x_t = matmul(V, matmul(matmul(mat_inv, V_t), transpose(maps_x_t, perm=[0, 2, 1])))  # B*6*HW
        reconstruct_y_t = matmul(V, matmul(matmul(mat_inv, V_t), transpose(maps_y_t, perm=[0, 2, 1])))

        if b_ is None:  # 判断batch是否为空，为空则置为1再进行最小二乘法求解
            reconstruct_x = tf.linalg.lstsq(tf.squeeze(V, axis=[0]), tf.squeeze(reconstruct_x_t, axis=[0]))
            reconstruct_y = tf.linalg.lstsq(tf.squeeze(V, axis=[0]), tf.squeeze(reconstruct_y_t, axis=[0]))
            reconstruct_x = tf.expand_dims(reconstruct_x, axis=0)  # shape(1,c,h*w) why can‘t(None,c,h*w)
            reconstruct_y = tf.expand_dims(reconstruct_y, axis=0)
        else:
            reconstruct_x = tf.linalg.lstsq(V, reconstruct_x_t)
            reconstruct_y = tf.linalg.lstsq(V, reconstruct_y_t)

        reconstruct_x = reshape(transpose(reconstruct_x, perm=[0, 2, 1]), [-1, h_, w_, 4])  # shape(-1,h,w,4)
        reconstruct_y = reshape(transpose(reconstruct_y, perm=[0, 2, 1]), [-1, h_, w_, 4])
        reconstruct_x_add = self.conv_residual(cat_x)  # 考虑补齐的残差信息是6D cat_x还是原始3D map_x_origin
        reconstruct_y_add = self.conv_residual(cat_y)
        reconstruct_x = Add()([reconstruct_x, reconstruct_x_add])
        reconstruct_y = Add()([reconstruct_y, reconstruct_y_add])
        return reconstruct_x, reconstruct_y


class ImageDowngradingBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ImageDowngradingBlock, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(filters=4, kernel_size=1, padding="same")  # projection v1
        self.conv2 = tf.keras.layers.Conv2D(filters=4, kernel_size=1, padding="same")  # projection v2
        self.conv3 = tf.keras.layers.Conv2D(filters=4, kernel_size=1, padding="same")  # projection v3
        self.conv4 = tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding="same")  # 对照组

    def call(self, inputs, *args, **kwargs):
        one = tf.ones(shape=(1, 1, 1, 1))
        v1 = self.conv1(one)
        v2 = self.conv2(one)
        v3 = self.conv3(one)
        v1 = reshape(v1, [-1, 4, 1])  # basis vector with shape(-1,3,1)
        v2 = reshape(v2, [-1, 4, 1])
        v3 = reshape(v3, [-1, 4, 1])
        V = concatenate([v1, v2, v3], axis=-1)  # the 3-D column subspace with shape(_,3,2)
        V = V / (1e-6 + reduce_sum(abs(V), axis=1, keepdims=True))
        b_, h_, w_, c = inputs.shape
        inputs_t = transpose(reshape(inputs, [-1, h_ * w_, c]), perm=[0, 2, 1])
        V_t = transpose(V, perm=[0, 2, 1])  # 1，2维之间的数据做转置
        mat = matmul(V_t, V)
        try:
            mat_inv = tf.linalg.inv(mat)
            # Projection --->  Y = V×(V_t×V)_(-1)×V_t×(X)
            reconstruct_inputs_t = matmul(V, matmul(matmul(mat_inv, V_t), inputs_t))
            if b_ is None:  # 判断batch是否为空，为空则置为1再进行最小二乘法求解
                reconstruct_inputs = tf.linalg.lstsq(tf.squeeze(V, axis=[0]),
                                                     tf.squeeze(reconstruct_inputs_t, axis=[0]))
                reconstruct_inputs = tf.expand_dims(reconstruct_inputs, axis=0)  # shape(1,c,h*w) why can‘t(None,c,h*w)
            else:
                reconstruct_inputs = tf.linalg.lstsq(V, reconstruct_inputs_t)
            reconstruct_inputs = transpose(reconstruct_inputs, perm=[0, 2, 1])  # shape(-1,h*w,c)
            reconstruct_inputs = reshape(reconstruct_inputs, [-1, h_, w_, c - 1])
        except:
            print("something wrong to Down-hilbert transform")
            reconstruct_inputs = self.conv4(inputs)
        return reconstruct_inputs





def hilbert_transform(x, N=None, axis=-1):
    x_complex = tf.complex(x, x * 0)
    if x_complex.dtype != tf.complex64 and x_complex.dtype != tf.complex128:
        raise ValueError("x must be complex.")
    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")
    x_t = transpose(x_complex, perm=[0, 2, 1])
    x_f = tf.signal.fft(x_t)
    x_f = transpose(x_f, perm=[0, 2, 1])

    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    if x.shape.ndims > 1:
        ind = [tf.newaxis] * x.shape.ndims
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    x_f_h = x_f * h
    x_f = transpose(x_f_h, perm=[0, 2, 1])
    x_hilbert = tf.signal.ifft(x_f)
    x_hilbert = transpose(x_hilbert, perm=[0, 2, 1])
    return tf.math.imag(x_hilbert)


class RQCA(tf.keras.layers.Layer):
    def __init__(self, RQsubchannel, RQchannel, **kwargs):
        super(RQCA, self).__init__(**kwargs)
        self.pool = GlobalAveragePooling2D()
        self.down = RQDense(RQsubchannel, activation=tf.nn.leaky_relu)
        self.up = RQDense(RQchannel, activation='sigmoid')

    def __call__(self, inputs):
        gap_x = self.pool(inputs)
        down_x = self.down(gap_x)
        up_x = self.up(down_x)
        return up_x


def EDDM():
    input_image = layers.Input(shape=(512, 512, 6), dtype="float32")
    x_4d, y_4d = ImageUpgradingBlock()(input_image)
    # downsample-1
    conv1 = RQConv2D(filters=8, kernel_size=3, padding="same")(x_4d)
    conv1 = LeakyReLU()(conv1)
    conv1 = RQConv2D(filters=8, kernel_size=3, padding="same")(conv1)
    bridge1 = LeakyReLU()(conv1)
    pool1 = RQConv2D(filters=8, kernel_size=4, strides=2, padding="same")(bridge1)
    # downsample-2
    conv2 = RQConv2D(filters=16, kernel_size=3, padding="same")(pool1)
    conv2 = LeakyReLU()(conv2)
    conv2 = RQConv2D(filters=16, kernel_size=3, padding="same")(conv2)
    bridge2 = LeakyReLU()(conv2)
    pool2 = RQConv2D(filters=16, kernel_size=4, strides=2, padding="same")(bridge2)

    # downsample-3
    conv3 = RQConv2D(filters=32, kernel_size=3, padding="same")(pool2)
    conv3 = LeakyReLU()(conv3)
    conv3 = RQConv2D(filters=32, kernel_size=3, padding="same")(conv3)
    bridge3 = LeakyReLU()(conv3)
    pool3 = RQConv2D(filters=32, kernel_size=4, strides=2, padding="same")(bridge3)
    # bottom
    conv4 = RQConv2D(filters=64, kernel_size=3, padding="same")(pool3)
    conv4 = LeakyReLU()(conv4)
    conv4 = RQConv2D(filters=64, kernel_size=3, padding="same")(conv4)
    conv4 = LeakyReLU()(conv4)

    # upsample-3
    up3 = RQConv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv4)
    # RQCAM-3 begin
    fussion3 = RQCA(RQchannel=32, RQsubchannel=4)(up3)
    rqca3 = Multiply()([fussion3, bridge3])
    concat3 = concatenate([rqca3, up3], axis=-1)
    # end
    conv7 = RQConv2D(filters=32, kernel_size=3, padding="same")(concat3)
    conv7 = LeakyReLU()(conv7)
    conv7 = RQConv2D(filters=32, kernel_size=3, padding="same")(conv7)
    conv7 = LeakyReLU()(conv7)
    # upsample-2
    up2 = RQConv2DTranspose(filters=16, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7)
    # RQCAM-2-begin
    fussion2 = RQCA(RQchannel=16, RQsubchannel=2)(up2)
    rqca2 = Multiply()([fussion2, bridge2])
    concat2 = concatenate([rqca2, up2], axis=-1)
    # end
    conv8 = RQConv2D(filters=16, kernel_size=3, padding="same")(concat2)
    conv8 = LeakyReLU()(conv8)
    conv8 = RQConv2D(filters=16, kernel_size=3, padding="same")(conv8)
    conv8 = LeakyReLU()(conv8)
    # upsample-1
    up1 = RQConv2DTranspose(filters=8, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv8)
    # RQCAM-1-begin
    fussion1 = RQCA(RQchannel=8, RQsubchannel=1)(up1)
    rqca1 = Multiply()([fussion1, bridge1])
    concat1 = concatenate([rqca1, up1], axis=-1)
    # end
    conv9 = RQConv2D(filters=8, kernel_size=3, padding="same")(concat1)
    conv9 = LeakyReLU()(conv9)
    conv9 = RQConv2D(filters=8, kernel_size=3, padding="same")(conv9)
    conv9 = LeakyReLU()(conv9)
    rq_x_4d = RQConv2D(filters=1, kernel_size=3, padding="same")(conv9)
    # recover 3-channel-R-G-B feature maps
    rq_x_3d = ImageDowngradingBlock()(rq_x_4d)
    res = Add()([input_image[:, :, :, :3], rq_x_3d])
    model = models.Model(inputs=input_image, outputs=[y_4d, rq_x_4d, res])
    return model


