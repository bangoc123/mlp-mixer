import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Normalization, Resizing, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from keras import layers
from tensorflow.keras.layers import Embedding, Input, LayerNormalization, Dense, GlobalAveragePooling1D, Dropout


class Patches(layers.Layer):
  def __init__(self, patch_size):
    super(Patches, self).__init__()
    self.patch_size = patch_size

  def call(self, images):
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(
        images= images, 
        sizes = [1, self.patch_size, self.patch_size, 1], 
        strides = [1, self.patch_size, self.patch_size, 1],
        rates = [1, 1, 1, 1], 
        padding = 'VALID', 
    )
    dim = patches.shape[-1]

    patches = tf.reshape(patches, (batch_size, -1, dim)) 
    return patches  
    
class MLPBlock(tf.keras.layers.Layer):
  def __init__(self, S, C, DS, DC):
    super(MLPBlock, self).__init__()
    self.layerNorm1 = LayerNormalization()
    self.layerNorm2 = LayerNormalization()
    w_init = tf.random_normal_initializer()
    self.DS = DS
    self.DC = DC
    self.W1 = tf.Variable(
            initial_value=w_init(shape=(DS, S), dtype="float32"),
            trainable=True,
    )
    self.W2 = tf.Variable(
            initial_value=w_init(shape=(S, DS), dtype="float32"),
            trainable=True,
    )
    self.W3 = tf.Variable(
            initial_value=w_init(shape=(C, DC), dtype="float32"),
            trainable=True,
    )
    self.W4 = tf.Variable(
            initial_value=w_init(shape=(DC, C), dtype="float32"),
            trainable=True,
    )

  def call(self, X):
    # patches (..., S, C)
    batch_size, S, C = X.shape
    
    # Token-mixing
    W1X = tf.matmul(self.W1, self.layerNorm1(X)) # (DS, S) x (..., S, C) = (..., DS, C)

    # assert W1X.shape == (batch_size, self.DS, C)

    U = tf.matmul(self.W2, tf.nn.gelu(W1X)) + X  # (S, DS) x (..., DS, C) + (..., S, C) = (..., S, C)

    # assert U.shape == (batch_size, S, C)

    # Channel-minxing
    W3U = tf.matmul(self.layerNorm2(U), self.W3) # (...,S, C) x (C, DC) = (..., S, DC)
    
    # assert W3U.shape == (batch_size, S, self.DC)
    
    Y = tf.matmul(tf.nn.gelu(W3U), self.W4) + U  # (..., S, DC) x (..., DC, C) + (..., S, C) = (..., S, C)

    # assert Y.shape == (batch_size, S, C)

    return Y


class MLPMixer(tf.keras.models.Model):
  def __init__(self, patch_size, S, C, DS, DC, num_of_mlp_blocks, image_size, batch_size, num_classes):
    super(MLPMixer, self).__init__()
    self.projection = Dense(C)
    self.mlpBlocks = [MLPBlock(S, C, DS, DC) for _ in range(num_of_mlp_blocks)]
    
    self.batch_size = batch_size
    self.patch_size = patch_size
    self.S = S
    self.C = C
    self.DS = DS
    self.DC = DC
    self.image_size = image_size
    self.num_classes = num_classes


    self.data_augmentation = tf.keras.Sequential(
        [
            Normalization(),
            # Resizing(image_size, image_size),
            RandomFlip("horizontal"),
            RandomRotation(factor=0.02),
            RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )

    self.classificationLayer = Sequential([
          GlobalAveragePooling1D(),
          Dropout(0.2),
          Dense(num_classes)
    ])

  def extract_patches(self, images, patch_size):
    batch_size = tf.shape(images)[0]

    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )

    patches = tf.reshape(patches, [batch_size, -1, 3 * patch_size ** 2])

    return patches

  def call(self, images):
    # input
    # images: (batch_size, IMAGE_SIZE, IMAGE_SIZE, 3) = (32, 64, 64, 3)

    batch_size = images.shape[0]
    
    augumented_images = self.data_augmentation(images)

    # assert augumented_images.shape == (batch_size, self.image_size, self.image_size, 3)

    # patches: (batch_size, S, C) 
    X = self.extract_patches(augumented_images, self.patch_size)

    # assert X.shape == (batch_size, self.S, self.C), (X.shape, self.S, self.C)
    for block in self.mlpBlocks:
      X = block(X)

    # assert X.shape == (batch_size, self.S, self.C), (batch_size, self.S, self.C)

    # out: (batch_size, C)
    out = self.classificationLayer(X)

    # assert out.shape == (batch_size, self.num_classes)
    return out

