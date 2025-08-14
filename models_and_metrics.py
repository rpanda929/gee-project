"""
Deep Learning Models for Geospatial Poverty Mapping
CNN, UNet, Hybrid CNN+UNet architectures
Evaluation metrics: Accuracy, Precision, Recall, F1 Score
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# CNN Model
def build_cnn_segmentation(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(num_classes, (1, 1), padding="same", activation="softmax")(x)
    outputs = layers.UpSampling2D(size=(4,4))(x)
    model = models.Model(inputs, outputs, name="CNN")
    return model

# UNet Model
def build_unet(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Encoder
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D()(c3)
    b = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    # Decoder
    u3 = layers.UpSampling2D()(b)
    m3 = layers.concatenate([u3, c3])
    c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(m3)
    c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(c4)
    u2 = layers.UpSampling2D()(c4)
    m2 = layers.concatenate([u2, c2])
    c5 = layers.Conv2D(128, 3, activation='relu', padding='same')(m2)
    c5 = layers.Conv2D(128, 3, activation='relu', padding='same')(c5)
    u1 = layers.UpSampling2D()(c5)
    m1 = layers.concatenate([u1, c1])
    c6 = layers.Conv2D(64, 3, activation='relu', padding='same')(m1)
    c6 = layers.Conv2D(64, 3, activation='relu', padding='same')(c6)
    outputs = layers.Conv2D(num_classes, (1,1), activation='softmax')(c6)
    model = models.Model(inputs, outputs, name='UNet')
    return model

# Hybrid CNN+UNet Model
def build_hybrid_cnn_unet(input_shape, num_classes):
    # Use UNet base, add SE (Squeeze-Excite Block) and ASPP (Atrous Spatial Pyramid Pooling)
    inputs = keras.Input(shape=input_shape)
    # Encoder block (similar to UNet)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    se1 = layers.GlobalAveragePooling2D()(c1)
    se1 = layers.Reshape((1,1,64))(se1)
    se1 = layers.Dense(64, activation='sigmoid')(se1)
    c1 = layers.multiply([c1, se1])
    p1 = layers.MaxPooling2D()(c1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    se2 = layers.GlobalAveragePooling2D()(c2)
    se2 = layers.Reshape((1,1,128))(se2)
    se2 = layers.Dense(128, activation='sigmoid')(se2)
    c2 = layers.multiply([c2, se2])
    p2 = layers.MaxPooling2D()(c2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    se3 = layers.GlobalAveragePooling2D()(c3)
    se3 = layers.Reshape((1,1,256))(se3)
    se3 = layers.Dense(256, activation='sigmoid')(se3)
    c3 = layers.multiply([c3, se3])
    p3 = layers.MaxPooling2D()(c3)
    b = layers.Conv2D(512, 3, activation='relu', padding='same', dilation_rate=2)(p3)
    # ASPP block
    a1 = layers.Conv2D(512, 3, activation='relu', padding='same', dilation_rate=1)(b)
    a2 = layers.Conv2D(512, 3, activation='relu', padding='same', dilation_rate=2)(b)
    a3 = layers.Conv2D(512, 3, activation='relu', padding='same', dilation_rate=4)(b)
    aspp = layers.concatenate([a1, a2, a3])
    # Decoder block
    u3 = layers.UpSampling2D()(aspp)
    m3 = layers.concatenate([u3, c3])
    c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(m3)
    u2 = layers.UpSampling2D()(c4)
    m2 = layers.concatenate([u2, c2])
    c5 = layers.Conv2D(128, 3, activation='relu', padding='same')(m2)
    u1 = layers.UpSampling2D()(c5)
    m1 = layers.concatenate([u1, c1])
    c6 = layers.Conv2D(64, 3, activation='relu', padding='same')(m1)
    outputs = layers.Conv2D(num_classes, (1,1), activation='softmax')(c6)
    model = models.Model(inputs, outputs, name='Hybrid_CNN_UNet')
    return model

# Metrics calculation wrapper
def compute_metrics(y_true, y_pred, num_classes):
    y_true_flat = np.reshape(y_true, (-1,))
    y_pred_flat = np.reshape(y_pred, (-1,))
    acc = accuracy_score(y_true_flat, y_pred_flat)
    prec = precision_score(y_true_flat, y_pred_flat, average="macro", zero_division=0)
    rec = recall_score(y_true_flat, y_pred_flat, average="macro", zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, average="macro", zero_division=0)
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    return acc, prec, rec, f1, cm