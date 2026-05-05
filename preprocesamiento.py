# Carga y prepara las imágenes del dataset MNIST
# By: Jhonatan Najarro

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def cargar_datos():
    print('Cargando dataset MNIST...')
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(f'Imágenes de entrenamiento : {X_train.shape}')
    print(f'Imágenes de prueba        : {X_test.shape}')
    return (X_train, y_train), (X_test, y_test)


def normalizar(X_train, X_test):
    X_train_norm = X_train / 255.0
    X_test_norm  = X_test  / 255.0
    print('Normalización aplicada: píxeles escalados a [0-1]')
    return X_train_norm, X_test_norm


def reshape_cnn(X_train, X_test):
    X_train_cnn = X_train.reshape(-1, 28, 28, 1)
    X_test_cnn  = X_test.reshape(-1, 28, 28, 1)
    print(f'Reshape aplicado: {X_train_cnn.shape}')
    return X_train_cnn, X_test_cnn


def codificar_etiquetas(y_train, y_test):
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat  = to_categorical(y_test, 10)
    print(f'Etiquetas codificadas: {y_train_cat.shape}')
    return y_train_cat, y_test_cat


def aplanar(X_train, X_test):
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat  = X_test.reshape(len(X_test), -1)
    print(f'Datos aplanados: {X_train_flat.shape}')
    return X_train_flat, X_test_flat


if __name__ == '__main__':
    print('=' * 55)
    print('  PREPROCESAMIENTO - TECNOFORMS')
    print('=' * 55)

    (X_train, y_train), (X_test, y_test) = cargar_datos()
    X_train_norm, X_test_norm             = normalizar(X_train, X_test)
    X_train_cnn,  X_test_cnn              = reshape_cnn(X_train_norm, X_test_norm)
    y_train_cat,  y_test_cat              = codificar_etiquetas(y_train, y_test)
    X_train_flat, X_test_flat             = aplanar(X_train_norm, X_test_norm)

    print('\nPreprocesamiento completado correctamente.')
    print('=' * 55)