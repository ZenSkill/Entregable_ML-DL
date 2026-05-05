# Define, entrena y guarda el modelo CNN
# By: Jhonatan Najarro

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from preprocesamiento import cargar_datos, normalizar, reshape_cnn, codificar_etiquetas


def construir_modelo():
    modelo = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return modelo


def entrenar_modelo(modelo, X_train_cnn, y_train_cat):
    print('Entrenando modelo CNN...')
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    historial = modelo.fit(
        X_train_cnn, y_train_cat,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    print('Entrenamiento finalizado.')
    return historial


def guardar_arquitectura_xml(modelo, nombre_archivo='arquitectura_cnn.xml'):
    raiz = ET.Element('modelo')
    raiz.set('nombre',          'CNN_TecnoForms')
    raiz.set('optimizador',     'adam')
    raiz.set('funcion_perdida', 'categorical_crossentropy')
    raiz.set('clases',          '10')
    raiz.set('input_shape',     '28x28x1')

    for i, capa in enumerate(modelo.layers):
        nodo = ET.SubElement(raiz, 'capa')
        nodo.set('numero', str(i + 1))
        nodo.set('tipo',   capa.__class__.__name__)
        nodo.set('nombre', capa.name)

        config = capa.get_config()
        if capa.__class__.__name__ == 'Conv2D':
            nodo.set('filtros',    str(config['filters']))
            nodo.set('kernel',     str(config['kernel_size']))
            nodo.set('activacion', str(config['activation']))
        elif capa.__class__.__name__ == 'Dense':
            nodo.set('neuronas',   str(config['units']))
            nodo.set('activacion', str(config['activation']))
        elif capa.__class__.__name__ == 'Dropout':
            nodo.set('tasa',       str(config['rate']))
        elif capa.__class__.__name__ == 'MaxPooling2D':
            nodo.set('pool_size',  str(config['pool_size']))

    arbol = ET.ElementTree(raiz)
    ET.indent(arbol, space='    ')
    arbol.write(nombre_archivo, encoding='utf-8', xml_declaration=True)
    print(f'Arquitectura guardada en: {nombre_archivo}')


if __name__ == '__main__':
    print('=' * 55)
    print('  ENTRENAMIENTO CNN - TECNOFORMS')
    print('=' * 55)

    (X_train, y_train), (X_test, y_test) = cargar_datos()
    X_train_norm, X_test_norm             = normalizar(X_train, X_test)
    X_train_cnn,  X_test_cnn              = reshape_cnn(X_train_norm, X_test_norm)
    y_train_cat,  y_test_cat              = codificar_etiquetas(y_train, y_test)

    print('\n=== ARQUITECTURA DE LA CNN ===')
    modelo = construir_modelo()
    modelo.summary()

    print('\n=== ENTRENANDO ===')
    historial = entrenar_modelo(modelo, X_train_cnn, y_train_cat)

    modelo.save('modelo_digitos.keras')
    print('\nModelo guardado como: modelo_digitos.keras')

    print('\n=== GUARDANDO ARQUITECTURA XML ===')
    guardar_arquitectura_xml(modelo)

    print('=' * 55)