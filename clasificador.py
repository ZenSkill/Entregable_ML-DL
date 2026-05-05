# Carga el modelo entrenado y clasifica dígitos nuevos
# By: Jhonatan Najarro

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.makedirs('graficos', exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

from preprocesamiento import cargar_datos, normalizar, reshape_cnn, codificar_etiquetas

carpeta = 'graficos'


if __name__ == '__main__':
    print('=' * 55)
    print('  CLASIFICADOR DE DÍGITOS - TECNOFORMS')
    print('=' * 55)

    (X_train, y_train), (X_test, y_test) = cargar_datos()
    X_train_norm, X_test_norm             = normalizar(X_train, X_test)
    X_train_cnn,  X_test_cnn              = reshape_cnn(X_train_norm, X_test_norm)
    y_train_cat,  y_test_cat              = codificar_etiquetas(y_train, y_test)

    print('\nCargando modelo: modelo_digitos.keras')
    modelo = load_model('modelo_digitos.keras')
    print('Modelo cargado correctamente.')

    print('\n=== EVALUACIÓN DEL MODELO ===')
    perdida, precision = modelo.evaluate(X_test_cnn, y_test_cat, verbose=0)
    print(f'Precisión en test : {precision:.4f} ({precision*100:.2f}%)')
    print(f'Pérdida en test   : {perdida:.4f}')

    y_pred = np.argmax(modelo.predict(X_test_cnn, verbose=0), axis=1)

    print('\nReporte de clasificación:')
    print(classification_report(y_test, y_pred))

    # matriz de confusión
    print('\n=== MATRIZ DE CONFUSIÓN ===')
    matriz = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
                xticklabels=[str(i) for i in range(10)],
                yticklabels=[str(i) for i in range(10)])
    plt.title('Matriz de Confusión - CNN - TecnoForms', fontsize=13)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(f'{carpeta}/matriz_confusion.png', dpi=120)
    plt.show()
    print(f'Gráfico guardado: {carpeta}/matriz_confusion.png')

    # predicciones visuales
    print('\n=== PREDICCIONES VISUALES ===')
    plt.figure(figsize=(12, 4))
    for i in range(12):
        plt.subplot(2, 6, i + 1)
        plt.imshow(X_test[i], cmap='gray')
        pred  = y_pred[i]
        real  = y_test[i]
        color = 'green' if pred == real else 'red'
        plt.title(f'P:{pred} R:{real}', color=color, fontsize=9)
        plt.axis('off')
    plt.suptitle('Predicciones CNN (verde=correcto, rojo=error)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{carpeta}/predicciones_visuales.png', dpi=120)
    plt.show()
    print(f'Gráfico guardado: {carpeta}/predicciones_visuales.png')

    print('=' * 55)