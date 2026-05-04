# graficos.py
# Genera visualizaciones generales del dataset
# TecnoForms - Clasificación de Dígitos Manuscritos
# By: Jhonatan Najarro

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.makedirs('graficos', exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from preprocesamiento import cargar_datos

carpeta = 'graficos'


if __name__ == '__main__':
    print('=' * 55)
    print('  GENERANDO GRÁFICOS - TECNOFORMS')
    print('=' * 55)

    (X_train, y_train), (X_test, y_test) = cargar_datos()

    # muestras del dataset
    print(f'\nGenerando: {carpeta}/muestras_mnist.png')
    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_train[i], cmap='gray')
        plt.title(f'Dígito: {y_train[i]}')
        plt.axis('off')
    plt.suptitle('Muestras del Dataset MNIST - TecnoForms', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{carpeta}/muestras_mnist.png', dpi=120)
    plt.show()

    # distribución de clases
    print(f'Generando: {carpeta}/distribucion_clases.png')
    plt.figure(figsize=(8, 4))
    clases, conteos = np.unique(y_train, return_counts=True)
    plt.bar(clases, conteos, color='#2980B9', edgecolor='black')
    plt.title('Distribución de Clases - TecnoForms', fontsize=13)
    plt.xlabel('Dígito')
    plt.ylabel('Cantidad de imágenes')
    plt.xticks(range(10))
    plt.tight_layout()
    plt.savefig(f'{carpeta}/distribucion_clases.png', dpi=120)
    plt.show()

    print(f'\nGráficos guardados en: {carpeta}/')
    print(f'  {carpeta}/muestras_mnist.png')
    print(f'  {carpeta}/distribucion_clases.png')

    print('=' * 55)