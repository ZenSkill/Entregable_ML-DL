
# Clasificación de Dígitos Manuscritos

![Python](https://img.shields.io/badge/Python-3.10-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-green) ![Dataset](https://img.shields.io/badge/Dataset-MNIST-lightgrey)

Este proyecto fue desarrollado para preprocesar, entrenar y clasificar dígitos manuscritos en tiempo real usando Python, TensorFlow/Keras y Scikit-Learn. Permite entrenar una Red Neuronal Convolucional (CNN) con el dataset MNIST, comparar su rendimiento frente a modelos clásicos (KNN, SVM, K-Means) y generar evidencias visuales automáticamente.

---

## Descripción del Proyecto

**Empresa:** TecnoForms
**Objetivo:** Automatizar el reconocimiento de dígitos escritos a mano en formularios físicos para reducir errores de lectura manual.
**Alumno:** Jhonatan Najarro
**Institución:** SENATI

El sistema carga el dataset MNIST (70,000 imágenes de 28×28 píxeles), entrena una CNN con arquitectura multicapa, evalúa el modelo entrenado y guarda las gráficas de rendimiento en la carpeta `graficos/`.

---

## Estructura del Proyecto

Entregable_ML-DL/
│

├── preprocesamiento.py # Carga, normaliza y prepara el dataset MNIST

├── entrenamiento.py # Define, entrena y guarda el modelo CNN

├── clasificador.py # Evalúa el modelo y genera visualizaciones

├── graficos.py # Genera gráficas generales del dataset

├── comparar_modelos.py # Compara CNN vs KNN vs SVM vs K-Means

│

├── modelo_digitos.keras # Modelo entrenado (generado al ejecutar)

├── arquitectura_cnn.xml # Arquitectura de la CNN en formato XML

│

└── graficos/ # Carpeta con todas las evidencias generadas

├── distribucion_clases.png

├── muestras_dataset.png

├── curva_accuracy.png

├── curva_perdida.png

├── matriz_confusion.png

├── predicciones_visuales.png

└── comparacion_modelos.png
