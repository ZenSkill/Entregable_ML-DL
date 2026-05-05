# Comparar CNN vs KNN vs SVM e incluye datos ficticios y K-Means
# By: Jhonatan Najarro

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.makedirs('graficos', exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

from preprocesamiento import cargar_datos, normalizar, reshape_cnn, codificar_etiquetas, aplanar

carpeta = 'graficos'


if __name__ == '__main__':
    print('=' * 55)
    print('  COMPARACIÓN DE MODELOS - TECNOFORMS')
    print('=' * 55)

    (X_train, y_train), (X_test, y_test) = cargar_datos()
    X_train_norm, X_test_norm             = normalizar(X_train, X_test)
    X_train_cnn,  X_test_cnn              = reshape_cnn(X_train_norm, X_test_norm)
    y_train_cat,  y_test_cat              = codificar_etiquetas(y_train, y_test)
    X_train_flat, X_test_flat             = aplanar(X_train_norm, X_test_norm)

    X_sub = X_train_flat[:5000]
    y_sub = y_train[:5000]

    # CNN
    print('\n=== CNN (datos reales) ===')
    modelo_cnn       = load_model('modelo_digitos.keras')
    _, precision_cnn = modelo_cnn.evaluate(X_test_cnn, y_test_cat, verbose=0)
    print(f'Precisión CNN : {precision_cnn*100:.2f}%')

    # KNN
    print('\n=== KNN (datos reales) ===')
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_sub, y_sub)
    precision_knn = accuracy_score(y_test[:1000], knn.predict(X_test_flat[:1000]))
    print(f'Precisión KNN (k=3) : {precision_knn*100:.2f}%')

    # SVM
    print('\n=== SVM (datos reales) ===')
    svm = SVC(kernel='rbf', C=1, gamma='scale')
    svm.fit(X_sub, y_sub)
    precision_svm = accuracy_score(y_test[:1000], svm.predict(X_test_flat[:1000]))
    print(f'Precisión SVM (rbf) : {precision_svm*100:.2f}%')

    # datos ficticios
    print('\n=== DATOS FICTICIOS ===')
    np.random.seed(42)
    X_fic        = np.random.rand(500, 784)
    y_fic        = np.random.randint(0, 10, 500)
    scaler       = StandardScaler()
    X_fic_scaled = scaler.fit_transform(X_fic)

    knn_fic = KNeighborsClassifier(n_neighbors=5)
    scores  = cross_val_score(knn_fic, X_fic_scaled, y_fic, cv=5)
    print(f'KNN datos ficticios (5-fold CV) : {scores.mean():.4f}')

    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    kmeans.fit(X_fic_scaled)
    print(f'K-Means 10 clusters | Inercia  : {kmeans.inertia_:.2f}')

    # gráfico comparación
    print('\n=== COMPARACIÓN FINAL ===')
    modelos     = ['CNN', 'KNN (k=3)', 'SVM (rbf)']
    precisiones = [precision_cnn, precision_knn, precision_svm]

    for m, p in zip(modelos, precisiones):
        print(f'  {m:<15} → {p*100:.2f}%')

    plt.figure(figsize=(7, 4))
    colores = ['#27AE60', '#2980B9', '#E67E22']
    plt.bar(modelos, [p*100 for p in precisiones],
            color=colores, edgecolor='black')
    plt.title('Comparación de Precisión por Modelo - TecnoForms', fontsize=13)
    plt.ylabel('Precisión (%)')
    plt.ylim(80, 100)
    for i, p in enumerate(precisiones):
        plt.text(i, p*100 + 0.2, f'{p*100:.2f}%', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{carpeta}/comparacion_modelos.png', dpi=120)
    plt.show()
    print(f'Gráfico guardado: {carpeta}/comparacion_modelos.png')

    print('=' * 55)