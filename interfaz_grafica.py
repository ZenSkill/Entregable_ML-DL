# Interfaz gráfica para dibujar y clasificar dígitos manuscritos
# TecnoForms - Clasificación de Dígitos Manuscritos
# By: Jhonatan Najarro

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import numpy as np
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageDraw, ImageOps
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURACIÓN
# ============================================================
TAMANO_CANVAS  = 280   # tamaño del área de dibujo
GROSOR_PINCEL  = 18    # grosor del trazo del dígito
COLOR_FONDO    = 'black'
COLOR_PINCEL   = 'white'
COLOR_PANEL    = '#1E1E2E'
COLOR_BOTON_P  = '#27AE60'
COLOR_BOTON_L  = '#E74C3C'
COLOR_TEXTO    = 'white'


# ============================================================
# CLASE PRINCIPAL
# ============================================================
class AplicacionDigitos:

    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title('Clasificador de Dígitos - TecnoForms')
        self.ventana.configure(bg=COLOR_PANEL)
        self.ventana.resizable(False, False)

        # cargar modelo entrenado
        print('Cargando modelo: modelo_digitos.keras')
        self.modelo = load_model('modelo_digitos.keras')
        print('Modelo cargado correctamente.')

        # imagen PIL donde se guarda el dibujo
        self.imagen_pil  = Image.new('L', (TAMANO_CANVAS, TAMANO_CANVAS), 0)
        self.draw        = ImageDraw.Draw(self.imagen_pil)
        self.dibujando   = False

        self._construir_interfaz()


    def _construir_interfaz(self):
        # fuentes
        fuente_titulo  = tkfont.Font(family='Helvetica', size=13, weight='bold')
        fuente_digito  = tkfont.Font(family='Helvetica', size=60, weight='bold')
        fuente_normal  = tkfont.Font(family='Helvetica', size=11)
        fuente_pequena = tkfont.Font(family='Helvetica', size=10)

        # título
        tk.Label(
            self.ventana,
            text='CLASIFICADOR DE DÍGITOS MANUSCRITOS',
            bg=COLOR_PANEL, fg=COLOR_TEXTO,
            font=fuente_titulo
        ).grid(row=0, column=0, columnspan=2, pady=(15, 5))

        tk.Label(
            self.ventana,
            text='TecnoForms — By Jhonatan Najarro',
            bg=COLOR_PANEL, fg='#888888',
            font=fuente_pequena
        ).grid(row=1, column=0, columnspan=2, pady=(0, 10))

        # canvas de dibujo
        self.canvas = tk.Canvas(
            self.ventana,
            width=TAMANO_CANVAS,
            height=TAMANO_CANVAS,
            bg=COLOR_FONDO,
            cursor='crosshair',
            highlightthickness=2,
            highlightbackground='#27AE60'
        )
        self.canvas.grid(row=2, column=0, padx=(15, 10), pady=5)

        # eventos del mouse
        self.canvas.bind('<Button-1>',        self._iniciar_dibujo)
        self.canvas.bind('<B1-Motion>',       self._dibujar)
        self.canvas.bind('<ButtonRelease-1>', self._detener_dibujo)

        # panel de resultados
        panel = tk.Frame(self.ventana, bg=COLOR_PANEL, width=220)
        panel.grid(row=2, column=1, padx=(0, 15), pady=5, sticky='n')

        tk.Label(
            panel,
            text='Dígito detectado',
            bg=COLOR_PANEL, fg='#888888',
            font=fuente_pequena
        ).pack(pady=(10, 0))

        self.lbl_digito = tk.Label(
            panel,
            text='?',
            bg=COLOR_PANEL, fg='#27AE60',
            font=fuente_digito
        )
        self.lbl_digito.pack()

        tk.Label(
            panel,
            text='Confianza',
            bg=COLOR_PANEL, fg='#888888',
            font=fuente_pequena
        ).pack(pady=(5, 0))

        self.lbl_confianza = tk.Label(
            panel,
            text='—',
            bg=COLOR_PANEL, fg=COLOR_TEXTO,
            font=fuente_normal
        )
        self.lbl_confianza.pack()

        # separador
        tk.Frame(panel, bg='#444444', height=1, width=180).pack(pady=10)

        tk.Label(
            panel,
            text='Top 3 predicciones',
            bg=COLOR_PANEL, fg='#888888',
            font=fuente_pequena
        ).pack()

        # etiquetas top 3
        self.lbl_top = []
        for _ in range(3):
            lbl = tk.Label(
                panel,
                text='—',
                bg=COLOR_PANEL, fg=COLOR_TEXTO,
                font=fuente_pequena
            )
            lbl.pack(pady=2)
            self.lbl_top.append(lbl)

        # separador
        tk.Frame(panel, bg='#444444', height=1, width=180).pack(pady=10)

        # instrucción
        tk.Label(
            panel,
            text='Dibuja un dígito en el\nrecuadro y presiona\nPREDECIR',
            bg=COLOR_PANEL, fg='#888888',
            font=fuente_pequena,
            justify='center'
        ).pack()

        # botones
        frame_botones = tk.Frame(self.ventana, bg=COLOR_PANEL)
        frame_botones.grid(row=3, column=0, columnspan=2, pady=15)

        tk.Button(
            frame_botones,
            text='PREDECIR',
            bg=COLOR_BOTON_P, fg='white',
            font=tkfont.Font(family='Helvetica', size=12, weight='bold'),
            width=12, height=1,
            relief='flat', cursor='hand2',
            command=self._predecir
        ).pack(side='left', padx=10)

        tk.Button(
            frame_botones,
            text='LIMPIAR',
            bg=COLOR_BOTON_L, fg='white',
            font=tkfont.Font(family='Helvetica', size=12, weight='bold'),
            width=12, height=1,
            relief='flat', cursor='hand2',
            command=self._limpiar
        ).pack(side='left', padx=10)


    def _iniciar_dibujo(self, evento):
        self.dibujando  = True
        self.x_anterior = evento.x
        self.y_anterior = evento.y


    def _dibujar(self, evento):
        if not self.dibujando:
            return

        x, y = evento.x, evento.y

        # dibujar en canvas visual
        self.canvas.create_oval(
            x - GROSOR_PINCEL, y - GROSOR_PINCEL,
            x + GROSOR_PINCEL, y + GROSOR_PINCEL,
            fill=COLOR_PINCEL, outline=COLOR_PINCEL
        )

        # dibujar en imagen PIL (para el modelo)
        self.draw.ellipse(
            [x - GROSOR_PINCEL, y - GROSOR_PINCEL,
             x + GROSOR_PINCEL, y + GROSOR_PINCEL],
            fill=255
        )

        self.x_anterior = x
        self.y_anterior = y


    def _detener_dibujo(self, evento):
        self.dibujando = False


    def _predecir(self):
        # redimensionar imagen a 28x28 como MNIST
        img_redim = self.imagen_pil.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img_redim) / 255.0
        img_input = img_array.reshape(1, 28, 28, 1)

        # predecir
        predicciones = self.modelo.predict(img_input, verbose=0)[0]
        digito       = np.argmax(predicciones)
        confianza    = predicciones[digito] * 100

        # top 3 predicciones
        top3_indices = np.argsort(predicciones)[::-1][:3]

        # actualizar interfaz
        self.lbl_digito.config(text=str(digito))
        self.lbl_confianza.config(text=f'{confianza:.2f}%')

        for i, idx in enumerate(top3_indices):
            self.lbl_top[i].config(
                text=f'Dígito {idx}  →  {predicciones[idx]*100:.2f}%'
            )


    def _limpiar(self):
        # limpiar canvas visual
        self.canvas.delete('all')

        # limpiar imagen PIL
        self.imagen_pil = Image.new('L', (TAMANO_CANVAS, TAMANO_CANVAS), 0)
        self.draw       = ImageDraw.Draw(self.imagen_pil)

        # resetear resultados
        self.lbl_digito.config(text='?')
        self.lbl_confianza.config(text='—')
        for lbl in self.lbl_top:
            lbl.config(text='—')


# ============================================================
# INICIO DE LA APLICACIÓN
# ============================================================
if __name__ == '__main__':
    ventana = tk.Tk()
    app     = AplicacionDigitos(ventana)
    ventana.mainloop()