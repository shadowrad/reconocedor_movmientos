# Importamos las librerías necesarias de django


import sys
# Django specific settings
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "detector_acciones.settings")
import django

django.setup()

# Your application specific imports
from detector_admin.models import Movimiento

import datetime
import numpy as np
import cv2
import time


def debe_reiniciar_inicio(inicio):
    ahora = datetime.datetime.now()
    if (ahora - inicio).total_seconds() > 0.3:
        return True
    return False


def verfifcar_promedio(array, promedio):
    cantidad = len(array)
    acum = 0
    for i in array:
        acum += i[0]
    print(acum / cantidad)
    print(promedio)


def guardar_datos(lista2d):
    with open("datos_guardados/datos.json", "w+") as f:
        Movimiento.objects.all().delete()
        for lista in lista2d:
            if len(str(lista['frame'])) > 0:
                mov = Movimiento()
                mov.crear(features=str(lista['frame']),tiempo_ini=str(lista['tiempo_ini']),tiempo_fin=str(lista['tiempo_fin']))
                mov.save()


class Detector(object):
    diferencias = {'frames': [], 'tiempo': []}
    diferencias_en_rango = []
    items = []
    tiempo_inicio = datetime.datetime.now()
    video_inicio = 0
    anterior = None
    gris = None
    camara = cv2.VideoCapture('videos/chicos_estudiando.mp4')
    video_inicio = datetime.datetime.now()

    # camara = cv2.VideoCapture(0)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')

    def get_prom_diferencias(self, dif):
        promedio_cols_tiempo = np.mean(dif, axis=0, dtype=np.float64)
        if promedio_cols_tiempo.max() > 0:
            self.diferencias['frames'].append(promedio_cols_tiempo)
            self.diferencias['tiempo'].append(datetime.datetime.now() - self.video_inicio)

    def set_prom_rango(self):
        if len(self.diferencias['frames']) > 0:
            promedio_cols_tiempo = np.mean(self.diferencias['frames'], axis=0, dtype=np.float64)
            dato = {'frame': promedio_cols_tiempo, 'tiempo_ini': min(self.diferencias['tiempo']),
                    'tiempo_fin': max(self.diferencias['tiempo'])}
            self.diferencias_en_rango.append(dato)
            # verfifcar_promedio(self.diferencias, promedio_cols_tiempo[0])
            self.diferencias['frames'] = []
            self.diferencias['tiempo'] = []

    def administrar_imagen_inicio(self):
        if debe_reiniciar_inicio(self.tiempo_inicio) or self.anterior is None:
            self.tiempo_inicio = datetime.datetime.now()
            self.anterior = self.gris
            self.set_prom_rango()

    def obtener_imagen_nueva_gris(self, frame):
        # Convertimos a escala de grises
        self.gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Aplicamos suavizado para eliminar ruido
        self.gris = cv2.GaussianBlur(self.gris, (21, 21), 0)

    def detectar_mov(self):
        out = None
        # Recorremos todos los frames
        while True:
            # Obtenemos el frame
            (grabbed, frame) = self.camara.read()

            # Si hemos llegado al final del vídeo salimos
            if not grabbed:
                break

            self.obtener_imagen_nueva_gris(frame)
            # Admisnistro el fondo (cada 2 segundos)
            self.administrar_imagen_inicio()

            # Calculo de la diferencia entre el fondo y el frame actual
            resta = cv2.absdiff(self.anterior, self.gris)

            # Aplicamos un umbral del 50%
            umbral = cv2.threshold(resta, 40, 255, cv2.THRESH_BINARY)[1]
            # Dilatamos el umbral para tapar agujeros
            umbral = cv2.dilate(umbral, None, iterations=2)

            # Copiamos el umbral para detectar los contornos
            contornosimg = umbral.copy()

            # Buscamos contorno en la imagen
            im, contornos, hierarchy = cv2.findContours(contornosimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Recorremos todos los contornos encontrados
            for c in contornos:
                # Eliminamos los contornos más pequeños
                if cv2.contourArea(c) < 500:
                    continue

                # Obtenemos el bounds del contorno, el rectángulo mayor que engloba al contorno
                (x, y, ancho, alto) = cv2.boundingRect(c)

                # Dibujo el rectangulo
                cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (0, 0, 255), 2)

            # Mostramos las imágenes de la cámara, el umbral y la resta
            cv2.imshow("Camara", frame)

            cv2.imshow("Umbral", umbral)

            cv2.imshow("Diferencia", resta)
            cv2.imshow("Contorno", contornosimg)

            self.get_prom_diferencias(resta)

            if out is None:
                height, width, layers = frame.shape

                out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, (width, height))
            out.write(frame)

            # Capturamos una tecla para salir
            key = cv2.waitKey(1) & 0xFF

            # Tiempo de espera para que se vea bien
            time.sleep(0.015)

            # Si ha pulsado la letra s, salimos
            if key == ord("s"):
                break

        # Liberamos la cámara y cerramos todas las ventanas
        self.camara.release()
        out.release()
        cv2.destroyAllWindows()
        guardar_datos(self.diferencias_en_rango)


detetor = Detector()
detetor.detectar_mov()
