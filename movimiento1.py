# Importamos las librerías necesarias
import datetime

import numpy as np
import cv2
import time

# Cargamos el vídeo
camara = cv2.VideoCapture(0)


def debe_reiniciar_fondo(inicio):
    ahora = datetime.datetime.now()
    if (ahora - inicio).total_seconds() > 2:
        return True
    return False


class Detector():
    tiempo_inicio = datetime.datetime.now()
    fondo = None
    gris=None

    def administrar_fondo(self):
        if debe_reiniciar_fondo(self.tiempo_inicio) or self.fondo is None:
            self.tiempo_inicio = datetime.datetime.now()
            self.fondo = self.gris

    def obtener_imagen_nueva_gris(self,frame):
        # Convertimos a escala de grises
        self.gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Aplicamos suavizado para eliminar ruido
        self.gris = cv2.GaussianBlur(self.gris, (21, 21), 0)

    def detectar_mov(self):
        # Recorremos todos los frames
        while True:
            # Obtenemos el frame
            (grabbed, frame) = camara.read()

            # Si hemos llegado al final del vídeo salimos
            if not grabbed:
                break
            self.obtener_imagen_nueva_gris(frame)
            # Admisnistro el fondo (cada 2 segundos)
            self.administrar_fondo()

            # Calculo de la diferencia entre el fondo y el frame actual
            resta = cv2.absdiff(self.fondo, self.gris)

            # Aplicamos un umbral del 50%
            umbral = cv2.threshold(resta, 50, 100, cv2.THRESH_BINARY)[1]

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

            # Capturamos una tecla para salir
            key = cv2.waitKey(1) & 0xFF

            # Tiempo de espera para que se vea bien
            time.sleep(0.015)

            # Si ha pulsado la letra s, salimos
            if key == ord("s"):
                break

        # Liberamos la cámara y cerramos todas las ventanas
        camara.release()
        cv2.destroyAllWindows()


detetor = Detector()
detetor.detectar_mov()
