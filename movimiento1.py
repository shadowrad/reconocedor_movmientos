# Importamos las librerías necesarias
import datetime

import numpy as np
import cv2
import time

# Cargamos el vídeo
camara = cv2.VideoCapture('videos/chicos_estudiando.mp4')


# camara = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')


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
    print(acum/cantidad)
    print(promedio)


class Detector(object):
    diferencias = []
    diferencias_en_rango = []
    items = []
    tiempo_inicio = datetime.datetime.now()
    anterior = None
    gris = None

    def get_prom_diferencias(self, dif):
        promedio_cols_tiempo = np.mean(dif, axis=0, dtype=np.float64)
        if promedio_cols_tiempo.max() > 0:
            self.diferencias.append(promedio_cols_tiempo)

    def set_prom_rango(self):
        if len(self.diferencias) > 0:
            if len(self.diferencias) == 2:
                pass
            promedio_cols_tiempo = np.mean(self.diferencias, axis=0, dtype=np.float64)
            self.diferencias_en_rango.append(promedio_cols_tiempo)
            verfifcar_promedio(self.diferencias, promedio_cols_tiempo[0])
            self.diferencias = []

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
            (grabbed, frame) = camara.read()

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
        camara.release()
        out.release()
        for a in self.diferencias:
            print(a)
        cv2.destroyAllWindows()


detetor = Detector()
detetor.detectar_mov()
