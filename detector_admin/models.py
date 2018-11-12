from django.db import models


# Create your models here.
class Movimiento(models.Model):
    features = models.TextField()
    tiempo_ini = models.TimeField(verbose_name='inicio', null=True)
    tiempo_fin = models.TimeField(verbose_name='fin', null=True)

    def crear(self, features, tiempo_ini, tiempo_fin):
        self.features = features
        self.tiempo_ini = tiempo_ini
        self.tiempo_fin = tiempo_fin

    def __str__(self):
        return self.features
