from django.contrib import admin

# Register your models here.
from detector_admin.models import Movimiento


class MovimientoAdmin(admin.ModelAdmin):
    list_display = ['tiempo_ini', 'tiempo_fin']


admin.site.register(Movimiento, MovimientoAdmin)
