from django.contrib import admin

# Register your models here.
from recommend.models import Userinfo, Uselog
admin.site.register(Userinfo)
admin.site.register(Uselog)