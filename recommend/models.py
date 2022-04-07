from django.db import models
from django.utils import timezone

# Create your models here.
class Userinfo(models.Model):
	name = models.CharField(max_length=256)
	token = models.CharField(max_length=256)

class Uselog(models.Model):
	id = models.AutoField(primary_key=True)
	userid = models.CharField(max_length=32)
	targetid = models.CharField(max_length=32)
	clicked_item = models.CharField(max_length=64)
	# dt = models.DateTimeField(default=timezone.now)