# instant-model viewer by TFujishima

import keras
from keras.models import Model
from keras.utils.vis_utils import model_to_dot
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

from keras.layers import Dense
from keras.layers import Input
from keras.layers.merge import add


# Viewing function attached with Model class
class VModel(Model):
	def __init__(self,inputs,outputs):
		super().__init__(inputs,outputs)
		self.pnb = model_to_dot(self).create_png()
		self.png = Image.open(BytesIO(self.pnb))
	def show(self):
		plt.imshow(self.png)
		plt.show()
	def save(self,FilePath):
		self.png.save(FilePath)
		print('Model-image is saved as ' + FilePath + '.')




def TestRun():
	inp = Input((2,))
	den = Dense(3,activation='relu')(inp)
	den = Dense(2,activation='sigmoid')(den)
	s = VModel(inp,den)
	s.show()

if __name__ == '__main__':
	TestRun()
