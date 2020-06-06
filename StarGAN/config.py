
from tensorflow.python.keras.initializers import RandomNormal
img_width=240
img_height=240
img_channel=1
img_shape=(240,240,1)
batch_size=1
conv_init = RandomNormal(0, 0.02)
class_num=3