from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import cv2
import argparse
import os

parser = argparse.ArgumentParser(description='image url.')
parser.add_argument('-u', '--url', type=str, help='url of image to parse')
parser.add_argument('-f', '--folder', type=str, help='folder of image')
args = parser.parse_args()

img = load_img(os.path.join(args.folder, args.url))
data = img_to_array(img)
samples = expand_dims(data, 0)
datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
it = datagen.flow(samples, batch_size=1)
for i in range(10):
	batch = it.next()
	image = batch[0].astype('uint8')
	cv2.imwrite(args.folder+"\s%d.jpg" % i, image)

print("done")