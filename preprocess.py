import numpy as np
import os
from PIL import Image, ImageFilter
import cv2



class BlurAndDownsample():

    def __init__(self, img, blur_factor= 2, downsample_factor = 0.33):
        self.img = img
        self.blur_factor = blur_factor
        self.downsample_factor = 1-downsample_factor


    def process(self):

        # apply pixel blur
        #kernel_size = (2*self.blur_factor)+1

        #img_blur = cv2.GaussianBlur(self.img,(kernel_size,kernel_size),0)
        blurred_img = self.img.filter(ImageFilter.GaussianBlur(radius=self.blur_factor))

        #Downsample
        #width = int(img_blur.shape[1]*self.downsample_factor)
        #height = int(img_blur.shape[0]*self.downsample_factor)

        # Downsample image by 33%
        width, height = blurred_img.size
        img_downsampled = blurred_img.resize((int(width * self.downsample_factor), int(height * self.downsample_factor)))

        #use openCV to resize the image

        #img_downsampled = cv2.resize(img_blur, (width, height), interpolation = cv2.INTER_AREA)

        return img_downsampled


'''
img = Image.open('/home/adeyemi/Documents/Projects/Vlepsis/Velpsis_data/data/raw/Highway/Truth/out-000001.png')

processor = BlurAndDownsample(img)

img_processed = processor.process()

img_processed.save('output.png')
'''


