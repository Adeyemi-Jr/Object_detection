import os
import torch
import cv2
from ultralytics import YOLO
import sys
sys.path.insert(0, '/home/adeyemi/Documents/mypythonlibrary')
from myfunctions import images_to_video
import numpy as np
from distutils.dir_util import copy_tree
from preprocess import *
from natsort import natsorted
import yaml
import time


###############################################
#        Function Definition - START
###############################################

def blur_and_downsample_path(images_path, output_path):

    training_files = natsorted(os.listdir(images_path))

    for file in training_files:
        file_ = file[4:]
        print(file)
        #img = cv2.imread(os.path.join(images_path, file))
        img = Image.open(os.path.join(images_path, file))
        processor = BlurAndDownsample(img, blur, downsam_factor)

        img_processed = processor.process()

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        final_path = os.path.join(output_path, file_)
        #cv2.imwrite(final_path, img_processed)
        img_processed.save(final_path)


###############################################
#        Function Definition - END
###############################################


'''
images_path_input  = '../../Velpsis_data/Highway/Truth/'
images_path_output = '../../Velpsis_data/Highway/predictions'
model_path = os.path.join('.', 'runs', 'detect', 'train5', 'weights', 'last.pt')


# Load the model
model = YOLO(model_path)  # load a custom model

model.predict(images_path_input,save = True, conf=0.5)



#convert images to video
images_to_video(images_path_output, 'Towson')
'''



#######################################################
#                   Predictions                       #
#######################################################
#datasets= ['Highway', 'Towson']
datasets= ['Highway','Towson']



for dataset in datasets:
    #downsample the test images

    blur_rate = [round(num, 2) for num in list(np.arange(1, 3 + 0.25, 0.25))]

    #blur = 5
    downsam_factor = 0.33

    for i in blur_rate:
        blur = i
        blur_and_downsample_str = 'blur_' + str(blur) + '_pixels_' + 'downsample_factor_' + str(downsam_factor)


        ############ preprocess traninng data and train the model
        #training_img_path = '../../Object_detection/data/JPEGImages'
        test_img_path = '../data/raw/' + dataset + '/Truth'

        output_downsampled_path = '../data/processed/' + blur_and_downsample_str + '/test/' + dataset
        blur_and_downsample_path(test_img_path, output_downsampled_path +'/images')

        if dataset == 'Highway':
            #copy test label
            test_label_source_dir = '../data/raw/'+dataset+'/labels'
            #train_label_source_dir = '../../Object_detection/data/labels'

            test_label_destination_dir = output_downsampled_path +'/labels'

            copy_tree(test_label_source_dir,test_label_destination_dir)
            A = 1


        #####################################################
        #                        Predict                    #
        #####################################################

        model_path = os.path.join('..', 'models', blur_and_downsample_str, 'weights', 'last.pt')


        # Load the model
        model = YOLO(model_path)  # load a custom model

        predict_output_path = os.path.join('../data/processed/', blur_and_downsample_str, 'test/'+dataset+'/prediction')


        #if not os.path.exists(predict_output_path):
        #    os.makedirs(predict_output_path)

        #model.predict(predict_output_path,save = True, conf=0.5)
        start_time = time.time()
        model.predict(output_downsampled_path +'/images',save = True,
                      conf=0.5,
                      project = os.path.join('../data/processed/', blur_and_downsample_str, 'test/'+dataset+'/prediction'),
                      name = dataset,
                      line_thickness = 1
                      )
        elapsed_time = time.time() - start_time

        print( 'inference time for {} is......... {}(s)'.format(blur_and_downsample_str, round(elapsed_time,2)))

        # convert images to video
        images_to_video(predict_output_path+ '/'+ dataset, dataset + '_'+blur_and_downsample_str)
        A = 1
        #sys.exit()

        #copy_tree('runs/detect/predict', predict_output_path)
        #shutil.rmtree('runs/detect/predict')

        #convert images to video
        #images_to_video(predict_output_path, dataset)
        #sys.exit()


'''
        #####################################################
        #                        Predict                    #
        #####################################################

        model_path = os.path.join('..', 'models', blur_and_downsample_str, 'weights', 'last.pt')


        # Load the model
        model = YOLO(model_path)  # load a custom model

        predict_output_path = os.path.join('../data/processed/', blur_and_downsample_str, 'prediction', dataset)



        data_path = os.path.join('/home/adeyemi/Documents/Projects/Vlepsis/Velpsis_data/data/processed',blur_and_downsample_str)
        # edit yaml file - change the path only
        with open('config_test.yaml') as f:
            list_doc = yaml.safe_load(f)

        list_doc['path'] = data_path

        with open('config_test.yaml', 'w') as f:
            yaml.dump(list_doc, f)

        #if not os.path.exists(predict_output_path):
        #    os.makedirs(predict_output_path)

        #model.predict(predict_output_path,save = True, conf=0.5)
        metrics = model.val(data = 'config_test.yaml')


        # convert images to video
        images_to_video(predict_output_path, dataset + '_'+blur_and_downsample_str)
        #sys.exit()

        #copy_tree('runs/detect/predict', predict_output_path)
        #shutil.rmtree('runs/detect/predict')

        #convert images to video
        #images_to_video(predict_output_path, dataset)
        #sys.exit()
'''



