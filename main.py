from preprocess import *
import os
from natsort import natsorted
from PIL import Image, ImageFilter
from ultralytics import YOLO
from distutils.dir_util import copy_tree
import sys
sys.path.insert(0, '/home/adeyemi/Documents/mypythonlibrary')
from myfunctions import images_to_video
import shutil
import yaml
from ultralytics import YOLO
import time






blur_rate = [round(num,2) for num in list(np.arange(1,3+0.25,0.25))]

#blur = 5
downsam_factor = 0.33

for i in blur_rate:
    blur = i
    blur_and_downsample_str = 'blur_'+str(blur)+'_pixels_'+'downsample_factor_'+str(downsam_factor)

    def blur_and_downsample_path(images_path, output_path):

        training_files = natsorted(os.listdir(images_path))

        for file in training_files:

            #img = cv2.imread(os.path.join(images_path, file))
            img = Image.open(os.path.join(images_path, file))
            processor = BlurAndDownsample(img, blur, downsam_factor)

            img_processed = processor.process()

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            final_path = os.path.join(output_path, file)
            #cv2.imwrite(final_path, img_processed)
            img_processed.save(final_path)

    ####################################################################
    #                      Training - Preprocess                       #
    ####################################################################

    ############ preprocess traninng data and train the model
    #training_img_path = '../../Object_detection/data/JPEGImages'
    training_img_path = '../../Object_detection/data/train'

    output_downsampled_training_path = '../data/processed/' + blur_and_downsample_str
    blur_and_downsample_path(training_img_path+ '/images', output_downsampled_training_path +'/train/images')

    #copy training label
    train_label_source_dir = '../../Object_detection/data/train/labels'
    #train_label_source_dir = '../../Object_detection/data/labels'

    train_label_destination_dir = output_downsampled_training_path +'/train/labels'

    copy_tree(train_label_source_dir,train_label_destination_dir)






    val_img_path = '../../Object_detection/data/val'
    output_downsampled_val_path = '../data/processed/'+ blur_and_downsample_str
    blur_and_downsample_path(val_img_path+ '/images', output_downsampled_val_path +'/val/images')

    val_label_source_dir = '../../Object_detection/data/val/labels'
    val_label_destination_dir = output_downsampled_val_path +'/val/labels'

    copy_tree(val_label_source_dir,val_label_destination_dir)


    ###############################################################
    #                      Training - Model                       #
    ###############################################################
    model_path = os.path.join('/home/adeyemi/Documents/Projects/Vlepsis/Velpsis_data/data/processed',blur_and_downsample_str)
    #edit yaml file - change the path only
    with open('config.yaml') as f:
        list_doc = yaml.safe_load(f)

    list_doc['path'] = model_path

    with open('config.yaml', 'w') as f:
        yaml.dump(list_doc,f)



    #load a model
    start = time.time()
    model = YOLO("yolov8n.yaml") #build new model from scratch
    model = YOLO("yolov8n.pt") #pretrained model


    #Use model
    results = model.train(data='config.yaml',epochs = 100,
                          scale = 0.5,
                          degrees = 0.5,
                          translate = 0.5,
                          shear = 0.5,
                          flipud = 0.5,
                          fliplr = 0.5,
                          mosaic= 1,
                          project = '../models',
                          name = blur_and_downsample_str)
    end = time.time()

    print('Training Finished....: '+ str(end-start))



















'''
#######################################################
#                   Predictions                       #
#######################################################
datasets= ['Highway', 'Towson']


for dataset in datasets:
    #downsample the test images
    test_img_path = '../data/raw/' + dataset + '/Truth'
    output_downsampled_test_path = os.path.join('../data/processed/', blur_and_downsample_str, 'test', dataset)
    blur_and_downsample_path(test_img_path, output_downsampled_test_path)



    #####################################################
    #                        Predict                    #
    #####################################################

    model_path = os.path.join('..', 'models', blur_and_downsample_str, 'weights', 'last.pt')


    # Load the model
    model = YOLO(model_path)  # load a custom model

    predict_output_path = os.path.join('../data/processed/', blur_and_downsample_str, 'prediction', dataset)


    #if not os.path.exists(predict_output_path):
    #    os.makedirs(predict_output_path)

    #model.predict(predict_output_path,save = True, conf=0.5)
    model.predict(output_downsampled_test_path,save = True,
                  conf=0.5,
                  project = os.path.join('../data/processed/', blur_and_downsample_str, 'prediction'),
                  name = dataset
                  )


    # convert images to video
    images_to_video(predict_output_path, dataset + '_'+blur_and_downsample_str)
    #sys.exit()

    #copy_tree('runs/detect/predict', predict_output_path)
    #shutil.rmtree('runs/detect/predict')

    #convert images to video
    #images_to_video(predict_output_path, dataset)
    #sys.exit()
'''




