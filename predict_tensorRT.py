from ultralytics import YOLO

model_path = 'path to model'
image_path = 'path to image' #can also accept video
model = YOLO(model_path)  # load a custom model
model.predict(image_path, save=True,
              conf=0.5, # object confidence threshold for detection
              device= 0, # device to run on, i.e. cuda device=0/1/2/3 or device=cpu
              project='output directory',
              name='output name',
              line_thickness=3)