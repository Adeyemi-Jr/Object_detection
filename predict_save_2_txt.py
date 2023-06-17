import os
import sys

import cv2
from natsort import natsorted

from ultralytics import YOLO


'''
data_type = 'Towson'
#image_path = '../data/raw/Highway/Truth/out-000001.png'
model = YOLO('../models/blur_1.0_pixels_downsample_factor_0.33/weights/best.pt')
#predictions = model(image_path, save_txt=None)

primary_path = '../data/raw/'+data_type+'/Truth'
filenames = natsorted(os.listdir(primary_path))


#make directory for output if it doesnt exisit
output_path = '../data/results/' + data_type
if not os.path.exists(output_path):
    os.makedirs(output_path)

#filenames = filenames[:3]
for filename in filenames:
    image_path = os.path.join(primary_path, filename)
    #predictions = model.predict(image_path, save_txt=True,project = output_path, name = filename)
    predictions = model.predict(image_path)


    output_path_file = os.path.join(output_path, filename)
    output_path_file = output_path_file[:-3]+'txt'
    with open(output_path_file, '+w') as file:

          for idx, prediction in enumerate(predictions[0].boxes.xywh): # change final attribute to desired box format
              cls = int(predictions[0].boxes.cls[idx].item())
              confidence = predictions[0].boxes.conf[idx].item()

              # Get the class probability/confidence - only get class probability that is >50%
              if confidence > 0.5:
                # Write line to file in YOLO label format : cls x y w h
                file.write(f"{cls} {confidence} {prediction[0].item()} {prediction[1].item()} {prediction[2].item()} {prediction[3].item()}\n")
              else:
                  continue
'''
#sys.exit()





image_path = '../data/raw/Highway/Truth/out-000199.png'  # Specify the path to your image file
txt_file_path = '../data/results/Highway/out-000199.txt'  # Specify the path to your YOLO format labels file
#txt_file_path = '../data/results/Highway/out-000001.png/labels/out-000001.txt'  # Specify the path to your YOLO format labels file



# Load the image
image = cv2.imread(image_path)

# Read the labels from the text file
with open(txt_file_path, 'r') as file:
    lines = file.readlines()

# Overlay the labels on the image
for line in lines:
    label_data = line.split(" ")
    class_id = int(label_data[0])
    confidence = float(label_data[1])
    x, y, w, h = map(float, label_data[2:])


    # Convert coordinates to integers
    x1, y1 = int(x - w / 2), int(y - h / 2)
    x2, y2 = int(x + w / 2), int(y + h / 2)

    # Draw bounding box and label
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f'Class {class_id} (Confidence: {confidence:.2f})'
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the image with labels overlay
cv2.imwrite('image_test.jpg', image)

#cv2.imshow('Image with Labels', image)
#cv2.waitKey(0)
cv2.destroyAllWindows()
