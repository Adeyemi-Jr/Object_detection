from ultralytics.yolo.utils.benchmarks import benchmark
from ultralytics import YOLO
import numpy as np




#model_path = '../models/blur_1.5_pixels_downsample_factor_0.33/weights/best.pt'
blur_rate = [round(num,2) for num in list(np.arange(1,3+0.25,0.25))]
downsam_factor = 0.33

for i in blur_rate:
    blur = i
    blur_and_downsample_str = 'blur_'+str(blur)+'_pixels_'+'downsample_factor_'+str(downsam_factor)
    model_path = '../models/' + blur_and_downsample_str + '/weights/best.pt'
    #export format
    model = YOLO(model_path)
    # Export the model
    model.export(format='engine',device=0)
