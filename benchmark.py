from ultralytics.yolo.utils.benchmarks import benchmark
from ultralytics import YOLO

model_path = '../models/blur_1.5_pixels_downsample_factor_0.33/weights/best.pt'

benchmark(model=model_path, imgsz=640, half=False, device=0)

#export format
model = YOLO(model_path)
# Export the model
#model.export(format='engine',device=0)