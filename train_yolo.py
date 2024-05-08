from ultralytics import YOLO
import torch

# Load pretrained model
model = YOLO('models/yolo/yolov8x.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Configuration for training
data_yaml_path = '/mnt/d/Files/coursework/670/project/dataset/training dataset in yolo/data.yaml'
batch = 16
imgsz = 640
epochs = 300
freeze = None

training = model.train(data=data_yaml_path, epochs=epochs, imgsz=imgsz, batch=batch, freeze=freeze,single_cls=True)