from ultralytics import YOLO
import torch

# Load pretrained model
model = YOLO('models/yolo/yolov8m.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Configuration for training
data_yaml_path = '/mnt/d/Files/coursework/670/project/dataset/Traffic Camera Object Detection/data.yaml'
batch_size = 32
img_size = 640
epochs = 300

# Configure optimizer settings
optimizer_name = "Adam"  # Use the optimizer name as a string
lr0 = 0.001  # Initial learning rate

# Define a function to save the model
def save_model(model, epoch, path='models/yolo/yolov8m_trained.pth'):
    torch.save(model.state_dict(), f"{path}_epoch_{epoch}.pth")

# Start training
best_accuracy = 0
for epoch in range(epochs):
    train_results = model.train(data=data_yaml_path, batch=batch_size, epochs=1, imgsz=img_size, optimizer=optimizer_name, lr0=lr0)
    val_results = model.val(data=data_yaml_path)

    # 假设 IoU 0.5 对应的 mAP 值在索引 0
    if hasattr(train_results, 'maps'):
        train_mAP_50 = train_results.maps[0]  # 直接从数组中获取
        val_mAP_50 = val_results.maps[0]

        print(f"Epoch {epoch}: Train mAP@0.5: {train_mAP_50}, Validation mAP@0.5: {val_mAP_50}")

        # 根据某个性能指标保存最佳模型，这里使用验证集的 mAP
        if val_mAP_50 > best_accuracy:
            best_accuracy = val_mAP_50
            save_model(model, epoch)

    else:
        print("DetMetrics class does not contain 'maps' attribute.")

# 保存最终模型
save_model(model, 'final')


