import torch
import torch.hub
from pathlib import Path

# 检查 PyTorch 版本和 CUDA 可用性
print(torch.__version__, torch.cuda.is_available())

pretrained = True
resume_path = 'detr-r50_no-class-head.pth'

if pretrained:
    # 获取预训练权重
    checkpoint = torch.hub.load_state_dict_from_url(
        url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
        map_location='cpu',
        check_hash=True)

    # 删除类别权重
    del checkpoint["model"]["class_embed.weight"]
    del checkpoint["model"]["class_embed.bias"]

    # 保存修改后的模型
    torch.save(checkpoint, resume_path)

# 设置训练参数
dataset_file = "coco"
data_dir = '/dataset/'  # 适当调整路径
num_classes = 2
out_dir = 'outputs'

# 确保输出目录存在
Path(out_dir).mkdir(parents=True, exist_ok=True)

# 以下是替代命令行的配置和执行逻辑
def main():
    from main import main as train_model  # 假设 train_script.py 是你原始的 main.py
    train_model(
        dataset_file=dataset_file,
        coco_path=data_dir,
        output_dir=out_dir,
        resume=resume_path,
        num_classes=num_classes,
        lr=1e-5,
        lr_backbone=1e-6,
        epochs=1
    )

if __name__ == "__main__":
    main()
