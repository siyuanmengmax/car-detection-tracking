import random
import datetime
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from detr_master.models.build_model import build_model
from detr_master.engine import evaluate, train_one_epoch
from detr_master.datasets.build_dataset import build_dataset, get_coco_api_from_dataset
import detr_master.util.misc as utils
import json

class Args:
    lr = 1e-4
    lr_backbone = 1e-5
    batch_size = 2
    weight_decay = 1e-4
    epochs = 300
    lr_drop = 200
    clip_max_norm = 0.1
    num_classes = 2  # COCO has 90 classes + 1 background
    frozen_weights = None
    pretrained = True
    backbone = 'resnet50'
    dilation = False
    position_embedding = 'sine'
    enc_layers = 6
    dec_layers = 6
    dim_feedforward = 2048
    hidden_dim = 256
    dropout = 0.1
    nheads = 8
    num_queries = 100
    pre_norm = False
    masks = False
    aux_loss = True
    set_cost_class = 1
    set_cost_bbox = 5
    set_cost_giou = 2
    mask_loss_coef = 1
    dice_loss_coef = 1
    bbox_loss_coef = 5
    giou_loss_coef = 2
    eos_coef = 0.1
    dataset_file = 'coco'
    coco_path = 'dataset/training dataset in coco'
    coco_panoptic_path = None
    remove_difficult = False
    output_dir = 'models/detr/new1'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42
    resume = ''
    start_epoch = 0
    eval = False
    num_workers = 2
    world_size = 1
    dist_url = 'env://'

def main():
    args = Args()  # 创建一个参数实例
    print("Configuration:", vars(args))

    # 设置环境和种子
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load and modify pretrained model
    if args.pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
            map_location='cpu',
            check_hash=True)
        # checkpoint = torch.load("../models/detr/detr-r101-2c7b67e5.pth", map_location='cpu')
        # Remove class weights if required (only if not starting from a specific checkpoint)
        if not args.resume:
            del checkpoint["model"]["class_embed.weight"]
            del checkpoint["model"]["class_embed.bias"]

        resume_path = 'models/detr/detr-50-no-class-head-new.pth'
        torch.save(checkpoint, resume_path)
        args.resume = resume_path


    # 建立模型、数据加载器等
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    data_loader_train = DataLoader(dataset_train, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=utils.collate_fn)
    data_loader_val = DataLoader(dataset_val, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=utils.collate_fn)
    base_ds = get_coco_api_from_dataset(dataset_val)

    output_dir = Path(args.output_dir)
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


print("Training completed.")

if __name__ == '__main__':
    main()
