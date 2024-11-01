import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from predict import get_args_parser
from predict import predict_main as detr_main
import argparse
import torch
from PIL import Image

def main(video_path, detr_model_path, deep_sort_config_path):
    # 加载DeepSORT配置和模型
    cfg = get_config()
    cfg.merge_from_file(deep_sort_config_path)
    deepsort = DeepSort(
        cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=True
    )

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    # Get video properties for the output video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format

    # Initialize video writer
    out_video = cv2.VideoWriter('detr_output_video.mp4', codec, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Use the model
        parser = argparse.ArgumentParser('DETR inference script', parents=[get_args_parser()])
        args = parser.parse_args()
        img = Image.fromarray(frame)
        detections = detr_main(args,detr_model_path,img) # predict on an image
        # print(detections)
        bbox_xywh = []
        confidences = []
        class_ids = []

        # 处理检测结果，准备DeepSORT所需的数据格式
        for detection in detections:
            bbox_xywh.append(detection['bbox'])
            confidences.append(detection['score'])
            class_ids.append(detection['class_id'])

        # 调用DeepSORT进行追踪
        outputs = deepsort.update(torch.tensor(bbox_xywh), torch.tensor(confidences), frame)

        # 显示追踪结果
        for value in outputs:
            x1, y1, x2, y2, track_id = value
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(track_id), (x1, y1 - 10), 0, 0.75, (0, 255, 0), 2)

        # Write the frame with the tracking information
        out_video.write(frame)

        # cv2.imshow('Tracker', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out_video.release()  # Make sure to release the video writer
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "/mnt/d/Files/coursework/670/project/dataset/Top-View Vehicle Detection Image/sample_video.mp4"
    detr_model_path = "checkpoint0299.pth"
    deep_sort_config_path = "deep_sort.yaml"
    main(video_path, detr_model_path, deep_sort_config_path)
