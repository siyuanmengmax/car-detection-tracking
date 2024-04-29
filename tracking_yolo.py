import cv2
import torch
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from ultralytics import YOLO



def main(video_path, yolo_model_path, deep_sort_config_path):
    # 加载YOLOv8模型
    # Load a model
    det_model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    det_model.to(device)

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
    out_video = cv2.VideoWriter('yolo_output_video.mp4', codec, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Use the model
        detections = det_model(frame)  # predict on an image
        bbox_xywh = []
        confidences = []
        class_ids = []

        # 处理检测结果，准备DeepSORT所需的数据格式
        for detection in detections:
            # print(detection)
            results = detection.boxes.data
            for result in results:
                x1, y1, x2, y2, conf, cls_id = result
                if cls_id == 2 or 7:  # 类别2是车辆
                    bbox = [(x1+x2)/2, (y1+y2)/2, x2 - x1, y2 - y1]  # 转换为x, y, w, h
                    bbox_xywh.append(bbox)
                    confidences.append(conf)
                    class_ids.append(cls_id)

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
    video_path = "/Users/rcfadmin/project/traffic-project/dataset/Top-View Vehicle Detection Image/sample_video.mp4"
    yolo_model_path = ""
    deep_sort_config_path = "deep_sort.yaml"
    main(video_path, yolo_model_path, deep_sort_config_path)
