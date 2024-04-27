import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from new_predict import get_args_parser, load_model
from new_predict import predict_main as detr_main
import argparse
import torch
from PIL import Image

def main(video_path, detr_model_path, deep_sort_config_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model and transformation only once before the loop
    model, transform = load_model(detr_model_path, device)
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
    previous_tracks = []
    total_id_swaps = 0
    out_video = cv2.VideoWriter('detr_output_video.mp4', codec, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Use the model
        parser = argparse.ArgumentParser('DETR inference script', parents=[get_args_parser()])
        args = parser.parse_args()
        img = Image.fromarray(frame)
        detections = detr_main(args,model, transform, device, img) # predict on an image
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

        # 调用ID Swap检测函数
        if len(previous_tracks):
            # print("Previous Tracks:", previous_tracks)
            id_swaps = track_id_swaps(outputs, previous_tracks)
            total_id_swaps += id_swaps

        # 更新追踪历史
        previous_tracks = outputs.copy()

        # 显示追踪结果
        for value in outputs:
            x1, y1, x2, y2, track_id = value
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(track_id), (x1, y1 - 10), 0, 0.75, (0, 255, 0), 2)

        # Write the frame with the tracking information
        out_video.write(frame)

        cv2.imshow('Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_video.release()  # Make sure to release the video writer
    cv2.destroyAllWindows()

    print("Total ID Swaps:", total_id_swaps)
# def track_id_swaps(current_tracks, previous_tracks):
#     # ID swaps 计数器
#     id_swaps = 0
#
#     # 从上一帧的跟踪中获取ID映射
#     prev_id = [track_id for _, _, _, _, track_id in previous_tracks]
#     prev_box = {track_id:lt for lt, _, _, _, track_id in previous_tracks}
#     # 当前帧的ID映射
#     current_id = [track_id for _, _, _, _, track_id in current_tracks]
#     current_box = {track_id:lt for lt, _, _, _, track_id in current_tracks}
#     # print(previous_tracks, current_tracks)
#     # print('============')
#     # 检查ID变化
#     for c_id in current_id:
#         if c_id in prev_id:
#             if abs(current_box[c_id] - prev_box[c_id])>10:
#                 id_swaps += 1
#
#     return id_swaps
def calculate_iou(box1, box2):
    """计算两个边界框的交并比（IoU）"""
    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = box2
    # 交集的坐标
    inter_x1 = max(x1, xx1)
    inter_y1 = max(y1, yy1)
    inter_x2 = min(x2, xx2)
    inter_y2 = min(y2, yy2)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    else:
        inter_area = 0

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (xx2 - xx1) * (yy2 - yy1)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

def track_id_swaps(current_tracks, previous_tracks):
    id_swaps = 0
    # 生成上一帧ID到边界框的映射
    previous_mapping = {track[-1]: track[:-1] for track in previous_tracks}

    # 检查当前帧每个跟踪对象的最佳匹配对象
    for current in current_tracks:
        current_box = current[:-1]
        current_id = current[-1]
        best_match_id = None
        best_iou = 0

        # 在前一帧中找到与当前对象IoU最高的对象
        for prev_id, prev_box in previous_mapping.items():
            iou = calculate_iou(current_box, prev_box)
            if iou > best_iou:
                best_iou = iou
                best_match_id = prev_id

        # 如果最佳匹配对象的ID与当前对象的ID不同，且IoU超过阈值，则认为发生了ID swap
        if best_match_id is not None and best_match_id != current_id and best_iou > 0.1:
            id_swaps += 1

    return id_swaps

if __name__ == "__main__":
    video_path = "/mnt/d/Files/coursework/670/project/dataset/Top-View Vehicle Detection Image/sample_video.mp4"
    detr_model_path = "checkpoint0299.pth"
    deep_sort_config_path = "deep_sort.yaml"
    main(video_path, detr_model_path, deep_sort_config_path)
