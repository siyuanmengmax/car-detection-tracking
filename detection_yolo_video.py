import cv2
import torch
import pandas as pd
import os
from ultralytics import YOLO

def main(video_path, yolo_model_path, output_csv_path, output_video_folder,output_images_folder):
    # Load a model
    model = YOLO(yolo_model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_video_folder):
        os.makedirs(output_video_folder)
    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)

    # Initialize video capture and check
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video stream or file")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format

    # Initialize video writer
    output_video_path = os.path.join(output_video_folder, 'result.mp4')
    out_video = cv2.VideoWriter(output_video_path, codec, fps, (width, height))

    results = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = model(frame)
        for detection in detections:
            for result in detection.boxes.data:
                x1, y1, x2, y2, conf, cls_id = result
                if cls_id in [2, 5, 7]:  # Filter by class ID
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'Class {int(cls_id)}: {conf:.2f}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    results.append({
                        "frame_id": frame_count,
                        "class_id": int(cls_id),
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "confidence": float(conf)
                    })

        out_video.write(frame)
        output_image_path = os.path.join(output_images_folder, f'{frame_count}.png')
        cv2.imwrite(output_image_path, frame)
        frame_count += 1

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

    pd.DataFrame(results).to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    video_path = "dataset/Harpy Data Vehicle/DJI_0406_cut.mp4"
    yolo_model_path = "models/yolo/yolov8m.pt"
    output_csv_path = "output/detection_yolo_video/results.csv"
    output_video_folder = "output/detection_yolo_video/video"
    output_images_folder = "output/detection_yolo_video/images"
    main(video_path, yolo_model_path, output_csv_path, output_video_folder,output_images_folder)

