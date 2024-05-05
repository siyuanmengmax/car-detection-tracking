import cv2
import torch
import pandas as pd
import os
from ultralytics import YOLO
from pathlib import Path

def main(image_folder, yolo_model_path, output_csv_path, output_images_folder, output_video_folder,image_format='PNG'):
    # Load a model
    model = YOLO(yolo_model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f"Using device: {device}")
    model.to(device)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)
    if not os.path.exists(output_video_folder):
        os.makedirs(output_video_folder)

    # Initialize video writer with a guessed fps
    first_image_path = next(Path(image_folder).glob(f'*.{image_format}'))
    first_frame = cv2.imread(str(first_image_path))
    height, width = first_frame.shape[:2]
    fps = 30  # Assuming a typical frame rate; adjust as necessary
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(output_video_folder, 'result.mp4')
    out_video = cv2.VideoWriter(output_video_path, codec, fps, (width, height))

    # Prepare to save results
    results = []
    frame_count = 0

    # Process each image in the folder
    for image_file in sorted(Path(image_folder).glob(f'*.{image_format}')):
        frame = cv2.imread(str(image_file))
        if frame is None:
            continue

        # Use the model to predict on the image
        detections = model(frame)
        # Process detection results
        for detection in detections:
            # Extract bounding box data
            for result in detection.boxes.data:
                x1, y1, x2, y2, conf, cls_id = result
                # print(f"Class ID: {cls_id}, Confidence: {conf:.2f}, Bounding box: ({x1}, {y1}) - ({x2}, {y2})")
                if cls_id in [2, 5, 7]: # car, bus, truck
                    # print(f"Class ID: {cls_id}, Confidence: {conf:.2f}, Bounding box: ({x1}, {y1}) - ({x2}, {y2})")
                    # Draw bounding boxes on the frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'Class {int(cls_id)}: {conf:.2f}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    # Save results for CSV
                    results.append({
                        "frame_id": frame_count,
                        "class_id": int(cls_id),
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "confidence": float(conf)
                    })
        # Save the frame as an image in the output folder
        output_image_path = os.path.join(output_images_folder, image_file.name)
        cv2.imwrite(output_image_path, frame)
        out_video.write(frame)
        frame_count += 1

    # Create a DataFrame from the results and save to CSV
    out_video.release()
    cv2.destroyAllWindows()
    pd.DataFrame(results).to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    image_folder = "dataset/Car Tracking & Object Detection/images"
    yolo_model_path = "models/yolo/yolov8m.pt"
    output_csv_path = "output/detection_yolo_image/results.csv"
    image_format = 'PNG'
    output_images_folder = "output/detection_yolo_image/images"
    output_video_folder = "output/detection_yolo_image/video"
    main(image_folder, yolo_model_path, output_csv_path, output_images_folder,output_video_folder,image_format)
