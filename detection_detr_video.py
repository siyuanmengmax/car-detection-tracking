import cv2
import torch
import pandas as pd
import os
from PIL import Image
from transformers import AutoImageProcessor, DetrForObjectDetection

def main(video_path, detr_model_path, output_csv_path, output_images_folder, output_video_folder):
    # Load a model
    image_processor = AutoImageProcessor.from_pretrained(detr_model_path)
    model = DetrForObjectDetection.from_pretrained(detr_model_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)
    if not os.path.exists(output_video_folder):
        os.makedirs(output_video_folder)

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

    # Prepare to save results
    res = []
    frame_count = 0

    # Process each image in the folder
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Use the model
        img = Image.fromarray(frame)
        inputs = image_processor(images=img, return_tensors="pt")
        outputs = model(**inputs)
        # Process detection results
        target_sizes = torch.tensor([img.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]
        print(results)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            conf = score.item()
            cls_name = model.config.id2label[label.item()]
            # print(f"Class: {cls_name}, Confidence: {conf:.2f}, Bounding box: ({x1}, {y1}) - ({x2}, {y2})")
            # Draw bounding boxes on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'Class {cls_name}: {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # Save results for CSV
            res.append({
                "frame_id": frame_count,
                "class_name": cls_name,
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "confidence": float(conf)
            })
        # Save the frame as an image in the output folder
        output_image_path = os.path.join(output_images_folder, f'{frame_count}.png')
        cv2.imwrite(output_image_path, frame)
        out_video.write(frame)
        print(f'Processed frame {frame_count}')
        frame_count += 1

    # Create a DataFrame from the results and save to CSV
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    pd.DataFrame(res).to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    image_folder = "dataset/Harpy Data Vehicle/DJI_0406_cut.MP4"
    detr_model_path = "facebook/detr-resnet-101"
    output_csv_path = "output/detection_detr_video/results.csv"
    output_images_folder = "output/detection_detr_video/images"
    output_video_folder = "output/detection_detr_video/video"
    main(image_folder, detr_model_path, output_csv_path, output_images_folder,output_video_folder)