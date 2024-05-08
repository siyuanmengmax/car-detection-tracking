import cv2
import torch
import os
import pandas as pd
from deep_sort_pytorch.deep_sort import DeepSort
from pathlib import Path
from PIL import Image
# untrained model
from transformers import AutoImageProcessor, DetrForObjectDetection
# trained model
from detr_master.predict import load_model
from detr_master.predict import predict_main as detr_main

def main(image_folder, detr_model_path, output_csv_path, output_video_folder, output_images_folder,image_format='png'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load a model
    if detr_model_path in ["facebook/detr-resnet-50", "facebook/detr-resnet-101"]:
        image_processor = AutoImageProcessor.from_pretrained(detr_model_path)
        model = DetrForObjectDetection.from_pretrained(detr_model_path)
        model.to(device)
    else:
        model, transform = load_model(detr_model_path, device)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)
    if not os.path.exists(output_video_folder):
        os.makedirs(output_video_folder)

    # Load DeepSORT configuration and model
    # deepsort = DeepSort(
    #     model_path='models/deepsort_reid/ckpt.t7',
    #     max_dist=0.2,
    #     min_confidence=0.3,
    #     nms_max_overlap=1.0,
    #     max_iou_distance=0.7,
    #     max_age=70,
    #     n_init=3,
    #     nn_budget=100,
    #     use_cuda=device == 'cuda'
    # )
    deepsort = DeepSort(
        model_path='models/deepsort_reid/ckpt.t7',
        max_dist=0.3,
        min_confidence=0.5,
        nms_max_overlap=0.5,
        max_iou_distance=0.5,
        max_age=30,
        n_init=3,
        nn_budget=100,
        use_cuda=device == 'cuda'
    )

    # Initialize video writer with a guessed fps
    first_image_path = next(Path(image_folder).glob(f'*.{image_format}'))
    first_frame = cv2.imread(str(first_image_path))
    height, width = first_frame.shape[:2]
    fps = 30  # Assuming a typical frame rate; adjust as necessary
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(output_video_folder, 'result.mp4')
    out_video = cv2.VideoWriter(output_video_path, codec, fps, (width, height))

    res = []
    frame_count = 0

    # Process each image in the folder
    for image_file in sorted(Path(image_folder).glob(f'*.{image_format}')):
        frame = cv2.imread(str(image_file))
        # print(type(frame))
        if frame is None:
            continue
        bbox_xywh = []
        confidences = []
        # Use the model
        if detr_model_path in ["facebook/detr-resnet-50", "facebook/detr-resnet-101"]:
            img = Image.fromarray(frame)
            inputs = image_processor(images=img, return_tensors="pt").to(device)
            outputs = model(**inputs)
            # Process detection results
            target_sizes = torch.tensor([img.size[::-1]])
            results = image_processor.post_process_object_detection(outputs, threshold=0.25, target_sizes=target_sizes)[0]
            # print(results)
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                conf = score.item()
                cls_name = model.config.id2label[label.item()]
                if cls_name in ['car', 'truck', 'bus']:  # Filter by class name
                    bbox = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
                    bbox_xywh.append(bbox)
                    confidences.append(conf)
        else:
            img = Image.fromarray(frame)
            detections = detr_main(model, transform, device, img, confidence_threshold=0.25)  # predict on an image
            # Process detection results
            for detection in detections:
                conf = detection['score']
                bbox = detection['bbox']
                bbox_xywh.append(bbox)
                confidences.append(conf)

        if bbox_xywh:
            outputs = deepsort.update(torch.tensor(bbox_xywh), torch.tensor(confidences), frame)
            for idx, output in enumerate(outputs):
                x1, y1, x2, y2, track_id = output
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'Vehicle {int(track_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                res.append({
                    "frame_id": frame_count,
                    "vehicle_id": int(track_id),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                })

        out_video.write(frame)
        output_image_path = os.path.join(output_images_folder, image_file.name)
        cv2.imwrite(output_image_path, frame)
        print(f'Processed frame {frame_count}')
        frame_count += 1

    out_video.release()
    cv2.destroyAllWindows()

    # Save results to CSV
    pd.DataFrame(res).to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    image_folder = "dataset/Car Tracking & Object Detection/images"
    # detr_model_path = "facebook/detr-resnet-101"
    detr_model_path = "models/detr/checkpoint0299.pth"
    output_csv_path = "output/tracking_detr_image_50_full_e300/results.csv"
    output_video_folder = 'output/tracking_detr_image_50_full_e300/video'
    output_images_folder = "output/tracking_detr_image_50_full_e300/images"
    image_format = 'PNG'
    main(image_folder, detr_model_path, output_csv_path, output_video_folder, output_images_folder,image_format)
