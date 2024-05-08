import cv2
import torch
import os
import pandas as pd
from deep_sort_pytorch.deep_sort import DeepSort

def main(video_path, ground_truth_path, output_csv_path, output_video_folder,output_images_folder):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video stream or file")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize video writer
    output_video_path = os.path.join(output_video_folder, 'result.mp4')
    out_video = cv2.VideoWriter(output_video_path, codec, fps, (width, height))

    # Load ground truth data from CSV
    df = pd.read_csv(ground_truth_path)

    results = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Filter dataframe for current frame
        current_frame_df = df[df['frame_id'] == frame_count]
        bbox_xywh = []
        confidences = []
        for index, row in current_frame_df.iterrows():
            x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            bbox = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
            bbox_xywh.append(bbox)
            confidences.append(1.0)
        # print(f"Processed frame {frame_count}, found {len(bbox_xywh)} vehicles")

        if bbox_xywh:
            outputs = deepsort.update(torch.tensor(bbox_xywh), torch.tensor(confidences), frame)
            for idx, output in enumerate(outputs):
                x1, y1, x2, y2, track_id = output
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'Vehicle {int(track_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                results.append({
                    "frame_id": frame_count,
                    "vehicle_id": int(track_id),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                })
        print(f"Processed frame {frame_count}, found {len(outputs)} vehicles")
        out_video.write(frame)
        output_image_path = os.path.join(output_images_folder, f'{frame_count}.png')
        cv2.imwrite(output_image_path, frame)
        frame_count += 1

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

    # Save results to CSV
    pd.DataFrame(results).to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    video_path = "dataset/Harpy Data Vehicle/DJI_0406_cut.MP4"
    ground_truth_path = "dataset/Harpy Data Vehicle/annotations.csv"
    output_csv_path = "output/tracking_groundtruth_video_new/results.csv"
    output_video_folder = "output/tracking_groundtruth_video_new/video"
    output_images_folder = "output/tracking_groundtruth_video_new/images"
    main(video_path, ground_truth_path, output_csv_path, output_video_folder,output_images_folder)