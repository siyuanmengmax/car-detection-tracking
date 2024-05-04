import cv2
import torch
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from ultralytics import YOLO

def main(video_path, yolo_model_path):
    # Load YOLO model
    det_model = YOLO(yolo_model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    det_model.to(device)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Get video properties for the output video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format

    # Initialize video writer
    out_video = cv2.VideoWriter('Harpy_output_video.mp4', codec, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        detections = det_model(frame)

        # Process detections
        for detection in detections:
            boxes = detection.boxes.data  # Access bounding box coordinates
            print(boxes)  # Print bounding box coordinates for debugging
            for box in boxes:
                x1, y1, x2, y2, conf, cls_id = box  # Extract coordinates, confidence, and class ID
                cls_id = int(cls_id)  # Convert class ID to integer
                if cls_id in [2, 7]:  # Filter class 2/7
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(frame, str(cls_id), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Write the annotated frame to the output video
        out_video.write(frame)

        # Display frame with bounding boxes
        cv2.imshow('bbox', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "/Users/rcfadmin/project/traffic-project/dataset/DJI_0406_cut.MP4"
    yolo_model_path = "yolov8m.pt"
    main(video_path, yolo_model_path)
