import numpy as np
import os
import cv2
import torch
import pandas as pd
from ultralytics import YOLO


def main(input_path, model_path):

    det_model = YOLO(model_path)

    files=os.listdir(input_path)

    files_to_csv=[]
    columns_headers = ['frame','class','x1', 'y1', 'x2', 'y2','score']

    # Check if there are any files in the directory
    if len(files) == 0:
        print("No files found in the directory.")
    else:
        # Construct full paths for each file
        file_paths = [os.path.join(input_path, filename) for filename in files]

        # Read the first image (assuming there's at least one image in the directory)

        for file in file_paths: #range(len(file_paths))

            frame = cv2.imread(file) #file_paths[file]

            # Check if the image was read successfully
            if frame is None:
                print(f"Failed to read the image: {file}")
            else:
                # print(file)

    
    
                detections = det_model(frame)

                boxes=detections[0].boxes.data

                for box in boxes:
                    x1, y1, x2, y2, conf, cls_id = box.tolist()  # Extract coordinates, confidence, and class ID
                    cls_id = int(cls_id)  # Convert class ID to integer
                    if cls_id in [2, 7]:  # Filter class 2/7
                        bbox = (int(x1), int(y1), int(x2), int(y2))
                        class_name="vehicle"
                        files_to_csv.append([os.path.basename(file),class_name,bbox[0],bbox[1],bbox[2],bbox[3],conf])
                        
                        # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        # cv2.putText(frame, str(cls_id), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    df = pd.DataFrame(files_to_csv)
    df.to_csv("dummy.csv", sep=',', header=columns_headers, index=False)
    # Display the image with bounding boxes
    # cv2.imshow('Bounding Boxes', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    input_path = "/Users/rcfadmin/project/traffic-project/detr_master/dataset/coco/val2017"
    yolo_model_path = "yolov8m.pt"
    main(input_path, yolo_model_path)