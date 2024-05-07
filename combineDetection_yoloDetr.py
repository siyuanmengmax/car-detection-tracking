import pandas as pd
import numpy as np

def load_csv(csv_path):
    labels = {}
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        frame_id = row['frame_id']
        x1, y1, x2, y2, conf = row['x1'], row['y1'], row['x2'], row['y2'], row['confidence']
        if frame_id not in labels:
            labels[frame_id] = []
        labels[frame_id].append([x1, y1, x2, y2, conf])
    return labels

def calculate_iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def matching_logic(box1, box2):
    better_detection = []
    if box1[-1] > box2[-1]:
        better_detection = box1
    else:
        better_detection = box2
    return better_detection

def combine_detections(detrPredictions, yoloPredictions):
    detections = []
    #loop through frames
    for img_id in detrPredictions:
        detrP = detrPredictions[img_id]
        yoloP = yoloPredictions[img_id]
        #loop through detections and match with corresponding similar ones
        for detr_box in detrP:
            better_detection = []
            yolo_matchIdx = np.argmax([calculate_iou(detr_box, yolo_box) for yolo_box in yoloP]) if yoloP else None
            #pick the one with higher confidence or if not found, add the detr ones
            #TODO: we could average them instead? or only do smtg when it's not finding stuff? idk 
            better_detection = (matching_logic(detr_box, yoloP[yolo_matchIdx])) if yolo_matchIdx else detr_box
            #add to list of detections
            detections.append({
                "frame_id": img_id,
                "x1": int(better_detection[0]),
                "y1": int(better_detection[1]),
                "x2": int(better_detection[2]),
                "y2": int(better_detection[3]),
                "confidence": float(better_detection[4])
            })
    #print(detections)
    df = pd.DataFrame(detections)
    return df

if __name__ == '__main__':
    dataset = 'image'
    detr_path = 'output/detction_detr_image_results.csv'
    yolo_path = 'output/detection_yolo_image/results.csv'
    output_path = f"output/detection_combined/results_selectHigher_{dataset}.csv"
    detr_predictions = load_csv(detr_path)
    yolo_predictions = load_csv(yolo_path)
    combined_predicitons = combine_detections(detr_predictions, yolo_predictions)
    combined_predicitons.to_csv(output_path, index=False)