import pandas as pd
import numpy as np

def load_labels_from_csv(csv_path):
    labels = {}
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        frame_id = row['frame_id']
        x1 = row['x1']
        y1 = row['y1']
        x2 = row['x2']
        y2 = row['y2']
        if frame_id not in labels:
            labels[frame_id] = []
        labels[frame_id].append([x1, y1, x2, y2])
    return labels

def calculate_iou(box1, box2):
    # 计算交并比
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area != 0 else 0
    return iou


def calculate_map(predictions, ground_truths, iou_threshold=0.5):
    precision_list = []
    recall_list = []
    for img_id in predictions:
        pred_boxes = predictions[img_id]
        true_boxes = ground_truths.get(img_id, [])
        correct_predictions = 0

        for pred_box in pred_boxes:
            best_iou = 0
            for true_box in true_boxes:
                iou = calculate_iou(pred_box, true_box)
                if iou > best_iou:
                    best_iou = iou
            if best_iou > iou_threshold:
                precision_list.append(1)
                correct_predictions += 1
            else:
                precision_list.append(0)

        if len(true_boxes) > 0:
            recall = correct_predictions / len(true_boxes)
            recall_list.append(recall)

    average_precision = np.mean(precision_list) if precision_list else 0
    average_recall = np.mean(recall_list) if recall_list else 0
    return average_precision, average_recall


if __name__ == '__main__':
    predictions = load_labels_from_csv('output/detecting_yolo_image/results.csv')
    ground_truths = load_labels_from_csv('dataset/Car Tracking & Object Detection/annotations.csv')
    ap, recall = calculate_map(predictions, ground_truths)
    print(f"Mean Average Precision: {ap}, Mean Average Recall: {recall}")
