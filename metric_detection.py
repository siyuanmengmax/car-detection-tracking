import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_csv(csv_path):
    labels = {}
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        frame_id = row['frame_id']
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        if frame_id not in labels:
            labels[frame_id] = []
        labels[frame_id].append([x1, y1, x2, y2])
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

def evaluate_at_thresholds(predictions, ground_truths, thresholds):
    precision_data = []
    recall_data = []
    f1_scores = []
    for iou_threshold in thresholds:
        precision_list = []
        recall_list = []
        for img_id in predictions:
            pred_boxes = predictions[img_id]
            true_boxes = ground_truths.get(img_id, [])
            correct_predictions = 0
            for pred_box in pred_boxes:
                best_iou = max(calculate_iou(pred_box, true_box) for true_box in true_boxes) if true_boxes else 0
                if best_iou >= iou_threshold:
                    precision_list.append(1)
                    correct_predictions += 1
                else:
                    precision_list.append(0)
            if len(true_boxes) > 0:
                recall = correct_predictions / len(true_boxes)
                recall_list.append(recall)
        precision = np.mean(precision_list) if precision_list else 0
        recall = np.mean(recall_list) if recall_list else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        precision_data.append(precision)
        recall_data.append(recall)
        f1_scores.append(f1_score)
    return precision_data, recall_data, f1_scores

def plot_precision_recall_curve(precision_data, recall_data, thresholds):
    plt.figure(figsize=(8, 6))
    plt.plot(recall_data, precision_data, marker='o')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    thresholds = np.arange(0.5, 1.0, 0.05)
    predictions = load_csv('output/detection_detr_image/results.csv')
    ground_truths = load_csv('dataset/Car Tracking & Object Detection/annotations.csv')
    precision_data, recall_data, f1_scores = evaluate_at_thresholds(predictions, ground_truths, thresholds)
    print(f'mAP50: {precision_data[0]}, mAR50: {recall_data[0]}')
    print(f"mAP50-95: {np.mean(precision_data)}, mAR@50-95: {np.mean(recall_data)}")
    print(f"F1 Scores50：{f1_scores[0]}")
    print(f"F1 Scores50-95：{np.mean(f1_scores)}")
    plot_precision_recall_curve(precision_data, recall_data, thresholds)
