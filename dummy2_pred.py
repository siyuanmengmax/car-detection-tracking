import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_csv(csv_path):
    """Loads annotations from a CSV into a numpy array without column names."""
    df = pd.read_csv(csv_path)
    labels = df[['frame_id', 'x1', 'y1', 'x2', 'y2']].to_numpy()
    return labels

def load_csv_prediction(csv_path):
    """Loads predictions from a CSV into a numpy array including confidence scores."""
    df = pd.read_csv(csv_path)
    labels = df[['frame_id', 'x1', 'y1', 'x2', 'y2', 'confidence']].to_numpy()
    return labels

def calculate_iou(box1, box2):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def main(predictions, ground_truths, iou_threshold, confidence_thresholds):
    precision_list = []
    recall_list = []
    f1_scores = []

    unique_frames = np.unique(predictions[:, 0])

    for th in confidence_thresholds:
        TP = FP = FN = 0
        filtered_preds = predictions[predictions[:, -1] >= th]

        for frame in unique_frames:
            frame_preds = filtered_preds[filtered_preds[:, 0] == frame]
            frame_truths = ground_truths[ground_truths[:, 0] == frame]

            if len(frame_preds) == 0 and len(frame_truths) == 0:
                continue

            matched = []

            for pred in frame_preds:
                ious = [calculate_iou(pred[1:5], truth[1:5]) for truth in frame_truths]
                best_iou = max(ious) if ious else 0
                if best_iou >= iou_threshold:
                    TP += 1
                    matched.append(np.argmax(ious))
                else:
                    FP += 1

            FN += len(frame_truths) - len(set(matched))

        prec = TP / (TP + FP) if TP + FP > 0 else 0
        rec = TP / (TP + FN) if TP + FN > 0 else 0
        f1_score = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        precision_list.append(prec)
        recall_list.append(rec)
        f1_scores.append(f1_score)
        print(f'Threshold: {th:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1 Score: {f1_score:.2f}')

    return precision_list, recall_list, f1_scores

def plot_precision_recall_curve(precision_list, recall_list, title):
    """Plots the precision-recall curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(recall_list, precision_list, marker='o')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.title(f'Precision-Recall Curve for {title}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.savefig(f'output/pr curve/prCurve_{title}.png')
    plt.show()




if __name__ == '__main__':
    confidence_thresholds = np.arange(0.20, 1, 0.05)
    iou_thresholds = 0.5
    name= 'Detr_Trained'
    predictions = load_csv_prediction('output/detection_detr_image_50_full_e300/results.csv')
    # ground_truths = load_csv('dataset/Harpy Data Vehicle/annotations.csv')
    ground_truths = load_csv('dataset/Car Tracking & Object Detection/annotations.csv')
    precision_data, recall_data, f1_scores = main(predictions, ground_truths, iou_thresholds, confidence_thresholds)
    # print(f'mAP50: {precision_data[0]}, mAR50: {recall_data[0]}')
    print(f"mAP50-95: {np.mean(precision_data)}, mAR@50-95: {np.mean(recall_data)}")
    # print(f"F1 Scores50：{f1_scores[0]}")
    print(f"F1 Scores50-95：{np.mean(f1_scores)}")
    plot_precision_recall_curve(precision_data, recall_data, name)