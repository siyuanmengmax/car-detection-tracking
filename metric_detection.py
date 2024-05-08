import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_csv(csv_path):
    """Loads annotations from a CSV into a dictionary."""
    labels = {}
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        frame_id = row['frame_id']
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        if frame_id not in labels:
            labels[frame_id] = []
        labels[frame_id].append([x1, y1, x2, y2])
    return labels

def load_csv_prediction(csv_path):
    """Loads predictions from a CSV into a dictionary, including confidence scores."""
    labels = {}
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        frame_id = row['frame_id']
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        confidence = row['confidence']
        if frame_id not in labels:
            labels[frame_id] = []
        labels[frame_id].append([x1, y1, x2, y2, confidence])
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
    return inter_area / union_area if union_area != 0 else 0

def main(predictions, ground_truths, iou_threshold, confidence_thresholds):
    precision_data = []
    recall_data = []
    f1_scores = []
    for threshold in confidence_thresholds:
        filtered_predictions = {frame_id: [box for box in boxes if box[4] >= threshold]
                                for frame_id, boxes in predictions.items()}
        precision_list = []
        recall_list = []
        for img_id, pred_boxes in filtered_predictions.items():
            true_boxes = ground_truths.get(img_id, [])
            correct_predictions = set()
            for pred_box in pred_boxes:
                best_iou = -1
                best_true_box_index = -1
                for i, true_box in enumerate(true_boxes):
                    iou = calculate_iou(pred_box[:4], true_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_true_box_index = i
                if best_iou >= iou_threshold and best_true_box_index not in correct_predictions:
                    correct_predictions.add(best_true_box_index)
                    precision_list.append(1)
                else:
                    precision_list.append(0)
            recall = len(correct_predictions) / len(true_boxes) if true_boxes else 0
            recall_list.append(recall)

        precision = np.mean(precision_list) if precision_list else 0
        recall = np.mean(recall_list) if recall_list else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        precision_data.append(precision)
        recall_data.append(recall)
        f1_scores.append(f1_score)
        print(f'Threshold: {threshold:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}')
    return precision_data, recall_data, f1_scores

def plot_precision_recall_curve(precision_data, recall_data, title):
    """Plots the precision-recall curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(recall_data, precision_data, marker='o')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.title(f'Precision-Recall Curve for {title}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.savefig(f'output/pr curve/prCurve_{title}.png')
    plt.show()

if __name__ == '__main__':
    confidence_thresholds = np.arange(0.0, 1, 0.005)
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




# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# def load_csv(csv_path):
#     labels = {}
#     df = pd.read_csv(csv_path)
#     for index, row in df.iterrows():
#         frame_id = row['frame_id']
#         x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
#         if frame_id not in labels:
#             labels[frame_id] = []
#         labels[frame_id].append([x1, y1, x2, y2])
#     return labels
#
# def load_csv_prediction(csv_path):
#     labels = {}
#
#     df = pd.read_csv(csv_path)
#     for index, row in df.iterrows():
#         frame_id = row['frame_id']
#         x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
#         confidence = row['confidence']
#         if frame_id not in labels:
#             labels[frame_id] = []
#         labels[frame_id].append([x1, y1, x2, y2, confidence])
#     return labels
#
# def calculate_iou(box1, box2):
#     inter_x1 = max(box1[0], box2[0])
#     inter_y1 = max(box1[1], box2[1])
#     inter_x2 = min(box1[2], box2[2])
#     inter_y2 = min(box1[3], box2[3])
#     inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
#     union_area = box1_area + box2_area - inter_area
#     return inter_area / union_area if union_area != 0 else 0
#
# def main(predictions, ground_truths, iou_threshold,confidence_thresholds):
#     precision_data = []
#     recall_data = []
#     f1_scores = []
#     for threshold in confidence_thresholds:
#         precision_list = []
#         recall_list = []
#         prediction = predictions[predictions['confidence'][:] >= threshold]
#         for img_id in prediction:
#             pred_boxes = prediction[img_id]
#             true_boxes = ground_truths.get(img_id, [])
#             # correct_predictions = 0
#             correct_predictions = set()
#             for pred_box in pred_boxes:
#                 best_iou = -1
#                 best_true_box_index = -1
#                 # best_iou = max(calculate_iou(pred_box, true_box) for true_box in true_boxes) if true_boxes else 0
#                 for i, true_box in enumerate(true_boxes):
#                     iou = calculate_iou(pred_box, true_box)
#                     if iou > best_iou:
#                         best_iou = iou
#                         best_true_box_index = i
#                 # if best_iou >= iou_threshold:
#                 #     precision_list.append(1)
#                 #     correct_predictions += 1
#                 # else:
#                 #     precision_list.append(0)
#                 if best_iou >= iou_threshold and best_true_box_index not in correct_predictions:
#                     correct_predictions.add(best_true_box_index)
#                     precision_list.append(1)
#                 else:
#                     precision_list.append(0)
#                 # Ensure no division by zero
#             if len(true_boxes) > 0:
#                 recall = len(correct_predictions) / len(true_boxes)
#             else:
#                 recall = 0
#             recall_list.append(recall)
#             # if len(true_boxes) > 0:
#             #     recall = correct_predictions / len(true_boxes)
#             #     recall_list.append(recall)
#
#         precision = np.mean(precision_list) if precision_list else 0
#         recall = np.mean(recall_list) if recall_list else 0
#         f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#         precision_data.append(precision)
#         recall_data.append(recall)
#         f1_scores.append(f1_score)
#     return precision_data, recall_data, f1_scores
#
# def plot_precision_recall_curve(precision_data, recall_data):
#     plt.figure(figsize=(8, 6))
#     plt.plot(recall_data, precision_data, marker='o')
#     plt.title('Precision-Recall Curve')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.grid(True)
#     plt.show()
#
# if __name__ == '__main__':
#     confidence_thresholds = np.arange(0.25, 0.95, 0.05)
#     iou_thresholds = 0.5
#     predictions = load_csv_prediction('output/detection_yolo_video_x_full_e300/results.csv')
#     ground_truths = load_csv('dataset/Harpy Data Vehicle/annotations.csv')
#     # ground_truths = load_csv('dataset/Car Tracking & Object Detection/annotations.csv')
#     precision_data, recall_data, f1_scores = main(predictions, ground_truths, iou_thresholds, confidence_thresholds)
#     # print(f'mAP50: {precision_data[0]}, mAR50: {recall_data[0]}')
#     print(f"mAP50-95: {np.mean(precision_data)}, mAR@50-95: {np.mean(recall_data)}")
#     # print(f"F1 Scores50：{f1_scores[0]}")
#     print(f"F1 Scores50-95：{np.mean(f1_scores)}")
#     plot_precision_recall_curve(precision_data, recall_data)
#
#



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# def load_csv(csv_path):
#     labels = {}
#     df = pd.read_csv(csv_path)
#     for index, row in df.iterrows():
#         frame_id = row['frame_id']
#         x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
#         if frame_id not in labels:
#             labels[frame_id] = []
#         labels[frame_id].append([x1, y1, x2, y2])
#     return labels
#
# def calculate_iou(box1, box2):
#     inter_x1 = max(box1[0], box2[0])
#     inter_y1 = max(box1[1], box2[1])
#     inter_x2 = min(box1[2], box2[2])
#     inter_y2 = min(box1[3], box2[3])
#     inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
#     union_area = box1_area + box2_area - inter_area
#     return inter_area / union_area if union_area != 0 else 0
#
# def main(predictions, ground_truths, thresholds):
#     precision_data = []
#     recall_data = []
#     f1_scores = []
#     for iou_threshold in thresholds:
#         precision_list = []
#         recall_list = []
#         for img_id in predictions:
#             pred_boxes = predictions[img_id]
#             true_boxes = ground_truths.get(img_id, [])
#             # correct_predictions = 0
#             correct_predictions = set()
#             for pred_box in pred_boxes:
#                 best_iou = -1
#                 best_true_box_index = -1
#                 # best_iou = max(calculate_iou(pred_box, true_box) for true_box in true_boxes) if true_boxes else 0
#                 for i, true_box in enumerate(true_boxes):
#                     iou = calculate_iou(pred_box, true_box)
#                     if iou > best_iou:
#                         best_iou = iou
#                         best_true_box_index = i
#                 # if best_iou >= iou_threshold:
#                 #     precision_list.append(1)
#                 #     correct_predictions += 1
#                 # else:
#                 #     precision_list.append(0)
#                 if best_iou >= iou_threshold and best_true_box_index not in correct_predictions:
#                     correct_predictions.add(best_true_box_index)
#                     precision_list.append(1)
#                 else:
#                     precision_list.append(0)
#                 # Ensure no division by zero
#             if len(true_boxes) > 0:
#                 recall = len(correct_predictions) / len(true_boxes)
#             else:
#                 recall = 0
#             recall_list.append(recall)
#             # if len(true_boxes) > 0:
#             #     recall = correct_predictions / len(true_boxes)
#             #     recall_list.append(recall)
#
#         precision = np.mean(precision_list) if precision_list else 0
#         recall = np.mean(recall_list) if recall_list else 0
#         f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#         precision_data.append(precision)
#         recall_data.append(recall)
#         f1_scores.append(f1_score)
#     return precision_data, recall_data, f1_scores
#
# def plot_precision_recall_curve(precision_data, recall_data):
#     plt.figure(figsize=(8, 6))
#     plt.plot(recall_data, precision_data, marker='o')
#     plt.title('Precision-Recall Curve')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.grid(True)
#     plt.show()
#
# if __name__ == '__main__':
#     thresholds = np.arange(0.5, 1.0, 0.05)
#     predictions = load_csv('output/detection_yolo_video_x_full_e300/results.csv')
#     ground_truths = load_csv('dataset/Harpy Data Vehicle/annotations.csv')
#     # ground_truths = load_csv('dataset/Car Tracking & Object Detection/annotations.csv')
#     precision_data, recall_data, f1_scores = main(predictions, ground_truths, thresholds)
#     print(f'mAP50: {precision_data[0]}, mAR50: {recall_data[0]}')
#     print(f"mAP50-95: {np.mean(precision_data)}, mAR@50-95: {np.mean(recall_data)}")
#     print(f"F1 Scores50：{f1_scores[0]}")
#     print(f"F1 Scores50-95：{np.mean(f1_scores)}")
#     plot_precision_recall_curve(precision_data, recall_data)
