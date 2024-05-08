import pandas as pd

def load_csv(csv_path, include_vehicle_id=False):
    """ Load CSV file and structure it into a dictionary organized by frame_id. """
    labels = {}
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        frame_id = row['frame_id']
        box = [row['x1'], row['y1'], row['x2'], row['y2']]
        if include_vehicle_id:
            box.append(row['vehicle_id'])
        labels.setdefault(frame_id, []).append(box)
    return labels

def calculate_iou(box1, box2):
    """ Calculate Intersection over Union (IoU) between two bounding boxes. """
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def track_id_swaps(current_tracks, previous_tracks, iou_threshold=0.5):
    """ Calculate ID swaps based on tracking continuity and ID changes with high IoU. """
    id_swaps = 0
    previous_mapping = {track[-1]: track[:-1] for track in previous_tracks}  # Map vehicle_id to box

    for current in current_tracks:
        current_box = current[:-1]
        current_id = current[-1]
        best_match_id = None
        best_iou = 0

        for prev_id, prev_box in previous_mapping.items():
            iou = calculate_iou(current_box, prev_box)
            if iou > best_iou:
                best_iou = iou
                best_match_id = prev_id

        if best_match_id is not None and best_match_id != current_id and best_iou > iou_threshold:
            id_swaps += 1

    return id_swaps

def main(predictions, ground_truths, iou_threshold=0.5):
    """ Calculate the Multiple Object Tracking Accuracy (MOTA). """
    total_false_negatives = 0
    total_false_positives = 0
    total_id_swaps = 0
    total_objects = sum(len(boxes) for boxes in ground_truths.values())  # Total GT boxes

    previous_tracks = []

    for frame_id in sorted(ground_truths):
        gt_boxes = ground_truths.get(frame_id, [])
        # print(gt_boxes)
        pred_boxes = predictions.get(frame_id, [])
        # print(pred_boxes)
        matched = []

        # Match predictions to ground truths
        for gt_box in gt_boxes:
            best_iou = 0
            best_match_idx = None
            for idx, pred_box in enumerate(pred_boxes):
                iou = calculate_iou(pred_box[:-1], gt_box)  # Exclude vehicle_id in pred_box
                if iou > best_iou and idx not in matched:
                    best_iou = iou
                    best_match_idx = idx
            if best_iou >= iou_threshold:
                matched.append(best_match_idx)
            else:
                if frame_id > 1:
                    total_false_negatives += 1

        total_false_positives += len(pred_boxes) - len(matched)

        # Count ID swaps if vehicle IDs are available
        if previous_tracks:
            total_id_swaps += track_id_swaps(pred_boxes, previous_tracks, iou_threshold)

        previous_tracks = pred_boxes.copy()
        print("Frame ID:", frame_id, "Total False Negatives:", total_false_negatives,"Total False Positives:", total_false_positives,"ID Swaps:", total_id_swaps)
    # Calculate MOTA
    mota = 1 - (total_false_negatives + total_false_positives + total_id_swaps) / total_objects if total_objects > 0 else 0
    if mota < 0:
        mota = 0
    return mota

if __name__ == '__main__':
    predictions = load_csv('output/tracking_combined_image_new/results.csv', include_vehicle_id=True)
    ground_truths = load_csv('dataset/Car Tracking & Object Detection/annotations.csv')
    # ground_truths = load_csv('dataset/Car Tracking & Object Detection/annotations.csv')
    mota_score = main(predictions, ground_truths)
    print(f"MOTA Score: {mota_score}")