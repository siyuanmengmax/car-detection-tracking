import json
import os
from PIL import Image

# Paths configuration
train_images_path = 'D://Files/coursework/670/project/dataset/Traffic Camera Object Detection/train/images'
val_images_path = 'D://Files/coursework/670/project/dataset/Traffic Camera Object Detection/valid/images'
train_annotations_path = 'D://Files/coursework/670/project/dataset/Traffic Camera Object Detection/train/labels'  # Assuming YOLO labels are here
val_annotations_path = 'D://Files/coursework/670/project/dataset/Traffic Camera Object Detection/valid/labels'    # Assuming YOLO labels are here



def convert_yolo_to_coco(images_path, annotations_path, is_train=True):
    image_id = 1
    annotation_id = 1
    for filename in os.listdir(images_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_path, filename)
            image = Image.open(image_path)
            width, height = image.size

            # Adding image information to COCO dataset
            coco_data['images'].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": filename
            })

            # YOLO annotation file
            yolo_annotation_file = os.path.join(annotations_path, filename.rsplit('.', 1)[0] + '.txt')
            if os.path.exists(yolo_annotation_file):
                with open(yolo_annotation_file, 'r') as file:
                    for line in file:
                        class_id, x_center, y_center, w, h = map(float, line.strip().split())
                        # Convert YOLO to absolute coordinates and COCO format
                        x_min = (x_center - w / 2) * width
                        y_min = (y_center - h / 2) * height
                        bbox_width = w * width
                        bbox_height = h * height

                        # Adding annotation information
                        coco_data['annotations'].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(class_id) + 1,  # YOLO is zero-indexed, COCO is not
                            "bbox": [x_min, y_min, bbox_width, bbox_height],
                            "area": bbox_width * bbox_height,
                            "segmentation": [],
                            "iscrowd": 0
                        })
                        annotation_id += 1

            image_id += 1

# Process training and validation datasets
# COCO dataset structure
coco_data = {
    "images": [],
    "annotations": [],
    "categories": [{'id': 1, 'name': 'car'}]  # Update this based on your 'names' in YAML
}
convert_yolo_to_coco(train_images_path, train_annotations_path, is_train=True)
# Save to JSON
with open('detr_master/dataset/coco/annotations/instances_train2017.json', 'w') as json_file:
    json.dump(coco_data, json_file, indent=4)

# Process training and validation datasets
# COCO dataset structure
coco_data = {
    "images": [],
    "annotations": [],
    "categories": [{'id': 1, 'name': 'car'}]  # Update this based on your 'names' in YAML
}
convert_yolo_to_coco(val_images_path, val_annotations_path, is_train=False)
# Save to JSON
with open('detr_master/dataset/coco/annotations/instances_val2017.json', 'w') as json_file:
    json.dump(coco_data, json_file, indent=4)

print("Conversion complete!")
