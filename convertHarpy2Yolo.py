import pandas as pd

# 加载数据
df = pd.read_csv('dataset/Harpy Data Vehicle/DJI_0406_tracks.csv')

# 计算 x1, y1, x2, y2
# df['x1'] = df['bboxX'] + (df['Width'] / 2)
# df['y1'] = df['bboxY'] - (df['Height'] / 2)
# df['x2'] = df['bboxX'] + (df['Width'] / 2)
# df['y2'] = df['bboxY'] + (df['Height'] / 2)
df['x1'] = df['bboxX']
df['y1'] = df['bboxY']
df['x2'] = df['bboxX'] + df['Width']
df['y2'] = df['bboxY'] + df['Height']

# 选择和重命名所需的列
output_df = df[['frame', 'veh_id', 'x1', 'y1', 'x2', 'y2']].copy()
output_df.columns = ['frame_id', 'vehicle_id', 'x1', 'y1', 'x2', 'y2']

# 保存为 CSV
output_path = 'dataset/Harpy Data Vehicle/annotations.csv'
output_df.to_csv(output_path, index=False)

import csv
from collections import defaultdict
import pandas as pd
#
# import cv2
#
#
# def get_video_dimensions(video_path):
#     # Open the video file
#     cap = cv2.VideoCapture(video_path)
#
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return None
#
#     # Get width and height of the video
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     # Release the video capture object
#     cap.release()
#
# #     return width, height
# #
# #
# # # Replace 'path_to_your_video.mp4' with the path to your video file
# # video_path = 'path_to_your_video.mp4'
# # dimensions = get_video_dimensions(video_path)
#
# if dimensions:
#     print("Video Width:", dimensions[0])
#     print("Video Height:", dimensions[1])
# else:
#     print("Failed")


    # Dummy image dimensions, replace these with actual image dimensions
# img_width = 1920
# img_height = 1080
#
# # Example data, replace `data` with your actual CSV data
# data = '/mnt/d/Files/coursework/670/project/dataset/Harpy Data Vehicle/DJI_0406_tracks.csv'
# reader = pd.read_csv(data,delimiter=',')
# # Parsing CSV data
# # reader = csv.DictReader(data.splitlines())
# annotations = defaultdict(list)
# # print(list(reader))
# for i,row in reader.iterrows():
#     frame = f"{row['frame']}"
#     x_center = (int(row['bboxX']) + int(row['Width']) / 2) / img_width
#     y_center = (int(row['bboxY']) + int(row['Height']) / 2) / img_height
#     width = int(row['Width']) / img_width
#     height = int(row['Height']) / img_height
#     # print(row)
#     annotations[frame].append(f"0 {x_center} {y_center} {width} {height}\n")
# # print(annotations)
#
# # Creating and writing to txt files
# import os
#
# output_folder = "annotations"
# os.makedirs(output_folder, exist_ok=True)
#
# for frame, bboxes in annotations.items():
#     with open(os.path.join(output_folder, f"{frame}.txt"), 'w') as file:
#         for bbox in bboxes:
#             file.write(bbox)
#
# print("Annotation files have been created.")
