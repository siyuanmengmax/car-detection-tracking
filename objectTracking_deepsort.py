import cv2
import torch
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import pandas as pd
import numpy as np

test = False

def track_objects(deep_sort_config_path):
    cfg = get_config()
    cfg.merge_from_file(deep_sort_config_path)
    deepsort = DeepSort(
        cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=True
    )
    #full dataset
    #vidPath = 'Harpy Data Vehicle/DJI_0406_full.MP4'
    #filePath = 'Harpy Data Vehicle/DJI_0406_tracks.csv'
    
    #cut dataset
    vidPath = 'Harpy Data Vehicle/DJI_0406_cut.MP4'
    filePath = 'Harpy Data Vehicle/DJI_0406_cut_tracks.csv'
    
    data = pd.read_csv(filePath, delimiter=',')
    datasetName = 'harpy'
    datasetType = 'full'

    video = cv2.VideoCapture(vidPath)
    # Get video properties for the output video
    #width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)/4)
    #height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)/4)
    width = 1280
    height = 720
    
    fps = video.get(cv2.CAP_PROP_FPS)

    codec = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    bboxes = data.loc[:, ['frame','bboxX','bboxY','Width','Height']]
    columns_headers = ['x1', 'y1', 'x2', 'y2', 'id', 'frame']

    confidences = torch.ones(bboxes.shape[0])
    tracking_info = []
    # Initialize video writer
    out_video = cv2.VideoWriter(f'deepsort_{datasetName}{datasetType}_output_videoTest.mp4', codec, fps, (width, height))
    while video.isOpened():
        ret, frame = video.read()
        frame = cv2.resize(frame, (width,height))
        current_frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES)) #-1
        if not ret or current_frame_number>100:
            break  

        boxes = bboxes.loc[bboxes['frame']==current_frame_number].values
        #print(boxes)
        if boxes[:,1:].shape[0] > 0:
            if test:
                for b in range(0, len(boxes)):
                    #print(boxes[b,1:])
                    bounding_Box = boxes[b,1:]
                    
                    x1 = int((2*bounding_Box[0]-bounding_Box[2])/2)#+300
                    x2 = int((2*bounding_Box[0]+bounding_Box[2])/2)#+300
                    y1 = int((2*bounding_Box[1]-bounding_Box[3])/2)#+75
                    y2 = int((2*bounding_Box[1]+bounding_Box[3])/2)#+75
                    #print(f"({x1}, {y1}) ({x2},{y2})")
                    if bounding_Box[0] == 630:
                        print(f"({x1}, {y1}) ({x2},{y2})")
                        print(f"frame: {current_frame_number}")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "1", (x1, y1 - 10), 0, 0.75, (0, 255, 0), 2)

            else:
                input_boxes = boxes[:,1:]
                input_conf = confidences[0:len(boxes)]
                outputs = deepsort.update(torch.tensor(input_boxes), input_conf, frame)
                #print("slice of box")
                #print(input_boxes)
                #print("output")
                #print(outputs)
                for value in outputs:
                    #write to video
                    #center, w, h
                    #x1*currentimgW/originalW
                    x1, y1, x2, y2, track_id = value
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, str(track_id), (x1, y1 - 10), 0, 0.75, (0, 255, 0), 2)
                    #format and store output data
                    single_box = np.append(value.astype(int),current_frame_number)
                    tracking_info.append(single_box)
        # Write the frame with the tracking information
        out_video.write(frame)
        cv2.imshow('Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #save output to a csv
    #df = pd.DataFrame(tracking_info)
    #df.to_csv(f"deepsort_{datasetName}{datasetType}_outputTest.csv", sep=',', header=columns_headers, index=False)
  

    video.release()
    out_video.release() 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    deep_sort_config_path = "deep_sort.yaml"
    track_objects(deep_sort_config_path)