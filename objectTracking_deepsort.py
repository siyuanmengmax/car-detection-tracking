import cv2
import torch
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import pandas as pd
import numpy as np



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
    vidPath = 'Harpy Data Vehicle/DJI_0406_full.mp4'
    video = cv2.VideoCapture(vidPath)
    filePath = 'Harpy Data Vehicle/DJI_0406_tracks.csv'
    data = pd.read_csv(filePath, delimiter=',')
    #print(data)
    bboxes = data.loc[:, ['frame','bboxX','bboxY','Width','Height']]
    #print(bboxes)
    confidences = torch.ones(bboxes.shape[0])
    #frames = data.loc[:, ['frame']].to_numpy()
   
    tracking_info = []
    #f = open("harpy_deepsort.csv", "w")
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        current_frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES)) -1
        print(current_frame_number)
        boxes = bboxes.loc[bboxes['frame']==current_frame_number].values
        #print(boxes.shape)
        
        outputs = deepsort.update(torch.tensor(boxes[:,1:]), confidences[0:len(boxes[0])], frame)
        tracking_info.append(outputs)
        print(outputs)
        np.savetxt("harpy_deepsort.csv", outputs, delimiter=",")

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    deep_sort_config_path = "deep_sort.yaml"
    track_objects(deep_sort_config_path)