import xml.etree.ElementTree as ET
import pandas as pd

def parse_annotation(xml_file, output_csv):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = []

    for track in root.findall('track'):
        track_id = track.get('id')  # 假设每个 track 有一个唯一的 ID
        label = track.get('label')

        for box in track.findall('box'):
            if box.get('outside') == '0':
                frame = box.get('frame')
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                if int(frame) >300:
                    break
                # 直接使用 xtl, ytl, xbr, ybr，不需要转换为中心坐标和宽高
                data.append({
                    "frame_id": frame,  # 假设每帧的图像文件名为 frame.jpg
                    "class_name": label,
                    "vehicle_id": track_id,
                    "x1": xtl,
                    "y1": ytl,
                    "x2": xbr,
                    "y2": ybr
                })

    # 将数据转换为 DataFrame 并保存为 CSV 文件
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    input_file = "dataset/Car Tracking & Object Detection/annotations.xml"
    output_file = "dataset/Car Tracking & Object Detection/annotations.csv"
    # 调用函数，处理注释并保存结果
    parse_annotation(input_file, output_file)
