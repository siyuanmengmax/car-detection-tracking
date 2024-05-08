import pandas as pd

df = pd.read_csv('dataset/Harpy Data Vehicle/DJI_0406_tracks.csv')

df['x1'] = df['bboxX']
df['y1'] = df['bboxY']
df['x2'] = df['bboxX'] + df['Width']
df['y2'] = df['bboxY'] + df['Height']


output_df = df[['frame', 'veh_id', 'x1', 'y1', 'x2', 'y2']].copy()
output_df = output_df[output_df['frame'] <= 1798]
output_df.columns = ['frame_id', 'vehicle_id', 'x1', 'y1', 'x2', 'y2']


output_path = 'dataset/Harpy Data Vehicle/annotations.csv'
output_df.to_csv(output_path, index=False)
