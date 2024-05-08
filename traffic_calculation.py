import pandas as pd
import matplotlib.pyplot as plt

def calculate_traffic_volume_and_speed(csv_path, frame_window):
    data = pd.read_csv(csv_path)
    data['x_center'] = (data['x1'] + data['x2']) / 2
    data['y_center'] = (data['y1'] + data['y2']) / 2
    data['time_window'] = data['frame_id'] // frame_window

    # label the previous frame id for each vehicle
    data['prev_frame_id'] = data.groupby('vehicle_id')['frame_id'].shift(1)
    data['continuous_presence'] = (data['frame_id'] - data['prev_frame_id'] == 1)

    # calculate the distance between the current and previous frame
    data['prev_x_center'] = data.groupby('vehicle_id')['x_center'].shift(1)
    data['prev_y_center'] = data.groupby('vehicle_id')['y_center'].shift(1)
    data['distance'] = ((data['x_center'] - data['prev_x_center']) ** 2 + (data['y_center'] - data['prev_y_center']) ** 2) ** 0.5

    # calculate the valid distance based on continuous presence
    data['valid_distance'] = data['distance'] * data['continuous_presence']

    # calculate the average speed per time window
    speed_per_window = data.groupby('time_window')['valid_distance'].mean().fillna(0)

    # calculate the number of unique vehicles per time window
    unique_vehicles_per_window = data.groupby('time_window')['vehicle_id'].nunique()

    return unique_vehicles_per_window, speed_per_window,data

def save_and_plot_traffic_data(groundtruth_data, prediction_data, groundtruth_speed, prediction_speed, output_path,dataset_name):
    # save the data to a CSV file
    combined_data = pd.DataFrame({
        'Groundtruth Traffic Volume': groundtruth_data,
        'Prediction Traffic Volume': prediction_data,
        'Groundtruth Average Speed': groundtruth_speed,
        'Prediction Average Speed': prediction_speed
    })
    combined_data.to_csv(output_path)

    # plot traffic volume
    plt.figure(figsize=(10, 5))
    plt.plot(combined_data.index, combined_data['Groundtruth Traffic Volume'], label='Groundtruth Traffic Volume', color='tab:red', marker='o')
    plt.plot(combined_data.index, combined_data['Prediction Traffic Volume'], label='Prediction Traffic Volume', color='tab:blue', linestyle='--', marker='x')
    plt.title(f'Car Count at Times for {dataset_name}')
    plt.xlabel('Time Window')
    plt.ylabel('Number of Unique Vehicles')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path.replace('.csv', '_volume.png'))
    plt.show()

    # plot average speed
    plt.figure(figsize=(10, 5))
    plt.plot(combined_data.index, combined_data['Groundtruth Average Speed'], label='Groundtruth Average Speed', color='tab:green', marker='o')
    plt.plot(combined_data.index, combined_data['Prediction Average Speed'], label='Prediction Average Speed', color='tab:orange', linestyle='--', marker='x')
    plt.title(f'Average Speed at Times for {dataset_name} (pixels/frame)')
    plt.xlabel('Time Window')
    plt.ylabel('Average Speed (pixels/frame)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path.replace('.csv', '_speed.png'))
    plt.show()

if __name__ == "__main__":
    dataset_name = 'Car Tracking & Object Detection Dataset'
    groundtruth_csv_path = 'output/tracking_groundtruth_image/results.csv'
    prediction_csv_path = 'output/tracking_yolo_image_x/results.csv'

    # dataset_name = 'Harpy Data Vehicle Dataset'
    # groundtruth_csv_path = 'output/tracking_groundtruth_video/results.csv'
    # prediction_csv_path = 'output/tracking_yolo_video_x/results.csv'

    frame_window = 30
    groundtruth_data, groundtruth_speed,data1= calculate_traffic_volume_and_speed(groundtruth_csv_path, frame_window)
    prediction_data, prediction_speed,data2 = calculate_traffic_volume_and_speed(prediction_csv_path, frame_window)
    data1.to_csv('output/traffic_analysis/groundtruth_data.csv')
    data2.to_csv('output/traffic_analysis/prediction_data.csv')

    output_path = 'output/traffic_analysis/traffic_volume_and_speed_analysis.csv'
    save_and_plot_traffic_data(groundtruth_data, prediction_data, groundtruth_speed, prediction_speed, output_path,dataset_name)