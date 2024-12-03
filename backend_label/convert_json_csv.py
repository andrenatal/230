import json
import csv
import os

# Define the input and output file paths
#json_file_path = 'backend_label/uploads/federer/forehand_7.json'
#csv_file_path = 'forehand_federer-my.csv'
destination = "/media/4tbdrive/engines/cs230/dataset/debug/"
source = "/media/4tbdrive/engines/cs230/backend_label/uploads/"
for root, dirs, files in os.walk(source):
    for file in files:
        file_path = os.path.join(root, file)
        print(file_path)
        final_destination = file_path.replace(source, destination).replace(".json", ".csv")
        print(final_destination)
        directory = os.path.dirname(final_destination)
        if not os.path.exists(directory):
            os.makedirs(directory)


        # Define the order of keypoints as they appear in the CSV
        keypoint_names = [
            "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
            "right_knee", "left_ankle", "right_ankle"
        ]

        # Read the JSON file
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

        # Open the CSV file for writing
        with open(final_destination, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write the header row
            header = [
                "nose_y", "nose_x", "left_shoulder_y", "left_shoulder_x", "right_shoulder_y", "right_shoulder_x",
                "left_elbow_y", "left_elbow_x", "right_elbow_y", "right_elbow_x", "left_wrist_y", "left_wrist_x",
                "right_wrist_y", "right_wrist_x", "left_hip_y", "left_hip_x", "right_hip_y", "right_hip_x",
                "left_knee_y", "left_knee_x", "right_knee_y", "right_knee_x", "left_ankle_y", "left_ankle_x",
                "right_ankle_y", "right_ankle_x", "shot"
            ]
            csv_writer.writerow(header)

            # Write the data rows
            for frame in data:
                row = []
                for keypoint_name in keypoint_names:
                    keypoint = next((kp for kp in frame if kp["name"] == keypoint_name), None)
                    if keypoint:
                        row.extend([keypoint["y"], keypoint["x"]])
                    else:
                        row.extend([None, None])
                row.append(file.split("_")[0])
                csv_writer.writerow(row)

        print(f"CSV file has been saved to {final_destination}")
