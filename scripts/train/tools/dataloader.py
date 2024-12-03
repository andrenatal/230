import os
import numpy as np
import json

class DataLoader:
    def check_float_array(self,array):
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError(f"Array contains non-float data: {array.dtype}")

    def normalize_input(self, input_data, mean_X, std_X):
        return (input_data - mean_X) / std_X

    def load_data_openpose(self, time_step_size):
        X = []
        Y = []
        folder_path = "/mnt/crucial1tb_ssd/cs230/data/dataset/Tennis Player Actions Dataset for Human Pose Estimation/annotations"
        poses_per_swing = {}
        for filename in os.listdir(folder_path):
            json_data = []
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    json_data.append(data)
                    poses_per_swing[filename] = json_data
        for key in poses_per_swing:
            for idx in range(0,len(poses_per_swing[key][0]["annotations"]),time_step_size):
                temp_X = []
                for i in range(idx,idx+time_step_size):
                    temp_X.append(poses_per_swing[key][0]["annotations"][i]["keypoints"])
                X.append(temp_X)
                if poses_per_swing[key][0]["annotations"][idx]["category_id"] == 5: # backhand
                    Y.append(0)
                if poses_per_swing[key][0]["annotations"][idx]["category_id"] == 6: # forehand
                    Y.append(1)
                if poses_per_swing[key][0]["annotations"][idx]["category_id"] == 7: # ready position
                    Y.append(2)
                if poses_per_swing[key][0]["annotations"][idx]["category_id"] == 8: # serve
                    Y.append(3)
        X = np.array(X)
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0)
        std_X[std_X == 0] = 1
        np.save("models/openpose/means/mean_X.npy", mean_X)
        np.save("models/openpose/means/std_X.npy", std_X)
        self.X = self.normalize_input(X, mean_X, std_X)
        self.Y = np.array(Y)
        print("Checking for NaN values in X:", np.isnan(self.X).any())
        print("Checking for infinite values in X:", np.isinf(self.X).any())

    def load_data_movenet(self, dataset):
        X = []
        Y = []
        self.poses_per_swing_per_player = {}
        folder_path = "/media/4tbdrive/engines/cs230/dataset/" + dataset
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for dir in dirnames:
                if dir == "shots":
                    continue
                #if "nadal" in dir:
                #    continue
                temp_poses_per_swing_per_player = {
                    "backhand" : [],
                    "forehand" : [],
                    "neutral": [],
                    "serve" : []
                }
                for filename in os.listdir(os.path.join(dirpath, dir)):
                    if filename.endswith('.csv'):
                        file_path = os.path.join(dirpath, dir, filename)
                        with open(file_path, 'r') as file:
                            temp_X = []
                            for line in file:
                                if line.startswith("nose_y"):
                                    continue
                                data = line.split(",")[:-1]
                                data = [float(x) for x in data]
                                temp_X.append(data)
                                for element in data[:-1]:
                                    if not isinstance(element, float):
                                        raise TypeError(f"List contains non-float data: {type(element)}")
                            X.append(temp_X)
                            temp_poses_per_swing_per_player[filename.split("_")[0]].append(temp_X)
                            if "backhand" in filename:
                                Y.append(0)
                            if "forehand" in filename:
                                Y.append(1)
                            if "neutral" in filename:
                                Y.append(2)
                            if "serve" in filename:
                                Y.append(3)
                self.poses_per_swing_per_player[dir] = temp_poses_per_swing_per_player
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.check_float_array(self.X)
        print("Length of X:", len(X))
        print("Length of Y:", len(Y))