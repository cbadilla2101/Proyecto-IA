import os
import cv2
import json
import numpy as np
import torch


class TUSimpleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path="TUSimple", train=True, size=(512, 256)):
        self._dataset_path = dataset_path
        self._mode = "train" if train else "test"
        self._image_size = size
        self._data = []

        suffixes = []
        if self._mode == "train":
            suffixes = ["0313", "0531"]
        elif self._mode == "test":
            suffixes = ["0601"]

        label_files = [
            os.path.join(self._dataset_path, "train_set", f"label_data_{suffix}.json")
            for suffix in suffixes
        ]

        for label_file in label_files:
            self._process_label_file(label_file)

    def __getitem__(self, idx):
        image_path = os.path.join(self._dataset_path, "train_set", self._data[idx][0])
        image = cv2.imread(image_path)
        h, w, c = image.shape
        image = cv2.resize(image, self._image_size, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[..., None]
        lanes = self._data[idx][1]

        image = torch.from_numpy(image).float().permute((2, 0, 1))

        segmentation_image = self._draw(h, w, lanes, "segmentation")
        segmentation_image = torch.from_numpy(segmentation_image.copy())
        segmentation_image = segmentation_image.to(torch.int64)

        instance_image = self._draw(h, w, lanes, "instance")
        instance_image = instance_image[..., None]
        instance_image = torch.from_numpy(instance_image.copy()).permute((2, 0, 1))

        return image, segmentation_image, instance_image

    def __len__(self):
        return len(self._data)

    def _draw(self, h, w, lanes, image_type):
        image = np.zeros((h, w), dtype=np.uint8)
        for i, lane in enumerate(lanes):
            color = 1 if image_type == "segmentation" else i + 1
            cv2.polylines(image, [lane], False, color, 10)

        image = cv2.resize(image, self._image_size, interpolation=cv2.INTER_NEAREST)

        return image

    def _process_label_file(self, file_path):
        with open(file_path) as f:
            for line in f:
                info = json.loads(line)
                image = info["raw_file"]
                lanes = info["lanes"]
                h_samples = info["h_samples"]
                lanes_coords = []
                for lane in lanes:
                    x = np.array([lane]).T
                    y = np.array([h_samples]).T
                    xy = np.hstack((x, y))
                    idx = np.where(xy[:, 0] > 0)
                    lane_coords = xy[idx]
                    lanes_coords.append(lane_coords)
                self._data.append((image, lanes_coords))
