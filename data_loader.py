import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from tqdm import tqdm


class IDDYOLODataset(Dataset):
    def __init__(self, images_folder, labels_folder, class_mapping_file=None):
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]

        if class_mapping_file:
            with open(class_mapping_file, 'r') as f:
                class_mapping = json.load(f)
            self.class_mapping = {v: k for k, v in class_mapping.items()}  # Reverse the mapping
        else:
            self.class_mapping = None

    def _load_image_and_labels(self, img_name):
        img_path = os.path.join(self.images_folder, img_name)
        label_path = os.path.join(self.labels_folder, img_name.replace('.jpg', '.txt'))

        image = Image.open(img_path).convert('RGB')
        orig_width, orig_height = image.size

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    x_center *= orig_width
                    y_center *= orig_height
                    width *= orig_width
                    height *= orig_height
                    boxes.append([x_center, y_center, width, height])
                    labels.append(int(class_id))

        boxes = np.array(boxes)
        labels = np.array(labels)

        return image, boxes, labels, img_name, label_path

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image, boxes, labels, img_name, label_path = self._load_image_and_labels(img_name)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        targets = {
            'boxes': boxes,
            'labels': labels
        }

        return image, targets, img_name, label_path

    def display_sample(self, idx):
        image, targets, img_name, label_path = self.__getitem__(idx)
        boxes = targets['boxes'].numpy()
        labels = targets['labels'].numpy()

        # Print filenames
        print(f"Image file: {img_name}")
        print(f"Label file: {label_path}")

        # Convert the image to PIL format
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()  # Use default font

        for box, label in zip(boxes, labels):
            x_center, y_center, width, height = box
            xmin = x_center - width / 2
            ymin = y_center - height / 2
            xmax = x_center + width / 2
            ymax = y_center + height / 2
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)

            if self.class_mapping:
                class_name = self.class_mapping[label]
            else:
                class_name = str(label)

            draw.text((xmin, ymin), class_name, fill="red", font=font)

        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def display_sample_with_class(self, class_id):
        for idx in range(len(self)):
            _, targets, _, _ = self.__getitem__(idx)
            if class_id in targets['labels'].numpy():
                self.display_sample(idx)
                break

    def class_distribution(self):
        class_counts = defaultdict(int)
        for img_name in tqdm(self.image_files, desc="Calculating class distribution"):
            _, _, labels, _, _ = self._load_image_and_labels(img_name)
            for label in labels:
                class_counts[label] += 1
        if self.class_mapping:
            class_counts_named = {self.class_mapping[label]: count for label, count in class_counts.items()}
        else:
            class_counts_named = class_counts
        return class_counts_named
