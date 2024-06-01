import random
import os
from pathlib import Path
import matplotlib.pyplot as plt

from idd_converted.data_loader import IDDYOLODataset
from torch.utils.data import Dataset, DataLoader


# Function to plot class distribution
def plot_class_distribution(class_distribution, title):
    classes = list(class_distribution.keys())
    counts = list(class_distribution.values())

    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Number of Instances')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()


def display_random_sample(dataset):
    """
    Function to display a random sample from the dataset.
    To be used for reference purposes.
    """
    idx = random.randint(0, len(dataset) - 1)
    dataset.display_sample(idx)


# creating train and val data loaders
current_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.dirname(current_directory)
modified_dataset = Path(os.path.join(root_directory, 'idd_converted', 'IDD10_converted', 'images'))

train_dataset = IDDYOLODataset(
    images_folder=os.path.join(modified_dataset, 'train', 'images'),
    labels_folder=os.path.join(modified_dataset, 'train', 'labels'),
    class_mapping_file=os.path.join(modified_dataset, 'class_mapping.json')
)

val_dataset = IDDYOLODataset(
    images_folder=os.path.join(modified_dataset, 'val', 'images'),
    labels_folder=os.path.join(modified_dataset, 'val', 'labels'),
    class_mapping_file=os.path.join(modified_dataset, 'class_mapping.json')
)

# displaying a random sample from train and val
"""
print("Random sample from training dataset:")
display_random_sample(train_dataset)

print("Random sample from validation dataset:")
display_random_sample(val_dataset)
"""

# displaying a sample from train containing a certain class_id
"""
print("Random sample from training dataset containing class ID 9:")
train_dataset.display_sample_with_class(2)
"""

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Print total number of images in training and validation
print(f"Total number of images in the training dataset: {len(train_dataset)}")
print(f"Total number of images in the validation dataset: {len(val_dataset)}")

# calculate class-wise distribution
train_class_distribution = train_dataset.class_distribution()
val_class_distribution = val_dataset.class_distribution()

# print and plot class-wise distribution
print("Class-wise distribution in the training dataset:")
for class_name, count in train_class_distribution.items():
    print(f"{class_name}: {count}")
plot_class_distribution(train_class_distribution, 'Class Distribution in Training Dataset')

print("Class-wise distribution in the validation dataset:")
for class_name, count in val_class_distribution.items():
    print(f"{class_name}: {count}")
plot_class_distribution(val_class_distribution, 'Class Distribution in Validation Dataset')
