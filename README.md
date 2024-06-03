# Class Distribution / Dataset Exploration

## IDD10_converted (as created by Kai on Kaggle)

### To use these scripts for dataset exploration

1. Change the paths to point to your dataset. This change needs to be made in explore_data.py
2. Run explore_data.py

### Class distribution on the Training Set
Total number of images: 31569

Class-wise distribution:
* Truck: 20,759 (5.74%)
* Person: 70,319 (19.45%)
* Autorickshaw: 24,498 (6.78%)
* Car: 65,676 (18.16%)
* Motorcycle: 78,119 (21.61%)
* Rider: 73,108 (20.22%)
* Bus: 13,829 (3.82%)
* Bicycle: 2,573 (0.71%)
* Traffic sign: 9,916 (2.74%)
* Traffic light: 2,780 (0.77%)

### Class distribution on the Validation Set
Total number of images: 10225

Class-wise distribution:
* Truck: 7,078 (5.97%)
* Car: 24,844 (20.97%)
* Person: 18,078 (15.26%)
* Motorcycle: 25,489 (21.51%)
* Rider: 24,518 (20.69%)
* Traffic sign: 4,287 (3.62%)
* Bus: 4,916 (4.15%)
* Autorickshaw: 7,782 (6.57%)
* Bicycle: 569 (0.48%)
* Traffic light: 919 (0.78%)

### Observations from the Training Set
* Major Classes: Motorcycle, Person, Rider, and Car remain the major classes, each comprising between 18% to 22% of the dataset.
* Minor Classes: Bicycle (0.71%) and Traffic light (0.77%) are still underrepresented compared to other classes.
* Intermediate Classes: Truck (5.74%), Autorickshaw (6.78%), Traffic sign (2.74%), and Bus (3.82%) have moderate representation.


# Training on IDD10_converted

## Plain YOLO (Pre-trained YOLOv8 nano)

### Configuration
* Epochs = 10
* Batch size = 16
* YOLO image size = 640 (resizing done automatically by YOLOv8)

### Metrics on the Validation Set

