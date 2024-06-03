import os
from ultralytics import YOLO
import torch

def main():
    # Path to your dataset and YAML configuration file
    dataset_path = os.path.abspath("data")
    yaml_path = os.path.join(dataset_path, "config.yaml")

    print(f"Dataset path: {dataset_path}")
    print(f"YAML path: {yaml_path}")

    # Ensure the GPU is being used
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize the model
    model = YOLO("yolov8n.pt")  # Use the pre-trained YOLOv8 nano model as a base

    # Check if a checkpoint exists
    checkpoint_path = "checkpoint.pt"
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # Total number of epochs we want to train for
    total_epochs = 10

    # Train the model with checkpoints
    for epoch in range(start_epoch, total_epochs):
        print(f"################################### Epoch {epoch + 1} begins ###################################.")
        model.train(data=yaml_path, epochs=1, imgsz=640, batch=16, device=device)
        print(f"################################### Epoch {epoch + 1} ends ###################################.")

        # Save the checkpoint after each epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.model.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    # Save the final model
    model.save("best_model.pt")
    print("Final model saved as best_model.pt")

    # Evaluate the model on the validation set
    val_results = model.val(data=yaml_path)
    print("Validation Results:", val_results)

    """
    # Evaluate the model on the test set
    test_images_path = os.path.join(dataset_path, "images", "test", "images")
    print(f"Test images path: {test_images_path}")
    test_results = model.predict(source=test_images_path, save=True)  # save=True will save predictions

    # Print results
    print("Test Results:", test_results)
    """

if __name__ == '__main__':
    main()
