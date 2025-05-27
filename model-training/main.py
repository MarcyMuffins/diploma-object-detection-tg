import os
from ultralytics import YOLO

def main():
    print("=== YOLOv11 Training Interface ===")
    print("This script will train a YOLOv11 model using your specified settings.\n")

    # Get user input
    print("=== Available models ===")
    print("yolo11n.pt: Nano – fastest, lowest accuracy")
    print("yolo11s.pt: Small – balance of speed and accuracy")
    print("yolo11m.pt: Medium – slower, higher accuracy")
    print("yolo11l.pt: Large – slowest, best accuracy")

    model_size = input("Enter YOLOv11 model size (e.g., 'yolo11s.pt', 'yolo11m.pt', etc.): ").strip()
    data_yaml = input("Enter path to your data.yaml file: ").strip()
    while not os.path.isfile(data_yaml):
        print(f"File '{data_yaml}' not found. Please try again.")
        data_yaml = input("Enter path to your data.yaml file: ").strip()

    try:
        epochs = int(input("Enter number of epochs (e.g., 1, 10, 50): ").strip())
    except ValueError:
        print("Invalid input. Using default: 1 epoch.")
        epochs = 1

    try:
        batch = int(input("Enter batch size (e.g., 8, 16, 32): ").strip())
    except ValueError:
        print("Invalid input. Using default: batch size 8.")
        batch = 8

    print("\nInitializing training process...")
    print(f"Model: {model_size}")
    print(f"Data: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch}\n")

    # Load and train the model
    try:
        model = YOLO(model_size)
        results = model.train(data=data_yaml, epochs=epochs, batch=batch)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return

    # Print summary
    print("\nTraining complete.")
    print(f"Results saved in: {results.save_dir}")

if __name__ == "__main__":
    main()
