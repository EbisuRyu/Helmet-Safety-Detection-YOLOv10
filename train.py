from ultralytics import YOLOv10
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Configuration for Training YOLOv10 model')
    parser.add_argument('-y', '--yaml_path', type=str, default='../datasets/safety_helmet_dataset/data.yaml',
                        help='Path to the YAML configuration file for the dataset')
    parser.add_argument('-m', '--model_path', type=str, default='./models/yolov10n.pt',
                        help='Path to the pre-trained YOLOv10 model file')
    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='Number of epochs for training')
    parser.add_argument('-s', '--imgsize', type=int, default=640,
                        help='Image size for training (e.g., 640)')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=54, help='Batch size for training')
    parser.add_argument('-p', '--project', type=str, default='./runs/detect',
                        help='Project directory to save training results')

    args = parser.parse_args()

    model = YOLOv10(args.model_path)
    model.train(data=args.yaml_path,
                epochs=args.epochs,
                batch=args.batch_size,
                imgsz=args.imgsize,
                project=args.project
                )


if __name__ == "__main__":
    main()
