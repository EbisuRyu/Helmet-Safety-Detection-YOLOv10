from ultralytics import YOLOv10
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Configuration for Validating YOLOv10 model')
    parser.add_argument('-m', '--model_path', type=str, default='./runs/detect/train/weights/best.pt',
                        help='Path to the trained YOLOv10 model file')
    parser.add_argument('-y', '--yaml_path', type=str, default='../datasets/safety_helmet_dataset/data.yaml',
                        help='Path to the YAML configuration file for the dataset')
    parser.add_argument('-s', '--imgsize', type=int, default=640,
                        help='Image size for validation (e.g., 640)')
    parser.add_argument('-t', '--split', type=str, default='test',
                        help='Data split to use for validation (e.g., test, val)')

    args = parser.parse_args()

    model = YOLOv10(args.model_path)

    model.val(data=args.yaml_path,
              imgsz=args.imgsize,
              split=args.split)


if __name__ == "__main__":
    main()
