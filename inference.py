from ultralytics import YOLOv10
import cv2
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Configuration for Running YOLOv10 on an Image')
    parser.add_argument('-m', '--model_path', type=str,
                        default='./runs/detect/train/weights/best.pt', help='Path to the YOLOv10 model file')
    parser.add_argument('-i', '--image_path', type=str,
                        default='./images/demo/demo.jpg', help='Path to the input image')
    parser.add_argument('-o', '--output_path', type=str,
                        default='./images/results/demo.png', help='Path to save the output image')

    args = parser.parse_args()

    model = YOLOv10(args.model_path)
    result = model(source=args.image_path)[0]
    plot_result = result.plot()
    cv2.imshow('Result', plot_result)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
    result.save(args.output_path)


if __name__ == "__main__":
    main()
