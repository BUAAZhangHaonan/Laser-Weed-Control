# train.py
import argparse
import torch
from ultralytics.models import YOLO


DATA_YAML = "/home/remote1/zhanghaonan/yolo-datasets/weed-dataset/datasets/all_fields_lincolnbeet/config/configuration.yaml"
PRETRAINED_WEIGHTS = "/home/remote1/zhanghaonan/projects/checkpoint/yolov11/yolo11s.pt"
EPOCHS = 1000
BATCH_SIZE = 64
IMG_SIZE = 640
DEVICE = "0, 1"


def main(class_id):
    assert torch.cuda.is_available(), "CUDA is not available!"
    model = YOLO(model=PRETRAINED_WEIGHTS)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        workers=8,
        optimizer="AdamW",
        lr0=5e-4,
        patience=20,
        save_period=200,
        classes=[class_id],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_id", type=int, required=True,
                        help="Class ID to train (e.g., 0 or 1)")
    args = parser.parse_args()

    main(args.class_id)
