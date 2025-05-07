import torch
from ultralytics.models import YOLO


DATA_YAML = "/home/hdd3/zhanghaonan/s2r2025/S2R2025-Datasets/YOLO/yolo_dataset_0324_1139_train/data.yaml"
PRETRAINED_WEIGHTS = "/home/hdd3/zhanghaonan/s2r2025/yolov12/checkpoints/yolov12s.pt"
EPOCHS = 200
BATCH_SIZE = 6*64
IMG_SIZE = 640
DEVICE = [0, 1]


def main():
    assert torch.cuda.is_available(), "CUDA is not available!"
    model = YOLO(model=PRETRAINED_WEIGHTS)
    model.train(data=DATA_YAML,
                epochs=EPOCHS,
                batch=BATCH_SIZE,
                imgsz=IMG_SIZE,
                device=DEVICE,
                workers=8,
                optimizer="AdamW",
                lr0=1e-3
                )


if __name__ == "__main__":
    main()
