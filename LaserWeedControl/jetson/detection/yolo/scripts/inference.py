from ultralytics import YOLO

# Load a model
model = YOLO(
    "/home/remote1/zhanghaonan/projects/laser-weed-control/checkpoint/yolov11/b/yolo11s_b.pt")

# Run batched inference on a list of images
results = model(
    ["/home/remote1/zhanghaonan/yolo-datasets/weed-dataset/datasets/all_fields_lincolnbeet/all/bbro_bbro_07_05_2021_v_0_0.png"])

# Process results list
for i, result in enumerate(results):
    boxes = result.boxes
    result.save(filename=f"results/result_{i}.jpg")
