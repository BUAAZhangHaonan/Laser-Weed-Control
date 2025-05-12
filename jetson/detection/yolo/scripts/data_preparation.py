import random
from pathlib import Path

# === 配置 ===
DATASET_DIR = Path(
    "/home/remote1/zhanghaonan/yolo-datasets/weed-dataset/datasets/all_fields_lincolnbeet/single_class")
OUTPUT_DIR = Path(
    "/home/remote1/zhanghaonan/yolo-datasets/weed-dataset/datasets/all_fields_lincolnbeet/config")
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# yaml 配置
YAML_FILENAME = OUTPUT_DIR / "configuration.yaml"
TRAIN_TXT = OUTPUT_DIR / "train.txt"
VAL_TXT = OUTPUT_DIR / "val.txt"
TEST_TXT = OUTPUT_DIR / "test.txt"

# 检查路径
assert DATASET_DIR.exists(), f"{DATASET_DIR} not found!"

# === 1. 遍历 PNG 文件 ===
image_files = list(DATASET_DIR.glob("*.png"))
print(f"Found {len(image_files)} images.")

valid_image_files = []

# === 2. 处理每个标签文件 ===
for img_path in image_files:
    label_path = img_path.with_suffix('.txt')
    if label_path.exists():
        new_lines = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    # 把 class_index 改为 0
                    parts[0] = '0'
                    new_line = ' '.join(parts)
                    new_lines.append(new_line)
        # 覆盖写回去
        with open(label_path, 'w') as f:
            for line in new_lines:
                f.write(line + '\n')
        valid_image_files.append(img_path)
    else:
        print(f"Label file {label_path} not found. Deleting image {img_path}.")
        img_path.unlink()  # 删除图片文件

# === 3. 打乱 & 划分数据集 ===
random.shuffle(valid_image_files)
total = len(valid_image_files)
train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_files = valid_image_files[:train_end]
val_files = valid_image_files[train_end:val_end]
test_files = valid_image_files[val_end:]

print(
    f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")


# === 4. 写入 txt 文件 ===
def write_txt(filelist, out_path):
    with open(out_path, 'w') as f:
        for img in filelist:
            f.write(str(img.resolve()) + '\n')


write_txt(train_files, TRAIN_TXT)
write_txt(val_files, VAL_TXT)
write_txt(test_files, TEST_TXT)

print(f"Written {TRAIN_TXT}, {VAL_TXT}, {TEST_TXT}")

# === 5. 写 yaml 文件 ===
yaml_content = f"""# YOLO Dataset Configuration
train: {TRAIN_TXT}
val: {VAL_TXT}
test: {TEST_TXT}

# number of classes
nc: 1

# class names
names: ["grass"]
"""

with open(YAML_FILENAME, 'w') as f:
    f.write(yaml_content)

print(f"YAML file written to {YAML_FILENAME}")
