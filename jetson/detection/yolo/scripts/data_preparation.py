import os
import yaml
import shutil
import tqdm
from sklearn.model_selection import train_test_split


SOURCE_DATA = "/home/hdd3/zhanghaonan/s2r2025/S2R2025-Datasets/YOLO/yolo_dataset_0324_1139"
TRAIN_DATA = "/home/hdd3/zhanghaonan/s2r2025/S2R2025-Datasets/YOLO/yolo_dataset_0324_1139_train"
CLASSES_FILE = os.path.join(SOURCE_DATA, "classes.txt")


def prepare_yolo_dataset():
    """创建YOLO标准数据集结构"""
    splits = ['train', 'val']

    # 创建目录结构
    for split in splits:
        os.makedirs(os.path.join(TRAIN_DATA, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(TRAIN_DATA, 'labels', split), exist_ok=True)
        print(f"Create {split} directory done.")

    # 获取所有图片文件
    all_images = [f for f in os.listdir(os.path.join(SOURCE_DATA, "images"))
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 划分训练集和验证集，test_size规定划分比例
    train_files, val_files = train_test_split(
        all_images, test_size=0.2, random_state=42)

    # 文件复制函数
    def copy_files(files, split: str):
        for f in tqdm.tqdm(files):
            base_name = os.path.splitext(f)[0]

            # 复制图片
            src_img = os.path.join(SOURCE_DATA, "images", f)
            dst_img = os.path.join(TRAIN_DATA, "images", split, f)
            shutil.copy(src_img, dst_img)

            # 复制标签
            src_label = os.path.join(SOURCE_DATA, "labels", f"{base_name}.txt")
            dst_label = os.path.join(
                TRAIN_DATA, "labels", split, f"{base_name}.txt")
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)

        print(f"Copy {split} files done.")

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')

    return TRAIN_DATA


def generate_yolo_yaml(yolo_dir: str):
    """生成YOLO数据配置文件"""
    with open(CLASSES_FILE, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]

    config = {
        'path': yolo_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(classes)},
        'nc': len(classes)
    }

    yaml_path = os.path.join(yolo_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    print("Generate yaml file done.")


def validate_dataset():
    """验证数据集基本结构"""
    if not os.path.exists(SOURCE_DATA):
        raise ValueError(f"Dataset root path does not exist: {SOURCE_DATA}")

    if not os.path.exists(os.path.join(SOURCE_DATA, "images")):
        raise ValueError(f"Images directory not found in {SOURCE_DATA}")

    if not os.path.exists(os.path.join(SOURCE_DATA, "labels")):
        raise ValueError(f"Labels directory not found in {SOURCE_DATA}")

    if not os.path.exists(CLASSES_FILE):
        raise ValueError(f"Classes file not found: {CLASSES_FILE}")


def main():
    try:
        validate_dataset()
        yolo_dir = prepare_yolo_dataset()
        generate_yolo_yaml(yolo_dir)
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
