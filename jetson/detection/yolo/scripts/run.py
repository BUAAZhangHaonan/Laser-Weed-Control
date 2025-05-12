import subprocess


def train_class(class_id):
    cmd = [
        "python3",
        "/home/remote1/zhanghaonan/projects/jetson/detection/yolo/scripts/train.py",
        "--class_id",
        str(class_id)
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    for class_id in [0, 1]:
        train_class(class_id)


if __name__ == "__main__":
    main()
