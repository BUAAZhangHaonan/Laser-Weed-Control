import os
import requests
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

SERVER_URL = "http://192.168.50.241:5000"
INPUT_DIR = "input"
OUTPUT_DIR = "output"


class FileHandler(FileSystemEventHandler):
    """监控input目录的新增文件"""

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.jpeg')):
            print(f"Detected new file: {event.src_path}")
            # 增加延迟处理（关键修改）
            time.sleep(0.1)  # 等待文件完全释放
            self.retry_upload(event.src_path)

    def retry_upload(self, filepath, max_retries=5):
        """带重试机制的文件上传"""
        for attempt in range(max_retries):
            try:
                with open(filepath, 'rb') as f:
                    files = {'file': (os.path.basename(filepath), f)}
                    response = requests.post(
                        f"{SERVER_URL}/upload", files=files)
                    if response.status_code == 200:
                        print(f"Uploaded: {filepath}")
                        filename = response.json()['filename']
                        self.check_result(filename)
                        return
            except PermissionError:
                print(
                    f"Retrying {filepath} (attempt {attempt+1}/{max_retries})")
                time.sleep(2)
            except Exception as e:
                print(f"Error: {str(e)}")
                break
        print(f"Failed to upload: {filepath}")

    def check_result(self, filename):
        """结果检查"""
        txt_name = os.path.splitext(filename)[0] + ".txt"
        for _ in range(10):  # 保持原有重试逻辑
            time.sleep(2)    # 延长等待时间
            response = requests.get(f"{SERVER_URL}/download/{filename}")
            if response.status_code == 200:
                save_path = os.path.join(OUTPUT_DIR, txt_name)
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                print(f"Received result: {save_path}")
                return
        print(f"Timeout waiting for result: {filename}")


if __name__ == '__main__':
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, INPUT_DIR, recursive=False)
    observer.start()

    print(f"Monitoring directory: {INPUT_DIR}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
