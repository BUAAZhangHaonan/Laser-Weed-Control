import os
import time
from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
from pathlib import Path
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

# 初始化YOLO模型
model = YOLO("weed.pt")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    """接收客户端上传的图片"""
    if 'file' not in request.files:
        return {'status': 'No file part'}, 400

    file = request.files['file']
    if file.filename == '':
        return {'status': 'No selected file'}, 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        # 执行YOLO检测（关键修改）
        results = model.predict(
            source=input_path,
            save=True,
            save_txt=True,
            project=RESULTS_FOLDER,
            name='predict',  # 固定输出子目录名称
            exist_ok=True    # 覆盖已有结果
        )

        return {'status': 'success', 'filename': filename}
    return {'status': 'Invalid file type'}, 400


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """提供检测结果文件下载"""
    txt_filename = f"{Path(filename).stem}.txt"
    # 修正路径到固定子目录
    txt_path = os.path.join(RESULTS_FOLDER, 'predict', 'labels', txt_filename)

    if os.path.exists(txt_path):
        return send_file(txt_path, as_attachment=True)
    return {'status': 'File not found'}, 404


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_FOLDER, 'predict'), exist_ok=True)
    app.run(host='192.168.50.241', port=5000, debug=True)
