# scripts/calibrate_simple_2d.py
import cv2
import numpy as np
import yaml
import os
import time
from datetime import datetime

from LaserWeedControl.camera.nyx_camera import NYXCamera


# --- 全局变量 ---
rgb_image_display = None
pixel_points = []       # 存储用户点击的像素坐标 [(u1,v1), (u2,v2), ...]
laser_points = []       # 存储对应的激光器坐标 [(Xl1,Yl1), (Xl2,Yl2), ...]
window_name = "2D 标定 - 点击图像选择点 (按 'q' 退出选择, 'c' 清除上一个点, 's' 保存)"
homography_matrix = None  # 计算得到的单应性矩阵


# --- 鼠标回调函数 ---
def mouse_callback(event, x, y):
    global rgb_image_display, pixel_points

    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_points.append((x, y))
        print(f"像素点 {len(pixel_points)} 已选择: ({x}, {y})")

        # 在图像上绘制标记
        cv2.circle(rgb_image_display, (x, y), 5, (0, 255, 0), -1)  # 绿色实心圆
        cv2.putText(rgb_image_display, str(len(pixel_points)), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # 红色数字
        cv2.imshow(window_name, rgb_image_display)


def get_laser_coordinates_from_user(pixel_point_index):
    """提示用户输入对应像素点的激光器坐标"""
    while True:
        try:
            input_str = input(
                f"  请输入像素点 {pixel_point_index} ({pixel_points[pixel_point_index-1]}) 对应的激光器坐标 (X Y, 用空格分隔): ")
            x_str, y_str = input_str.split()
            xl = float(x_str)
            yl = float(y_str)
            return xl, yl
        except ValueError:
            print("  输入无效，请输入两个用空格分隔的数字 (例如: 100.5 -50.0)。请重试。")
        except Exception as e:
            print(f"  发生错误: {e}。请重试。")


def save_calibration_data(matrix, filepath="config/calibration/calibrated_params.yaml"):
    """将计算得到的变换矩阵保存到 YAML 文件中"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(script_dir)))

    absolute_filepath = os.path.join(project_root, filepath)
    os.makedirs(os.path.dirname(absolute_filepath), exist_ok=True)

    data_to_save = {}
    # 尝试加载现有文件以保留其他参数
    if os.path.exists(absolute_filepath):
        try:
            with open(absolute_filepath, 'r') as f:
                data_to_save = yaml.safe_load(f)
                if data_to_save is None:
                    data_to_save = {}
        except Exception as e:
            print(f"警告：读取现有标定文件 '{absolute_filepath}' 失败: {e}。将创建新文件。")
            data_to_save = {}

    # 更新或添加 2D 变换矩阵
    if 'pixel_to_laser_transform_2d' not in data_to_save:
        data_to_save['pixel_to_laser_transform_2d'] = {}

    data_to_save['pixel_to_laser_transform_2d']['M'] = {
        'rows': matrix.shape[0],
        'cols': matrix.shape[1],
        'data': matrix.flatten().tolist()  # 将 NumPy 数组转换为列表
    }
    data_to_save['pixel_to_laser_transform_2d']['description'] = \
        "Homography matrix from pixel (u,v) to laser (Xl,Yl) coordinates, calculated on " + \
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_to_save['pixel_to_laser_transform_2d']['source_points_pixel'] = pixel_points
    data_to_save['pixel_to_laser_transform_2d']['destination_points_laser'] = laser_points

    try:
        with open(absolute_filepath, 'w') as f:
            yaml.dump(data_to_save, f, default_flow_style=None, sort_keys=False)
        print(f"\n简易 2D 标定矩阵已成功保存到: {absolute_filepath}")
        print("矩阵内容:")
        print(matrix)
    except Exception as e:
        print(f"\n错误：保存标定数据失败: {e}")


def test_homography(matrix_h):
    """测试模式：在图像上点击，显示预测的激光器坐标"""
    print("\n--- 进入测试模式 ---")
    print("在相机图像上点击任意点，将显示预测的激光器坐标。")
    print("按 'q' 退出测试模式。")

    test_window_name = "标定测试 - 点击查看预测 (按 'q' 退出)"
    cv2.namedWindow(test_window_name)

    temp_pixel_points_test = []  # 用于测试模式的点击

    def test_mouse_callback(event, x, y):
        nonlocal temp_pixel_points_test  # 允许修改外部函数的变量
        if event == cv2.EVENT_LBUTTONDOWN:
            temp_pixel_points_test.clear()  # 每次只处理一个点
            temp_pixel_points_test.append((x, y))

    cv2.setMouseCallback(test_window_name, test_mouse_callback)

    # 重新打开相机获取实时流进行测试
    with NYXCamera() as test_cam:
        if not test_cam.is_connected:
            print("测试模式错误：无法连接相机。")
            cv2.destroyWindow(test_window_name)
            return
        if not test_cam.start_streaming():
            print("测试模式错误：无法启动相机数据流。")
            cv2.destroyWindow(test_window_name)
            return

        while True:
            rgb_test, _ = test_cam.get_frames()
            if rgb_test is None:
                print("测试模式：无法获取相机帧，请检查相机。")
                time.sleep(0.1)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # 允许在此处也退出
                    break
                continue

            display_test_img = rgb_test.copy()

            if temp_pixel_points_test:
                u, v = temp_pixel_points_test[0]

                # 应用单应性变换
                pixel_h = np.array([u, v, 1], dtype=np.float64).reshape(3, 1)
                laser_h = matrix_h @ pixel_h
                if abs(laser_h[2, 0]) < 1e-6:
                    pred_xl_str, pred_yl_str = "Inf", "Inf"
                else:
                    pred_xl = laser_h[0, 0] / laser_h[2, 0]
                    pred_yl = laser_h[1, 0] / laser_h[2, 0]
                    pred_xl_str = f"{pred_xl:.2f}"
                    pred_yl_str = f"{pred_yl:.2f}"

                # 在图像上显示预测结果
                cv2.circle(display_test_img, (u, v),
                           7, (255, 0, 255), 2)  # 紫色圈
                text_to_show = f"Pixel:({u},{v}) -> Laser:({pred_xl_str},{pred_yl_str})"
                cv2.putText(display_test_img, text_to_show, (10, display_test_img.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(test_window_name, display_test_img)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break

    cv2.destroyWindow(test_window_name)
    print("--- 退出测试模式 ---")


def main():
    global rgb_image_display, pixel_points, laser_points, homography_matrix

    print("--- 简易 2D 像素到激光器坐标标定脚本 ---")
    print("说明:")
    print("  1. 将显示相机实时RGB图像。")
    print("  2. 在图像窗口中用鼠标左键点击至少4个不同的点。")
    print("  3. 对于每个选定的像素点，控制激光器打到物理世界中对应的位置。")
    print("  4. 在命令行中输入该位置的激光器坐标 (X Y)，按回车确认。")
    print("  5. 按 'c'键可以清除上一个选择的点和对应的激光器坐标。")
    print("  6. 选择完所有点后，按 's'键计算并保存标定矩阵。")
    print("  7. 按 'q'键可以中途退出点选择阶段（不会保存）。")
    print("-" * 40)

    # 初始化相机
    with NYXCamera() as cam:
        if not cam.is_connected:
            print("错误：无法连接到相机。请检查连接和SDK设置。")
            return
        if not cam.start_streaming():
            print("错误：无法启动相机数据流。")
            return

        print("\n相机已连接并开始推流。请在弹出的窗口中选择点。")
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)

        original_image_shape = None  # 用于测试模式

        while True:
            rgb_frame, _ = cam.get_frames()  # 我们只需要 RGB 帧
            if rgb_frame is None:
                print("警告：无法从相机获取RGB帧。")
                time.sleep(0.1)  # 等待一下再试
                # 允许在等待帧时按q退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("用户在等待帧时退出。")
                    break
                continue

            if original_image_shape is None:
                original_image_shape = rgb_frame.shape[:2]  # (height, width)

            # 每次循环都复制一份新的图像用于绘制，避免旧标记残留
            rgb_image_display = rgb_frame.copy()

            # 重绘已选择的点
            for i, (px, py) in enumerate(pixel_points):
                cv2.circle(rgb_image_display, (px, py), 5, (0, 255, 0), -1)
                cv2.putText(rgb_image_display, str(i + 1), (px + 10, py - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(window_name, rgb_image_display)
            key = cv2.waitKey(30) & 0xFF  # 等待30ms

            if key == ord('q'):  # 退出点选择
                print("用户选择退出点选择阶段。")
                break
            elif key == ord('c'):  # 清除上一个点
                if pixel_points:
                    removed_pixel = pixel_points.pop()
                    print(f"已清除像素点: {removed_pixel}")
                    if len(laser_points) == len(pixel_points) + 1:  # 如果激光点也已输入
                        removed_laser = laser_points.pop()
                        print(f"  同时清除对应的激光器坐标: {removed_laser}")
                else:
                    print("没有点可清除。")
            elif key == ord('s'):  # 保存并计算
                if len(pixel_points) < 4:
                    print(
                        f"错误：至少需要4个点对才能计算单应性矩阵。当前只有 {len(pixel_points)} 个像素点。")
                    print("请继续选择点，或者确保为已选像素点输入了激光器坐标。")
                    continue  # 继续选点

                # 确保为所有已选像素点获取激光器坐标
                while len(laser_points) < len(pixel_points):
                    print(f"\n--- 为像素点 {len(laser_points) + 1} 输入激光器坐标 ---")
                    try:
                        xl, yl = get_laser_coordinates_from_user(
                            len(laser_points) + 1)
                        laser_points.append((xl, yl))
                        print(f"  激光器坐标 {len(laser_points)} 已记录: ({xl}, {yl})")
                    except KeyboardInterrupt:
                        print("\n用户中断输入激光器坐标。标定将基于已有点进行。")
                        break  # 中断输入

                if len(pixel_points) != len(laser_points) or len(pixel_points) < 4:
                    print(
                        f"错误：像素点数量 ({len(pixel_points)}) 与激光器坐标点数量 ({len(laser_points)}) 不匹配，或点数少于4个。无法计算。")
                    print("请确保为所有像素点都输入了对应的激光器坐标。")
                else:
                    print("\n正在计算单应性矩阵...")
                    np_pixel_points = np.array(pixel_points, dtype=np.float32)
                    np_laser_points = np.array(laser_points, dtype=np.float32)

                    # 计算单应性矩阵 H，使得 laser_coord_homogeneous = H * pixel_coord_homogeneous
                    # 其中 pixel_coord 是 (u,v,1)T, laser_coord 是 (Xl,Yl,scale_factor)T
                    # 然后 Xl_norm = Xl/scale_factor, Yl_norm = Yl/scale_factor
                    homography_matrix, mask = cv2.findHomography(
                        np_pixel_points, np_laser_points, cv2.RANSAC, 5.0)

                    if homography_matrix is not None:
                        print("单应性矩阵计算成功。")
                        save_calibration_data(homography_matrix)

                        # 询问是否进入测试模式
                        test_choice = input(
                            "标定已保存。是否进入测试模式验证标定? (y/n): ").lower()
                        if test_choice == 'y' and original_image_shape is not None:
                            cam.stop_streaming()  # 先停止主循环的流
                            cv2.destroyWindow(window_name)  # 关闭标定窗口
                            test_homography(homography_matrix,
                                            original_image_shape)
                        break  # 计算完成并保存后退出主循环
                    else:
                        print("错误：单应性矩阵计算失败。请检查点是否共线或数量不足。")

        cv2.destroyAllWindows()
    print("\n--- 标定脚本结束 ---")


if __name__ == "__main__":
    main()
