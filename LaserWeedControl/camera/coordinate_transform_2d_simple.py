# camera/coordinate_transform_2d_simple.py
import cv2
import numpy as np

from LaserWeedControl.camera.nyx_camera import NYXCamera

# --- 全局变量 ---
WINDOW_NAME_CALIB = "简易2D标定 - 校准模式"
WINDOW_NAME_TEST = "简易2D标定 - 测试模式"
REFERENCE_PIXEL_POINTS_DEFAULT = [  # 预定义的4个参考像素点 (u,v)
    (100, 100),  # 左上
    (540, 100),  # 右上 (假设图像宽度约640)
    (100, 380),  # 左下 (假设图像高度约480)
    (540, 380)  # 右下
]
# 可以根据你的相机分辨率调整这些默认点

pixel_points_calib = []      # 存储标定过程中选择/确认的像素点
laser_points_calib = []      # 存储对应的激光器坐标
homography_matrix_2d = None  # 计算得到的单应性变换矩阵


# --- 辅助函数 ---
def draw_points(image, points, current_highlight_idx=None, color=(0, 255, 0), highlight_color=(0, 0, 255), radius=7, thickness=-1):
    """在图像上绘制点，并高亮显示当前点"""
    for i, (x, y) in enumerate(points):
        pt_color = highlight_color if i == current_highlight_idx else color
        cv2.circle(image, (x, y), radius, pt_color, thickness)
        cv2.putText(image, str(i + 1), (x + radius + 2, y - radius - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, pt_color, 1, cv2.LINE_AA)
    return image


def get_laser_coordinates_from_user_for_point(pixel_idx, pixel_coord):
    """提示用户为特定像素点输入激光器坐标"""
    while True:
        try:
            input_str = input(f"  请将激光器打到图像中的点 {pixel_idx+1} ({pixel_coord}) 对应的物理位置。\n"
                              f"  然后输入该位置的激光器坐标 (X Y, 用空格分隔): ")
            x_str, y_str = input_str.split()
            xl = float(x_str)
            yl = float(y_str)
            return xl, yl
        except ValueError:
            print("  输入无效，请输入两个用空格分隔的数字 (例如: 100.5 -50.0)。请重试。")
        except Exception as e:
            print(f"  发生错误: {e}。请重试。")


# --- 标定模式 ---
def run_calibration_mode(cam, reference_points):
    global pixel_points_calib, laser_points_calib, homography_matrix_2d
    print("\n--- 进入标定模式 ---")
    print(f"将在图像上显示 {len(reference_points)} 个参考点。")
    print("请按顺序，将激光器对准每个高亮的点，并输入其物理坐标。")
    print("按 'n' 进入下一个点，按 'q' 放弃标定并退出。")

    pixel_points_calib = list(reference_points)  # 使用预定义的参考点
    laser_points_calib = []
    homography_matrix_2d = None

    cv2.namedWindow(WINDOW_NAME_CALIB)
    current_point_idx = 0

    while current_point_idx < len(pixel_points_calib):
        rgb_frame, _ = cam.get_frames()
        if rgb_frame is None:
            print("标定模式：无法获取相机帧。")
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            continue

        display_img = rgb_frame.copy()
        display_img = draw_points(
            display_img, pixel_points_calib, current_highlight_idx=current_point_idx)

        # 显示提示信息
        text_y_offset = 30
        cv2.putText(display_img, f"标定点 {current_point_idx + 1}/{len(pixel_points_calib)}: ({pixel_points_calib[current_point_idx][0]},{pixel_points_calib[current_point_idx][1]})",
                    (10, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(display_img, "将激光打到此点, 然后在控制台输入激光器 (X,Y) 坐标。",
                    (10, text_y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(display_img, "按 'n' 确认当前点并到下一个, 'q' 退出。",
                    (10, text_y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME_CALIB, display_img)
        key = cv2.waitKey(1) & 0xFF  # 非阻塞等待

        if key == ord('q'):
            print("用户中止标定。")
            pixel_points_calib = []
            laser_points_calib = []
            cv2.destroyWindow(WINDOW_NAME_CALIB)
            return False  # 标定未完成

        elif key == ord('n'):  # 确认当前点，准备输入激光器坐标
            if current_point_idx < len(laser_points_calib):
                # 如果这个点已经有激光坐标了 (例如用户按了 'n' 两次)
                print(f"点 {current_point_idx+1} 的激光坐标已存在，将移动到下一个未标定的点。")
                current_point_idx += 1
                while current_point_idx < len(pixel_points_calib) and current_point_idx < len(laser_points_calib):
                    current_point_idx += 1
                if current_point_idx >= len(pixel_points_calib):
                    print("所有点都已有激光坐标。尝试计算矩阵。")
                continue

            print(f"\n--- 正在标定点 {current_point_idx + 1} ---")
            xl, yl = get_laser_coordinates_from_user_for_point(
                current_point_idx, pixel_points_calib[current_point_idx])
            laser_points_calib.append((xl, yl))
            print(
                f"  记录激光器坐标: ({xl}, {yl}) for pixel ({pixel_points_calib[current_point_idx]})")
            current_point_idx += 1
            if current_point_idx == len(pixel_points_calib):
                print("\n所有参考点的激光器坐标已输入完毕。")

    cv2.destroyWindow(WINDOW_NAME_CALIB)

    if len(pixel_points_calib) == len(laser_points_calib) and len(pixel_points_calib) >= 4:
        print("\n正在计算单应性变换矩阵...")
        np_pixel_points = np.array(pixel_points_calib, dtype=np.float32)
        np_laser_points = np.array(laser_points_calib, dtype=np.float32)

        homography_matrix_2d, mask = cv2.findHomography(
            np_pixel_points, np_laser_points, cv2.RANSAC, 5.0)

        if homography_matrix_2d is not None:
            print("单应性矩阵计算成功:")
            print(homography_matrix_2d)
            return True  # 标定和计算完成
        else:
            print("错误：单应性矩阵计算失败。请检查点是否共线或数量不足。")
            return False
    else:
        print("错误：点数不足或像素点与激光器点数不匹配，无法计算变换矩阵。")
        return False


# --- 测试模式 ---
test_mode_current_pixel = None  # 用于存储鼠标点击的像素点


def test_mode_mouse_callback(event, x, y, flags, param):
    global test_mode_current_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        test_mode_current_pixel = (x, y)
        print(f"测试模式：选中像素点 ({x},{y})")


def run_test_mode(cam, matrix_h):
    global test_mode_current_pixel
    if matrix_h is None:
        print("测试模式错误：单应性变换矩阵 (Homography) 未计算或无效。请先完成标定。")
        return

    print("\n--- 进入测试模式 ---")
    print("在相机图像上点击任意点，将在命令行显示预测的激光器坐标。")
    print("按 'q' 退出测试模式。")

    cv2.namedWindow(WINDOW_NAME_TEST)
    cv2.setMouseCallback(WINDOW_NAME_TEST, test_mode_mouse_callback)
    test_mode_current_pixel = None  # 重置

    while True:
        rgb_frame, _ = cam.get_frames()
        if rgb_frame is None:
            print("测试模式：无法获取相机帧。")
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            continue

        display_img = rgb_frame.copy()

        if test_mode_current_pixel is not None:
            u, v = test_mode_current_pixel

            # 应用单应性变换
            pixel_h_coords = np.array(
                [u, v, 1.0], dtype=np.float64).reshape(3, 1)
            laser_h_coords = matrix_h @ pixel_h_coords

            pred_xl_str, pred_yl_str = "N/A", "N/A"
            if abs(laser_h_coords[2, 0]) > 1e-7:  # 避免除以太小的值
                pred_xl = laser_h_coords[0, 0] / laser_h_coords[2, 0]
                pred_yl = laser_h_coords[1, 0] / laser_h_coords[2, 0]
                pred_xl_str = f"{pred_xl:.2f}"
                pred_yl_str = f"{pred_yl:.2f}"
                print(
                    f"  像素 ({u},{v}) -> 预测激光器坐标 (Xl, Yl): ({pred_xl_str}, {pred_yl_str})")
            else:
                print(f"  像素 ({u},{v}) -> 预测激光器坐标: 变换导致尺度因子过小或为零。")

            # 在图像上绘制标记和预测结果
            cv2.circle(display_img, (u, v), 7, (255, 0, 255), 2)  # 点击点标记
            text_to_show = f"P:({u},{v}) -> L:({pred_xl_str},{pred_yl_str})"
            cv2.putText(display_img, text_to_show, (10, display_img.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME_TEST, display_img)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyWindow(WINDOW_NAME_TEST)
    print("--- 退出测试模式 ---")


# --- 主函数 ---
def main():
    global homography_matrix_2d
    print("--- 简易2D坐标变换与标定脚本 ---")

    # 调整参考点以适应常见的640x480图像，并向内收缩一些
    # 你可以根据你的相机分辨率和期望的标定区域调整这些点
    image_width, image_height = 640, 480  # 假设的相机分辨率
    margin = 50  # 边距
    calib_reference_points = [
        (margin, margin),
        (image_width - margin, margin),
        (margin, image_height - margin),
        (image_width - margin, image_height - margin)
    ]
    print(f"使用的默认参考像素点: {calib_reference_points}")
    print("如果你的相机分辨率不同，请修改脚本中的 `image_width`, `image_height` 和 `margin`。")

    # 使用 with 语句确保相机资源被正确释放
    try:
        with NYXCamera() as cam:
            if not cam.is_connected:
                print("错误：无法连接到相机。脚本将退出。")
                return
            if not cam.start_streaming():
                print("错误：无法启动相机数据流。脚本将退出。")
                return

            while True:
                print("\n请选择操作模式:")
                print("  1. 开始/重新进行2D标定")
                print("  2. 进入测试模式 (需要先完成标定)")
                print("  q. 退出程序")
                choice = input("请输入选项: ").strip().lower()

                if choice == '1':
                    calibration_successful = run_calibration_mode(
                        cam, calib_reference_points)
                    if calibration_successful:
                        print("标定完成并成功计算了变换矩阵。")
                        # 询问是否立即进入测试
                        test_now = input("是否立即进入测试模式? (y/n): ").lower()
                        if test_now == 'y':
                            run_test_mode(cam, homography_matrix_2d)
                    else:
                        print("标定未完成或计算失败。")
                elif choice == '2':
                    if homography_matrix_2d is not None:
                        run_test_mode(cam, homography_matrix_2d)
                    else:
                        print("错误：没有可用的标定矩阵。请先运行标定模式 (选项 1)。")
                elif choice == 'q':
                    print("正在退出程序...")
                    break
                else:
                    print("无效选项，请重新输入。")

    except Exception as e:
        print(f"程序发生未处理的异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()  # 确保所有OpenCV窗口都关闭
        print("\n--- 脚本执行完毕 ---")


if __name__ == "__main__":
    main()
