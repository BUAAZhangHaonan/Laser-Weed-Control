# camera/coordinate_transform.py
import numpy as np
import yaml
import cv2
import os


class CoordinateTransformer:
    def __init__(self, calibration_file_path="config/calibration/calibrated_params.yaml"):
        """
        初始化坐标转换器。
        :param calibration_file_path: 标定参数 YAML 文件的相对路径 (相对于项目根目录)。
        """
        # 获取当前文件所在的目录 (LASER-WEED-CONTROL/camera/)
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取项目根目录 (LASER-WEED-CONTROL/)
        project_root_dir = os.path.dirname(current_file_dir)
        self.calibration_file_path = os.path.join(
            project_root_dir, calibration_file_path)

        # --- 用于完整 3D 转换的参数 (单位：通常像素用于相机矩阵，毫米用于深度和物理坐标) ---
        self.camera_matrix_rgb = None       # RGB 相机内参矩阵 (3x3 NumPy array)
        self.dist_coeffs_rgb = None         # RGB 相机畸变系数 (NumPy array)
        self.R_cam_to_laser = None          # 从相机到激光器的旋转矩阵 (3x3 NumPy array)
        # 从相机到激光器的平移向量 (3x1 NumPy array, 单位与激光器一致)
        self.T_cam_to_laser = None

        # --- 用于简化的 2D 像素到 2D 激光器坐标的映射 (初期使用) ---
        # 变换矩阵，例如通过 cv2.findHomography() 或 cv2.getAffineTransform() 计算得到
        # 如果是仿射变换，通常是 2x3；如果是单应性变换，是 3x3
        self.M_pixel_to_laser_2d = None     # 2D 变换矩阵

        self._load_calibration()

    def _load_calibration(self):
        """从 YAML 文件加载标定参数。"""
        print(f"坐标转换器：正在从 '{self.calibration_file_path}' 加载标定参数...")
        try:
            with open(self.calibration_file_path, 'r') as f:
                params = yaml.safe_load(f)
                if params is None:  # YAML 文件为空或无效
                    raise ValueError("标定文件为空或格式无效。")

            # --- 加载 RGB 相机内参和畸变 ---
            cam_matrix_data = params.get('camera_matrix_rgb', {}).get('data')
            if cam_matrix_data and len(cam_matrix_data) == 9:
                self.camera_matrix_rgb = np.array(
                    cam_matrix_data, dtype=np.float64).reshape((3, 3))
                print("  RGB 相机内参矩阵已加载。")
            else:
                print("警告：配置文件中未找到或格式不正确的 'camera_matrix_rgb'。3D转换将受影响。")

            dist_coeffs_data = params.get('dist_coeffs_rgb', {}).get('data')
            if dist_coeffs_data:  # 可以是4, 5, 8, 12 或 14 个元素
                self.dist_coeffs_rgb = np.array(
                    dist_coeffs_data, dtype=np.float64).flatten()
                print(f"  RGB 相机畸变系数已加载 (共 {len(self.dist_coeffs_rgb)} 个)。")
            else:
                self.dist_coeffs_rgb = np.zeros(
                    5, dtype=np.float64)  # 默认为无畸变 (k1,k2,p1,p2,k3)
                print("警告：配置文件中未找到 'dist_coeffs_rgb'，假设无畸变。")

            # --- 加载相机到激光器的外参 (用于3D转换) ---
            transform_params_3d = params.get(
                'camera_to_laser_transform_3d', {})
            R_data = transform_params_3d.get('R', {}).get('data')
            if R_data and len(R_data) == 9:
                self.R_cam_to_laser = np.array(
                    R_data, dtype=np.float64).reshape((3, 3))
                print("  相机到激光器旋转矩阵 (R) 已加载。")
            else:
                print("警告：配置文件中未找到或格式不正确的 'camera_to_laser_transform_3d.R'。3D转换将受影响。")

            T_data = transform_params_3d.get('T', {}).get('data')
            if T_data and len(T_data) == 3:
                self.T_cam_to_laser = np.array(
                    T_data, dtype=np.float64).reshape((3, 1))  # 确保是列向量
                print(f"  相机到激光器平移向量 (T) 已加载 (单位应与激光器一致)。")
            else:
                print("警告：配置文件中未找到或格式不正确的 'camera_to_laser_transform_3d.T'。3D转换将受影响。")

            # --- 加载简化的 2D 像素到激光器 XY 坐标的变换矩阵 ---
            transform_params_2d = params.get('pixel_to_laser_transform_2d', {})
            M_data = transform_params_2d.get('M', {}).get('data')  # 'M' 代表变换矩阵
            if M_data:
                M_rows = transform_params_2d.get('M', {}).get('rows')
                M_cols = transform_params_2d.get('M', {}).get('cols')
                if M_rows and M_cols and len(M_data) == M_rows * M_cols:
                    self.M_pixel_to_laser_2d = np.array(
                        M_data, dtype=np.float64).reshape((M_rows, M_cols))
                    print(f"  简易 2D 像素到激光器 XY 变换矩阵 ({M_rows}x{M_cols}) 已加载。")
                else:
                    print(
                        "警告：配置文件中 'pixel_to_laser_transform_2d.M' 的 data, rows, cols 不匹配或缺失。简易2D映射将受影响。")
            else:
                print("警告：配置文件中未找到 'pixel_to_laser_transform_2d.M'。简易2D映射将不可用。")

            print("坐标转换器：标定参数加载完成。")

        except FileNotFoundError:
            print(f"错误：标定文件未找到于 '{self.calibration_file_path}'")
            raise  # 重新抛出，让调用者知道出错了
        except Exception as e:
            print(f"错误：加载或解析标定文件 '{self.calibration_file_path}' 失败: {e}")
            import traceback
            traceback.print_exc()
            raise  # 重新抛出

    def _undistort_pixel(self, u, v):
        """内部方法：对单个像素点进行去畸变。"""
        if self.camera_matrix_rgb is None or self.dist_coeffs_rgb is None:
            # print("警告：相机内参或畸变系数未加载，无法去畸变，返回原始像素。")
            return float(u), float(v)

        pixel_coords_distorted = np.array(
            [[[float(u), float(v)]]], dtype=np.float32)
        # 使用 P=self.camera_matrix_rgb 可以让输出的坐标仍在像素尺度
        pixel_coords_undistorted = cv2.undistortPoints(
            pixel_coords_distorted,
            self.camera_matrix_rgb,
            self.dist_coeffs_rgb,
            None,  # R - 纠正变换（无）
            self.camera_matrix_rgb  # P - 新的相机矩阵（使用旧的，使输出仍在像素尺度）
        )
        return pixel_coords_undistorted[0, 0, 0], pixel_coords_undistorted[0, 0, 1]

    def pixel_to_camera_3d(self, u, v, depth_value_mm):
        """
        将去畸变后的 2D 像素坐标 (u, v) 和深度值（毫米）转换为相机坐标系下的 3D 点（毫米）。
        """
        if self.camera_matrix_rgb is None:
            print("错误：相机内参矩阵未加载。")
            return None
        if depth_value_mm is None or depth_value_mm <= 0:
            print(f"警告：无效的深度值 {depth_value_mm} 用于像素 ({u},{v})。")
            return None

        # 1. 对像素坐标进行去畸变
        u_undistorted, v_undistorted = self._undistort_pixel(u, v)

        # 2. 使用针孔相机模型反投影
        fx = self.camera_matrix_rgb[0, 0]
        fy = self.camera_matrix_rgb[1, 1]
        cx = self.camera_matrix_rgb[0, 2]
        cy = self.camera_matrix_rgb[1, 2]

        # Xc = (u_undistorted - cx) * depth / fx
        # Yc = (v_undistorted - cy) * depth / fy
        # Zc = depth
        Xc = (u_undistorted - cx) * depth_value_mm / fx
        Yc = (v_undistorted - cy) * depth_value_mm / fy
        Zc = float(depth_value_mm)  # 深度值单位为毫米，所以相机坐标也是毫米

        return np.array([Xc, Yc, Zc], dtype=np.float64)

    def camera_3d_to_laser_3d(self, camera_point_3d_mm):
        """
        将相机坐标系下的 3D 点（毫米）转换为激光器坐标系下的 3D 点（单位与T向量一致，通常也是毫米）。
        P_laser = R * P_camera + T
        """
        if self.R_cam_to_laser is None or self.T_cam_to_laser is None:
            print("错误：相机到激光器的外参 (R, T) 未加载。")
            return None
        if camera_point_3d_mm is None:
            return None

        if not isinstance(camera_point_3d_mm, np.ndarray):
            camera_point_3d_mm = np.array(camera_point_3d_mm, dtype=np.float64)

        if camera_point_3d_mm.shape == (3,):
            camera_point_3d_mm = camera_point_3d_mm.reshape((3, 1))
        elif camera_point_3d_mm.shape != (3, 1):
            print(
                f"错误：camera_point_3d_mm 必须是3元素向量或(3,1)数组, 实际为 {camera_point_3d_mm.shape}")
            return None

        laser_point_3d = (self.R_cam_to_laser @
                          camera_point_3d_mm) + self.T_cam_to_laser
        return laser_point_3d.flatten()  # 返回一个 1D 数组 [Xl, Yl, Zl]

    def pixel_to_laser_2d_direct(self, u, v):
        """
        (初期简化) 使用预加载的2D变换矩阵将像素坐标 (u,v) 直接映射到激光器的2D坐标 (Xl, Yl)。
        """
        if self.M_pixel_to_laser_2d is None:
            print("错误：简易 2D 像素到激光器 XY 变换矩阵 (M_pixel_to_laser_2d) 未加载。")
            return None

        pixel_homogeneous = np.array(
            [float(u), float(v), 1.0], dtype=np.float64)  # 齐次坐标

        if self.M_pixel_to_laser_2d.shape == (2, 3):  # 仿射变换
            laser_coord_xy = self.M_pixel_to_laser_2d @ pixel_homogeneous
        elif self.M_pixel_to_laser_2d.shape == (3, 3):  # 单应性变换
            transformed_homogeneous = self.M_pixel_to_laser_2d @ pixel_homogeneous
            if abs(transformed_homogeneous[2]) < 1e-6:  # 避免除以零或非常小的值
                print("警告：单应性变换导致投影尺度因子过小或为零。")
                return None
            laser_coord_xy = transformed_homogeneous[:2] / \
                transformed_homogeneous[2]
        else:
            print(f"错误：不支持的 2D 变换矩阵形状: {self.M_pixel_to_laser_2d.shape}")
            return None

        return laser_coord_xy[0], laser_coord_xy[1]  # 返回 Xl, Yl

    def get_target_pixel_from_bbox(self, detection_box):
        """
        从检测边界框确定目标像素 (u, v)。
        默认：边界框中心。
        detection_box 格式: [x_min, y_min, x_max, y_max]。
        """
        x_min, y_min, x_max, y_max = detection_box
        u = (x_min + x_max) / 2.0
        v = (y_min + y_max) / 2.0
        return int(round(u)), int(round(v))  # 四舍五入到最近整数

    def get_depth_for_pixel(self, u, v, aligned_depth_image_mm, window_size=5):
        """
        从对齐的深度图中为像素 (u, v)提取一个鲁棒的深度值（毫米）。
        通过取小窗口内的中值来处理无效深度值（例如0）和噪声。
        """
        if aligned_depth_image_mm is None:
            print("警告：对齐的深度图为空。")
            return None

        h, w = aligned_depth_image_mm.shape
        # 确保 u, v 是整数索引
        u_idx, v_idx = int(round(u)), int(round(v))

        if not (0 <= v_idx < h and 0 <= u_idx < w):
            print(f"警告：目标像素 ({u_idx},{v_idx}) 超出深度图边界 ({w}x{h})。")
            return None

        half_win = window_size // 2
        y_start, y_end = max(0, v_idx - half_win), min(h, v_idx + half_win + 1)
        x_start, x_end = max(0, u_idx - half_win), min(w, u_idx + half_win + 1)

        depth_patch = aligned_depth_image_mm[y_start:y_end, x_start:x_end]
        valid_depths = depth_patch[depth_patch > 0]  # 假设0是无效深度

        if valid_depths.size == 0:
            # 如果窗口内无有效深度，尝试获取单个像素的深度
            single_pixel_depth = aligned_depth_image_mm[v_idx, u_idx]
            if single_pixel_depth > 0:
                # print(f"  窗口无有效深度，使用单点深度: {float(single_pixel_depth)} mm")
                return float(single_pixel_depth)
            # print(f"警告：像素 ({u_idx},{v_idx}) 及其邻域均无有效深度。")
            return None

        return float(np.median(valid_depths))  # 返回中值

    def transform_detection_to_laser_coords(self, detection_box,
                                            aligned_depth_image_mm=None,
                                            mode="3d_if_possible"):
        """
        核心方法：将检测框转换为激光器坐标。
        :param detection_box: 检测框 [x_min, y_min, x_max, y_max]。
        :param aligned_depth_image_mm: 对齐的深度图（单位：毫米）。如果 mode="2d_direct"，则此参数可为 None。
        :param mode: "3d_if_possible" (默认) - 尝试3D转换，如果深度或参数不足则失败。
                     "2d_direct" - 强制使用简化的2D像素到激光器XY的直接映射。
                     "3d_only" - 强制使用3D转换，如果参数或深度不足则失败。
        :return: 如果是3D模式且成功，返回 (Xl, Yl, Zl)；
                 如果是2D模式且成功，返回 (Xl, Yl, None) 或 (Xl, Yl)；
                 如果失败，返回 None。
        """
        u, v = self.get_target_pixel_from_bbox(detection_box)
        print(f"坐标转换器：处理检测框在像素 ({u},{v}) 的目标。")

        if mode == "2d_direct":
            if self.M_pixel_to_laser_2d is not None:
                laser_xy = self.pixel_to_laser_2d_direct(u, v)
                if laser_xy:
                    print(
                        f"  2D直接映射结果: Xl={laser_xy[0]:.2f}, Yl={laser_xy[1]:.2f} (单位与标定一致)")
                    # 返回 (X, Y, Z=None) 表示是2D结果
                    return laser_xy[0], laser_xy[1], None
                else:
                    print("  2D直接映射失败。")
                    return None
            else:
                print("错误：请求了 '2d_direct' 模式，但简易2D变换矩阵未加载。")
                return None

        elif mode in ["3d_if_possible", "3d_only"]:
            if self.camera_matrix_rgb is None or self.R_cam_to_laser is None or self.T_cam_to_laser is None:
                print("错误：进行3D转换所需的相机内参或外参 (R_cam_to_laser, T_cam_to_laser) 未加载。")
                if mode == "3d_only":
                    return None
                # 如果是 "3d_if_possible"，可以考虑回退到2D，但这里我们先严格按模式
                return None  # 或者在这里尝试回退到2D模式 (如果2D参数存在)

            depth_value_mm = self.get_depth_for_pixel(
                u, v, aligned_depth_image_mm)
            if depth_value_mm is None:
                print(f"  未能获取像素 ({u},{v}) 的有效深度，无法进行3D转换。")
                return None
            # print(f"  获取到深度值: {depth_value_mm:.2f} mm")

            camera_point_3d = self.pixel_to_camera_3d(u, v, depth_value_mm)
            if camera_point_3d is None:
                # print(f"  像素 ({u},{v}) 到相机3D坐标转换失败。")
                return None
            # print(f"  相机坐标 (Xc,Yc,Zc): ({camera_point_3d[0]:.2f}, {camera_point_3d[1]:.2f}, {camera_point_3d[2]:.2f}) mm")

            laser_point_3d = self.camera_3d_to_laser_3d(camera_point_3d)
            if laser_point_3d is not None:
                print(
                    f"  3D转换结果: Xl={laser_point_3d[0]:.2f}, Yl={laser_point_3d[1]:.2f}, Zl={laser_point_3d[2]:.2f} (单位与T向量一致)")
                return laser_point_3d[0], laser_point_3d[1], laser_point_3d[2]
            else:
                # print("  相机3D坐标到激光器3D坐标转换失败。")
                return None
        else:
            print(f"错误：未知的转换模式 '{mode}'。")
            return None


# --- 示例用法和创建虚拟标定文件 ---
def _create_dummy_calibration_file_for_transformer(filepath="config/calibration/calibrated_params.yaml"):
    """内部辅助函数：为 CoordinateTransformer 测试创建一个虚拟标定文件。"""
    dummy_data = {
        'camera_matrix_rgb': {
            'rows': 3, 'cols': 3,
            'data': [600.0, 0.0, 320.0,  # fx, 0, cx
                     0.0, 600.0, 240.0,  # 0, fy, cy
                     0.0, 0.0, 1.0]
        },
        'dist_coeffs_rgb': {  # k1, k2, p1, p2, k3 (OpenCV基本模型)
            'rows': 1, 'cols': 5,
            'data': [0.01, -0.02, 0.001, 0.001, 0.005]  # 少量畸变
        },
        'camera_to_laser_transform_3d': {  # 用于3D转换
            'R': {  # 从相机到激光器的旋转
                'rows': 3, 'cols': 3,
                'data': [1.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         0.0, 0.0, 1.0]  # 单位旋转
            },
            'T': {  # 从相机原点到激光器原点的平移 (在相机坐标系下表达)
                'rows': 3, 'cols': 1,
                # 单位：毫米 (假设激光器在相机前方500mm, x+10, y+20)
                'data': [10.0, 20.0, 500.0]
            }
        },
        'pixel_to_laser_transform_2d': {  # 用于简化的2D直接映射
            'M': {  # 仿射变换矩阵 (2x3) 或 单应性矩阵 (3x3)
                'rows': 2, 'cols': 3,
                'data': [0.5, 0.0, -100.0,  # Xl = 0.5*u - 100
                         0.0, 0.5, -50.0]  # Yl = 0.5*v - 50
            }
        }
    }
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(dummy_data, f, default_flow_style=None, sort_keys=False)
    print(f"坐标转换器：虚拟标定文件已创建于 '{filepath}'")


if __name__ == '__main__':
    print("--- CoordinateTransformer 测试程序 ---")
    # 1. 创建一个虚拟的标定文件用于测试
    current_file_dir_ct = os.path.dirname(os.path.abspath(__file__))
    project_root_dir_ct = os.path.dirname(current_file_dir_ct)
    dummy_calib_file = os.path.join(
        project_root_dir_ct, "config/calibration/dummy_transformer_calib.yaml")
    _create_dummy_calibration_file_for_transformer(dummy_calib_file)

    try:
        transformer = CoordinateTransformer(
            calibration_file_path="config/calibration/dummy_transformer_calib.yaml")

        # --- 测试 1: 简化的 2D 直接映射 ---
        print("\n--- 测试 1: 简化 2D 直接映射 (pixel_to_laser_transform_2d.M) ---")
        # 假设 YOLO 检测到一个框，其左上角 (300,200)，右下角 (340,240) -> 中心 (320,220)
        detection_box_1 = [300, 200, 340, 240]
        # Xl = 0.5*320 - 100 = 160 - 100 = 60
        # Yl = 0.5*220 - 50  = 110 - 50  = 60
        # 期望输出: (60, 60, None)
        laser_coords_2d = transformer.transform_detection_to_laser_coords(
            detection_box_1, mode="2d_direct"
        )
        if laser_coords_2d:
            print(
                f"  检测框 {detection_box_1} (中心像素) --2D直接映射--> 激光器XY(Z): {laser_coords_2d}")
        else:
            print(f"  检测框 {detection_box_1} 2D直接映射失败。")

        # --- 测试 2: 完整的 3D 转换 ---
        print("\n--- 测试 2: 完整 3D 转换 ---")
        # 假设相机中心像素 (320, 240) 处，深度为 1000 mm
        detection_box_2 = [310, 230, 330, 250]  # 中心 (320,240)
        dummy_aligned_depth_mm = np.full(
            (480, 640), 0, dtype=np.uint16)  # 640x480
        dummy_aligned_depth_mm[235:245, 315:325] = 1000  # 在中心区域设置深度为1000mm

        laser_coords_3d = transformer.transform_detection_to_laser_coords(
            detection_box_2, dummy_aligned_depth_mm, mode="3d_if_possible"
        )
        if laser_coords_3d:
            print(
                f"  检测框 {detection_box_2} + 深度 --3D转换--> 激光器XYZ: {laser_coords_3d}")
            # 预期计算步骤：
            # 1. (320,240) 去畸变 (假设畸变后还是320,240因为系数小)
            # 2. pixel_to_camera_3d(320,240,1000):
            #    Xc = (320-320)*1000/600 = 0
            #    Yc = (240-240)*1000/600 = 0
            #    Zc = 1000
            #    P_cam = [0,0,1000]^T
            # 3. camera_3d_to_laser_3d(P_cam):
            #    P_laser = R*P_cam + T = I*P_cam + T = P_cam + T
            #            = [0,0,1000]^T + [10,20,500]^T = [10,20,1500]^T
            #    期望输出: (10.0, 20.0, 1500.0)
        else:
            print(f"  检测框 {detection_box_2} 3D转换失败。")

        # --- 测试 3: 3D 转换，但深度无效 ---
        print("\n--- 测试 3: 3D 转换但深度无效 ---")
        detection_box_3 = [10, 10, 20, 20]  # 中心 (15,15)
        # dummy_aligned_depth_mm 在 (15,15) 处为 0
        laser_coords_3d_no_depth = transformer.transform_detection_to_laser_coords(
            detection_box_3, dummy_aligned_depth_mm, mode="3d_if_possible"
        )
        if laser_coords_3d_no_depth is None:
            print("  深度无效时3D转换按预期失败。")
        else:
            print(f"  错误：深度无效时3D转换未失败，结果: {laser_coords_3d_no_depth}")

    except Exception as e_main:
        print(f"主测试程序发生错误: {e_main}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(dummy_calib_file):
            os.remove(dummy_calib_file)
            print(f"坐标转换器：已删除虚拟标定文件 '{dummy_calib_file}'")

    print("\n--- CoordinateTransformer 测试程序结束 ---")
