# 激光除草项目技术难点

目前最核心的未决技术点集中在精确地将YOLO检测到的像素目标，转化为激光器能够理解和打击的物理坐标，以及 提升运动中图像的质量。

## 相机标定 (极其关键)

这是连接“看到”和“打到”的桥梁。你需要两种标定：

### 相机内参标定 (Intrinsic Calibration)

1. 目的: 获取每个NYX 660相机的内部光学参数，包括焦距 (fx, fy)、主点 (cx, cy) 和畸变系数（径向和切向）。这些参数是进行精确的 "2D像素 + 深度 -> 3D相机坐标" 转换的基础。
2. 方法:
   - 检查SDK/文档: 首先确认NYX 660的SDK或文档是否提供了每台相机的精确内参。有些工业相机出厂时会进行标定并将数据存储在设备内或随设备提供。如果能直接获取并验证有效，可以省去手动标定。
   - 手动标定: 如果无法直接获取或需要更高精度/验证，你需要使用标定板（如棋盘格、圆点阵列或ChArUco板）和OpenCV库中的 cv2.calibrateCamera 函数。
     - 在相机视野内从不同角度和距离拍摄标定板的多张清晰图像（RGB图即可）。
     - 使用OpenCV找到标定板角点的像素坐标。
     - 提供角点对应的物理世界坐标（通常定义在标定板平面上，Z=0）。
     - cv2.calibrateCamera 会计算出相机矩阵（包含fx, fy, cx, cy）和畸变系数。
3. 重要性: 即使深度与RGB已对齐，内参也是将像素坐标反投影回相机空间的基础。对齐只保证了(u,v)处的深度值对应同一个空间点，但计算这个空间点的(Xc, Yc, Zc)坐标需要内参。

### 相机-激光器外参标定 (Extrinsic Calibration / Hand-Eye Calibration)

1. 目的: 确定相机坐标系与激光器（打标机）坐标系之间的相对姿态（旋转矩阵 R）和平移向量 T。这样，你才能将在相机坐标系下计算出的杂草3D坐标，转换成激光器能理解的目标坐标。
挑战: 激光器不像机械臂末端那样容易安装标定板。你需要一种方法让相机“看到”激光器“指示”的点，并且知道这个点在激光器坐标系下的坐标。
2. 方法
   - 方法一：打标辅助标定
     1. 定义激光器坐标系: 明确激光打标机的坐标系原点和方向（通常在其工作平面的中心或某个基准点）。
     2. 创建标定物: 准备一个平整的板材（如亚克力板、阳极氧化铝板），放置在激光器的工作范围内，并且能被相机清晰看到。
     3. 激光打标: 使用激光器的控制软件，精确地在板材上打出一系列已知坐标点 (Xl, Yl) （在激光器坐标系下）。例如，打出一个网格或者几个特定标记点。注意：这里的Zl可能需要根据激光器的焦距/工作距离来确定或假定在一个平面上。
     4. 相机拍摄: 用NYX 660相机拍摄这个带有激光标记的板材。确保能同时清晰看到标定板的特征（如果板材本身就是标定板，如带有图案）和激光打出的标记。
     5. 图像处理:
        - 在RGB图像中，精确检测出激光标记点的像素坐标 (u, v)。
        - 读取这些像素点对应的深度值 d。
        - 使用已标定的相机内参，将这些标记点的 (u, v, d) 转换为相机坐标系下的3D坐标 (Xc, Yc, Zc)。
     6. 求解变换: 现在你有多对对应的点：激光器坐标系下的 (Xl, Yl, Zl) 和相机坐标系下的 (Xc, Yc, Zc)。利用这些对应点对，可以使用算法（如OpenCV的 cv2.solvePnP 如果你知道板材在相机坐标系下的位姿，或者更通用的点云配准算法如SVD求解刚体变换）来计算出从激光器坐标系到相机坐标系的旋转矩阵 R_laser_to_cam 和平移向量 T_laser_to_cam。你需要的是其逆变换 R_cam_to_laser 和 T_cam_to_laser 来将相机坐标转换到激光器坐标。
   - 方法二：工具头辅助标定 (如果可行)
     - 如果可以在激光头附近（或者一个与激光器坐标系有已知固定关系的位置）临时安装一个小的视觉标记（如ArUco标记），并且可以用相机看到它。同时，你需要知道这个标记中心在激光器坐标系下的精确坐标。通过检测这个标记在相机中的位姿，可以直接建立相机到激光器的变换关系。这种方法可能在初始设置时更方便，但要求精确安装标记。
3. 精度要求: 外参标定的精度直接决定了激光打击的准确性，需要非常仔细地进行。
4. 实现库: OpenCV 提供了标定和坐标变换所需的大部分函数。

## 坐标转换算法 (核心计算逻辑)

这个算法将在Jetson Orin上运行，紧跟在YOLO检测之后：

- 输入:
  - YOLO检测到的杂草边界框 (BBox)。
  - 对应的RGB图像和深度图像 (已对齐)。
  - 相机内参 (fx, fy, cx, cy)。
  - 相机到激光器的外参 (R_cam_to_laser, T_cam_to_laser)。
- 步骤:
  1. 选择目标点像素坐标 (u, v): 从BBox中选择一个代表性的打击点。通常是BBox的中心点 (u = bbox_x + bbox_width/2, v = bbox_y + bbox_height/2)。也可以考虑质心或其他策略。
  2. 获取深度值 d: 读取深度图像在像素 (u, v) 处的深度值 d。
  注意: 深度图可能有噪声或无效值 (0 或 NaN)。需要进行有效性检查和滤波。可以考虑取 (u, v) 邻域内有效深度值的中值或均值，以提高稳定性。
  单位确认: 确认深度值的单位（通常是毫米或米），与内参的单位匹配。
  3. 像素坐标到相机坐标系3D点 (Xc, Yc, Zc): 使用针孔相机模型反投影公式：
  ```Xc = (u - cx) * d / fx```
  ```Yc = (v - cy) * d / fy```
  ```Zc = d```
  得到杂草目标点在相机坐标系下的三维坐标 P_camera = [Xc, Yc, Zc]^T。
  4. 相机坐标系到激光器坐标系 (Xl, Yl, Zl): 应用外参进行刚体变换：
  ```P_laser = R_cam_to_laser * P_camera + T_cam_to_laser```
  其中 P_laser = [Xl, Yl, Zl]^T 就是最终要发送给激光器控制软件的坐标。
  5. 激光器坐标系适应性调整:
  - 锥形范围: 了解激光打标机接受的坐标是2D (Xl, Yl) 还是3D (Xl, Yl, Zl)。
  - 如果接受3D: 发送计算出的 (Xl, Yl, Zl)。要确保这个Zl在激光器的有效工作深度范围内。
  - 如果只接受2D (Xl, Yl): 这意味着激光器可能工作在一个固定的焦平面或者自动对焦。你需要确认计算出的目标点是否接近这个平面。如果偏差太大，可能需要调整相机安装高度或只打击特定深度范围内的杂草。或者，你需要知道如何将 (Xl, Yl, Zl) 投影到激光器的工作平面上得到指令 (Xl', Yl')。这取决于激光器控制软件的具体工作方式。
  - 单位转换: 确保最终发送的坐标单位与激光器控制软件要求的单位一致（如毫米）。
- 实现: 使用 Python (配合NumPy进行矩阵运算) 或 C++ (配合Eigen库) 在Jetson上实现。

## 图像电子稳像 (EIS - Electronic Image Stabilization)

当小车移动时，相机的抖动会导致图像模糊和目标定位不准。EIS的目标是在软件层面补偿这种抖动。

1. 目的: 获得更清晰、更稳定的图像流输入给YOLO，并可能用于补偿坐标计算中的延迟。
2. 方法:
   - 基于特征点的稳像:
     1. 在连续的RGB图像帧之间，检测并匹配特征点（如ORB, SIFT, SURF，但ORB在嵌入式设备上速度更快）。
     2. 根据匹配的特征点，估计两帧之间的几何变换（通常是仿射变换或单应性变换）。
     3. 应用该变换（或其逆变换）到当前帧，使其与前一帧（或参考帧）对齐。
     4. 库: OpenCV 的 cv2.ORB, cv2.BFMatcher 或 cv2.FlannBasedMatcher, cv2.estimateAffine2D 或 cv2.findHomography, cv2.warpAffine 或 cv2.warpPerspective。
   - 基于光流的稳像:
   1. 计算连续帧之间的稠密光流（如Farneback）或稀疏光流（如Lucas-Kanade）。
   2. 从光流场估计全局运动。
   3. 补偿运动。
      - 库: OpenCV 的 cv2.calcOpticalFlowFarneback 或 cv2.calcOpticalFlowPyrLK。
   - 陀螺仪/IMU辅助稳像 (如果硬件支持): 如果相机或小车有IMU，可以直接读取角速度数据，积分得到角度变化，用于更快、更鲁棒地估计和补偿旋转抖动。这是效果通常最好的方法之一，但依赖硬件。检查NYX 660是否带IMU，或者是否可以在小车上加装一个靠近相机的IMU并将数据同步。
3. 挑战:
   - 实时性: EIS计算本身有开销，需要在Jetson Orin上高效实现，不能显著拖慢整个处理流程。可能需要GPU加速（OpenCV的CUDA模块支持部分功能）。
   - 效果: 对于剧烈或快速的非刚性运动（如果小车颠簸得很厉害），EIS效果可能有限。
   - 运动模糊: EIS主要解决帧间抖动，对于帧内运动模糊（快门时间长导致拖影），效果有限。需要配合尽可能短的曝光时间。
4. 集成位置: EIS应该在YOLO检测之前对输入的RGB图像进行处理。

下一步行动建议:

1. 优先攻克标定: 这是整个系统的精度基础
   - 研究NYX 660的文档，看能否获取内参。如果不能，准备标定板，编写或使用现有工具进行内参标定。
   - 设计并实施相机-激光器外参标定流程。这是最关键的一步。准备好标定物，编写或调用代码来完成计算。仔细验证标定结果。
2. 实现坐标转换逻辑: 在Jetson上编写代码，实现从像素+深度到激光器坐标的完整转换链。使用标定得到的内外参。
3. 测试基础功能: 在静态（小车不动）情况下，测试整个流程：检测杂草 -> 计算坐标 -> 发送给上位机 -> 激光器打击。验证打击精度。
4. 开发和集成EIS: 在基础功能调通后，再根据实际运行时图像抖动情况，选择合适的EIS方法进行开发和集成，优化动态性能。测试EIS对检测效果和最终打击精度的提升。
5. 考虑鲁棒性:
   - 深度数据处理: 添加处理深度图中无效值和噪声的逻辑。
   - 目标选择: 如果同时检测到多个杂草，需要有优先级策略（例如，打最近的？打最中间的？）。
   - 安全区域: 在上位机或Jetson端增加逻辑，确保计算出的打击坐标在预设的安全工作范围内，避免打到作物或超出激光器范围。

## 20250506

### 1. 项目现状回顾

- **已完成:**
  - 深度相机 (NYX 660)、Jetson AGX Orin、Windows 上位机选型。
  - YOLO 模型训练 (能够识别杂草)。
  - 基础的 HTTP 通信结构 (用于文件传输，但性能不满足实时需求)。
  - 激光打标机及其控制软件 (可通过坐标指令控制打击)。
  - 项目基础目录结构搭建。
- **待完成 (当前阶段核心):**
  - 相机标定 (内参、畸变、相机到激光器的外参)。
  - 可靠、高效的通信机制 (切换到 TCP/IP Socket)。
  - 基于深度和标定参数的坐标转换算法。
  - 图像电子稳像 (EIS) 算法。
  - 各模块在 Jetson 和 Windows 上的集成与串联。
- **待迁入 (已有代码):**
  - Jetson 上的 YOLO 推理代码 (位于另一个仓库，将迁入 `jetson/detection/`)。
  - Windows 上的激光器控制 C++ 代码 (位于另一个仓库，将通过 Python 封装调用或直接替换 `windows/laser_control/` 目录下的内容)。

## 2. 核心开发模块与文件

我们将按照项目结构分解开发任务：

- `common/`: 定义共享的消息格式。
- `config/`: 存放配置和标定参数。
- `scepter-sdk/`: 相机 SDK 文件 (外部依赖)。
- `camera/`: 相机交互、原始数据处理和几何计算。
  - `nyx_camera.py`: 相机 SDK 封装，数据获取，参数读取。
  - `image_stabilizer.py`: 图像电子稳像实现。
  - `coordinate_transform.py`: 坐标转换算法实现。
- `scripts/`: 独立的辅助脚本，特别是标定脚本。
  - `calibrate_intrinsics.py`: 相机内参标定脚本。
  - `calibrate_extrinsics.py`: 相机-激光器外参标定脚本。
- `jetson/`: Jetson 平台主程序及相关模块。
  - `main.py`: Jetson 主程序入口，串联各模块。
  - `communication/tcp/server.py`: 新的 TCP 服务器实现。
  - `detection/`: YOLO 推理模块 (待迁入)。
- `windows/`: Windows 平台主程序及相关模块。
  - `main.py`: Windows 主程序入口，串联各模块。
  - `communication/tcp/client.py`: 新的 TCP 客户端实现。
  - `laser_control/`: 激光器控制封装 (待迁入 C++ 或在此进行 Python 封装)。
  - `safety/`: 安全检查模块。

## 3. 详细开发流程规划

我们将按照功能依赖关系，分阶段进行开发：

### 阶段 1: 相机基础交互与数据获取 (`camera/nyx_camera.py`)

- **目标:** 成功连接相机，获取原始 RGB-D 图像流，并尝试读取相机出厂的标定参数。
- **任务:**
  1. 研究 NYX 660 SDK 文档，理解初始化、连接、数据流控制的 API。
  2. 在 `camera/nyx_camera.py` 中编写 `NYXCamera` 类。
  3. 实现相机连接 (`connect`, `disconnect`)。
  4. 实现数据流的启动和停止 (`start_streaming`, `stop_streaming`)。
  5. 实现单帧或连续帧的获取 (`get_frame`)。确保能获取到原始 RGB 图像和原始深度图像。
     1. **关键:** 调用 SDK API (参考 `DeviceParamSetGet` 例程) 尝试获取相机出厂的内参矩阵、畸变系数以及 RGB 与 Depth 传感器的相对外参。实现 `get_calibration_params()` 方法。
     2. **关键:** 调用 SDK API (参考 `TransformDepthImgToColorSensorFrame` 例程) 实现将深度图对齐到 RGB 图像空间的功能。`get_frame()` 方法的返回应包含对齐后的深度图。
- **验证:**
  - 运行测试代码，确认相机连接成功，能够稳定获取 RGB 和深度图像。
  - 打印或保存获取到的相机参数，检查其格式和有效性。
  - 可视化对齐后的深度图，与 RGB 图对比，初步检查对齐效果。

### 阶段 2: 相机标定 (`scripts/` 目录)

- **目标:** 获取精确的相机内参、畸变参数，以及最关键的相机坐标系到激光器坐标系的外部参数。将结果保存在 `config/calibration/calibrated_params.yaml`。
- **任务:**
    1. **如果阶段 1 获取的内参不准确或缺失:**
        - 编写 `scripts/calibrate_intrinsics.py` 脚本。
        - 使用 `camera/nyx_camera.py` 的 `get_frame` 或类似的捕获功能，拍摄棋盘格或其他标定板在不同角度和距离下的 RGB 图像集。
        - 使用 OpenCV 库 (如 `cv2.calibrateCamera`) 对图像集进行处理，计算相机内参矩阵和畸变系数。
        - 将计算结果保存到 `config/calibration/calibrated_params.yaml`。
    2. **实现相机到激光器外参标定:**
        - **这是精度关键。** 选择一种标定方法 (例如，激光打标辅助法)。
        - 编写 `scripts/calibrate_extrinsics.py` 脚本。
        - 脚本需要引导用户或自动化执行标定步骤 (例如，控制激光器在标定板上打点，同时用相机拍摄)。
        - 使用 `camera/nyx_camera.py` 获取带有激光点和/或标定板特征的 RGB-D 数据。
        - 编写图像处理代码来识别激光点在 RGB 图像中的像素坐标，并获取对应的对齐深度值。
        - 利用已知点在激光器坐标系下的位置，以及它们在相机坐标系下的 3D 位置 (通过像素+深度和相机内参计算)，求解相机坐标系到激光器坐标系的旋转矩阵 (R) 和平移向量 (T)。可以使用 OpenCV (如 `cv2.solvePnP` 的变种或更通用的方法)。
        - 将计算得到的 R 和 T (或其逆变换) 保存到 `config/calibration/calibrated_params.yaml`。
- **验证:**
  - 检查标定结果文件 (`calibrated_params.yaml`) 的格式是否正确。
  - 通过逆投影或重投影测试标定结果的精度 (例如，将几个已知激光器坐标点转换到相机坐标系，再投影回图像，看是否与实际像素位置吻合)。
  - 可能需要在实际场景中进行初步的打点测试来验证外参的准确性。

### 阶段 3: 几何坐标转换 (`camera/coordinate_transform.py`)

- **目标:** 实现将 YOLO 检测到的像素坐标结合深度信息，精确转换为激光器坐标系下的 3D 点。
- **任务:**
    1. 在 `camera/coordinate_transform.py` 中编写 `CoordinateTransformer` 类。
    2. 构造函数加载 `config/calibration/calibrated_params.yaml` 中的相机内外参和畸变参数。
    3. 实现 `pixel_to_camera_3d(u, v, depth_value, camera_matrix, dist_coeffs)` 方法，将像素坐标和深度值转换到相机坐标系下的 3D 点。需要考虑畸变校正。
    4. 实现 `camera_3d_to_laser_3d(camera_point_3d, R_cam_to_laser, T_cam_to_laser)` 方法，将相机坐标系下的 3D 点转换到激光器坐标系。
    5. 实现核心方法 `transform_detection(detection_box, aligned_depth_image)`。
        - 接收 YOLO 输出的检测框 (BBox)。
        - 从 BBox 中确定一个用于打击的像素点 (u, v) (例如 BBox 中心)。
        - 在对齐后的深度图上获取 (u, v) 处的深度值。需要处理深度图中的无效值和噪声 (例如，取局部窗口的有效深度中值)。
        - 调用 `pixel_to_camera_3d` 计算相机坐标。
        - 调用 `camera_3d_to_laser_3d` 计算激光器坐标。
        - 返回激光器坐标 (Xl, Yl, Zl)。
- **验证:**
  - 编写单元测试，使用已知的像素-深度-3D点对应关系 (例如，从标定板获取的点) 来验证转换计算的正确性。
  - 在 Jetson 上运行，结合阶段 1 的相机数据，测试 `transform_detection` 方法，输出计算出的 3D 坐标。

### 阶段 4: 图像电子稳像 (`camera/image_stabilizer.py`)

- **目标:** 在小车移动时，减少相机抖动对图像质量的影响。
- **任务:**
    1. 在 `camera/image_stabilizer.py` 中编写 `ImageStabilizer` 类。
    2. 选择一种 EIS 算法 (例如，基于 ORB 特征点检测和仿射变换估计)。
    3. 实现 `process_frame(rgb_frame)` 方法，接收原始 RGB 帧，计算帧间运动，并应用补偿变换，返回稳定后的 RGB 帧。
- **验证:**
  - 使用录制的或实时的抖动视频进行测试。
  - 可视化稳像前后的图像，观察效果。
  - 评估算法在 Jetson 上的实时性能开销。

### 阶段 5: 可靠 TCP/IP 通信 (`jetson/communication/tcp/server.py`, `windows/communication/tcp/client.py`, `common/messages.py`)

- **目标:** 替换原有的 HTTP 文件传输，建立稳定、高效的 TCP Socket 通信通道，用于传输结构化消息 (命令和坐标)。
- **任务:**
    1. 在 `common/messages.py` 中定义 `TargetCoordinateMsg` 和 `CommandMsg` 类，包含所需的字段 (坐标、ID、置信度、时间戳、命令类型等)。实现 `to_dict`, `from_dict`, `serialize_message`, `deserialize_message` 函数 (使用 JSON + UTF-8 编码)。
    2. 学习 Python `socket` 和 `struct` 库。
    3. 在 `jetson/communication/tcp/server.py` 中实现 TCP Server：
        - 创建 socket, 绑定 IP/Port, 监听。
        - 接受客户端连接。在一个单独的线程或协程中处理连接。
        - 实现带长度前缀的消息接收循环 (用于接收命令)，调用反序列化。
        - 提供一个公共方法 `send_coordinates(msg)` 供外部调用，实现消息序列化、长度前缀打包、通过 socket 发送。
        - 处理连接断开和错误。
    4. 在 `windows/communication/tcp/client.py` 中实现 TCP Client：
        - 创建 socket, 连接到 Server IP/Port。
        - 在一个单独的线程或协程中处理接收循环。
        - 实现带长度前缀的消息接收循环 (用于接收坐标)，调用反序列化，并将收到的消息传递给 Windows 主程序。
        - 提供一个公共方法 `send_command(msg)` 供外部调用，实现消息序列化、长度前缀打包、通过 socket 发送。
        - 实现连接断开和自动重连。
- **验证:**
  - 编写独立的测试脚本，运行 Server 和 Client，发送和接收简单的测试消息，验证连接建立、双向通信、消息分帧和序列化/反序列化是否正常。

### 阶段 6: Jetson 端集成 (`jetson/main.py`)

- **目标:** 在 Jetson 主程序中串联相机数据获取、稳像、检测、坐标转换和 TCP 服务器发送流程。
- **任务:**
    1. 在 `jetson/main.py` 中，初始化 `NYXCamera`, `ImageStabilizer`, `CoordinateTransformer`, `tcp/server` 的实例。
    2. 加载 `config/jetson_config.yaml` 获取配置参数 (如相机 IP, TCP 端口等)。
    3. 启动 `tcp/server` 的通信线程。
    4. 启动 `NYXCamera` 的数据流。
    5. 实现主处理循环：
        - 调用 `NYXCamera.get_frame()` 获取最新 RGB-D 帧。
        - 将 RGB 帧传递给 `ImageStabilizer.process_frame()` 获取稳定帧。
        - 将稳定帧传递给 YOLO 检测模块 (`jetson/detection/`) 获取检测结果 (BBoxes)。 (此部分依赖 YOLO 代码迁入)
        - 遍历检测结果，对每个 BBox：
            - 调用 `CoordinateTransformer.transform_detection()`，传入 BBox 和对齐深度图，获取激光器坐标 (Xl, Yl, Zl)。
            - 创建 `TargetCoordinateMsg` 实例。
            - 调用 `tcp/server` 实例的发送方法，将消息发送给 Windows。
        - 处理从 `tcp/server` 接收到的命令消息，根据命令控制主循环的状态 (开始/停止等)。
    6. 添加必要的错误处理和资源释放逻辑。
- **验证:**
  - 在 Jetson 上运行 `main.py`，在 Windows 上运行测试客户端或即将开发的完整客户端。
  - 确认 Jetson 能够稳定获取数据，检测到目标，计算出坐标，并通过 TCP 发送给 Windows。
  - 在 Windows 端接收并打印坐标信息，检查数据格式和内容是否正确。

### 阶段 7: Windows 端集成 (`windows/main.py`)

- **目标:** 在 Windows 主程序中串联 TCP 客户端接收、安全检查和激光器控制执行流程。
- **任务:**
    1. 在 `windows/main.py` 中，初始化 `tcp/client`, `safety/safety_checker`, `laser_control/laser_controller` 的实例。
    2. 加载 `config/windows_config.yaml` 获取配置参数 (如 Jetson IP, TCP 端口, 安全区域等)。
    3. 启动 `tcp/client` 的通信线程，连接到 Jetson。
    4. 在 TCP 客户端的接收处理函数中 (当收到 `TargetCoordinateMsg` 时):
        - 将收到的 `TargetCoordinateMsg` 传递给 `windows/main.py` 的主处理逻辑。
    5. 在 Windows 主处理逻辑中接收到 `TargetCoordinateMsg` 后：
        - 调用 `SafetyChecker.check_safety(target_coord_msg)` 进行安全检查。
        - 如果安全检查通过，调用 `LaserController.fire_laser(target_coord_msg)`，将坐标传递给激光器控制接口执行打击。 (此部分依赖激光器控制代码迁入或封装)
    6. 实现向 Jetson 发送命令 (例如，通过 UI 按钮或脚本触发 `tcp/client.send_command()` 调用)。
    7. 添加必要的错误处理和重连逻辑。
- **验证:**
  - 在 Windows 上运行 `main.py`，在 Jetson 上运行完整的 Jetson 端程序。
  - 确认 Windows 能够稳定接收来自 Jetson 的坐标数据。
  - 测试安全检查逻辑是否按预期工作。
  - 在连接真实激光器前，可以在 `LaserController` 中输出模拟的打击指令，确认坐标传递链条正常。

### 阶段 8: 系统联调、测试与优化

- **目标:** 整个系统端到端联调，发现并解决集成问题，进行全面的功能、性能和精度测试，并进行优化。
- **任务:**
    1. 在静态环境下，测试从相机数据获取到激光模拟打击的完整流程，验证坐标转换精度。
    2. 在小车低速移动环境下，测试引入 EIS 后的系统表现，评估稳像效果对检测和定位精度的影响。
    3. 进行动态环境下的端到端测试，验证实时性是否满足除草需求。
    4. 进行鲁棒性测试，包括不同光照、不同杂草形态、网络波动等情况。
    5. 优化性能瓶颈 (例如，YOLO 推理速度、图像处理速度、通信延迟)。
    6. 完善安全机制和错误报告。
    7. 撰写或更新项目文档和用户手册。

## 4. 关键技术细节与注意事项

- **标定精度:** 外参标定是整个系统精度的基石，务必仔细进行并反复验证。
- **深度数据质量:** NYX 660 在不同环境 (强光、复杂纹理、光滑表面) 下的深度数据质量可能有差异，需要做好深度数据的滤波和有效性检查。
- **时间同步:** 虽然目前没有严格的时间同步要求，但如果未来需要更复杂的行为 (如目标追踪)，可能需要考虑 Jetson 和 Windows 之间的时间同步。可以在消息中加入时间戳。
- **线程管理:** 使用多线程时，注意线程安全和死锁问题。
- **配置管理:** 所有可配置的参数 (IP 地址、端口、标定文件路径、算法阈值等) 都应放在配置文件中，不要硬编码。

## 5. 代码风格与协作

- 遵守统一的 Python 代码风格规范 (如 PEP 8)。
- 编写清晰的函数和类文档字符串 (Docstrings)。
- 使用版本控制系统 (如 Git)。
- 定期进行代码评审。
