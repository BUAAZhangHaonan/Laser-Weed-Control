import sys
import os
import time
import numpy as np
from ctypes import c_uint16, c_uint32, c_bool

current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from LaserWeedControl.camera.API.ScepterDS_api import ScepterTofCam
    from LaserWeedControl.camera.API.ScepterDS_define import *
    from LaserWeedControl.camera.API.ScepterDS_enums import ScFrameType, ScSensorType
except ImportError:
    print("Failed to import ScepterDS. Ensure the SDK's Python directory is in sys.path or PYTHONPATH.")
    sys.exit(1)


class NYXCamera:
    def __init__(self):
        # 初始化NYX相机对象，设置初始状态和参数
        self.camera = ScepterTofCam()
        self.device_info = None
        self.is_connected = False
        self.is_streaming = False

        self.rgb_intrinsic_params = None
        self.depth_intrinsic_params = None
        self.rgb_to_depth_extrinsic_params = None

    def connect(self) -> bool:
        """
        连接到NYX相机设备。如果已连接则直接返回。
        扫描设备、获取设备信息并尝试连接。
        Returns:
            bool: 连接是否成功
        """
        if self.is_connected:
            print("Camera already connected.")
            return True

        print("Scanning for devices...")
        ret = self.camera.scGetDeviceCount(c_uint32(3000))
        camera_count = ret[1] if isinstance(ret, tuple) else 0
        if (ret[0] if isinstance(ret, tuple) else ret) != 0 or camera_count == 0:
            print(
                f"Failed to get device count or no devices found. Ret: {ret}, Count: {camera_count}")
            return False

        print(f"Found {camera_count} device(s).")
        ret, device_info_list = self.camera.scGetDeviceInfoList(camera_count)
        if ret != 0:
            print(f"Failed to get device info list. Ret: {ret}")
            return False

        self.device_info = device_info_list[0]
        print(
            f"Attempting to connect to device SN: {self.device_info.serialNumber}")

        ret = self.camera.scOpenDeviceBySN(self.device_info.serialNumber)
        if ret != 0:
            print(f"Failed to open device. Ret: {ret}")
            return False

        self.is_connected = True
        print("Camera connected successfully.")
        return True

    def disconnect(self) -> bool:
        """
        断开与NYX相机的连接。如果正在流式传输，则先停止流。
        Returns:
            bool: 断开是否成功
        """
        if not self.is_connected:
            print("Camera not connected.")
            return True

        if self.is_streaming:
            self.stop_streaming()

        ret = self.camera.scCloseDevice()
        if ret != 0:
            print(f"Failed to close device. Ret: {ret}")
            self.is_connected = False
            return False

        self.is_connected = False
        print("Camera disconnected successfully.")
        return True

    def start_streaming(self) -> bool:
        """
        启动相机的数据流。需先连接设备。
        启用深度到彩色对齐，并开始流式传输。
        Returns:
            bool: 启动流是否成功
        """
        if not self.is_connected:
            print("Camera not connected. Cannot start streaming.")
            return False
        if self.is_streaming:
            print("Stream already started.")
            return True

        ret = self.camera.scSetTransformDepthImgToColorSensorEnabled(c_bool(True))
        if ret != 0:
            print(f"Failed to enable depth to color transform. Ret: {ret}")
        else:
            print("Depth to color transform enabled.")

        ret = self.camera.scStartStream()
        if ret != 0:
            print(f"Failed to start stream. Ret: {ret}")
            return False

        self.is_streaming = True
        print("Stream started successfully.")
        return True

    def stop_streaming(self) -> bool:
        """
        停止相机的数据流。
        Returns:
            bool: 停止流是否成功
        """
        if not self.is_streaming:
            print("Stream not started.")
            return True

        ret = self.camera.scStopStream()
        if ret != 0:
            print(f"Failed to stop stream. Ret: {ret}")
            return False

        self.is_streaming = False
        print("Stream stopped successfully.")
        return True

    def get_frame(self, timeout_ms: int = 1200) -> tuple:
        """
        获取一帧RGB图像和对齐的深度图像。
        Args:
            timeout_ms (int): 等待帧准备好的超时时间（毫秒）。
        Returns:
            tuple: (rgb_image, aligned_depth_image)
        """
        if not self.is_streaming:
            print("Stream not started. Cannot get frame.")
            return None, None

        ret, frameready = self.camera.scGetFrameReady(c_uint16(timeout_ms))
        if ret != 0:
            print(f"Failed to get frame ready status. Ret: {ret}")
            return None, None

        rgb_image = None
        aligned_depth_image = None

        # Get Color Frame
        if frameready.color:
            ret, color_frame = self.camera.scGetFrame(ScFrameType.SC_COLOR_FRAME)
            if ret == 0:
                if color_frame.pFrameData:
                    rgb_image = np.ctypeslib.as_array(
                        color_frame.pFrameData,
                        (color_frame.height, color_frame.width, 3)
                    ).copy()
            else:
                print(f"Failed to get color frame. Ret: {ret}")

        # Get Transformed Depth Frame (Aligned to Color Sensor)
        if frameready.transformedDepth:
            ret, depth_frame = self.camera.scGetFrame(ScFrameType.SC_TRANSFORM_DEPTH_IMG_TO_COLOR_SENSOR_FRAME)
            if ret == 0:
                if depth_frame.pFrameData:
                    aligned_depth_image = np.ctypeslib.as_array(
                        depth_frame.pFrameData,
                        (depth_frame.height, depth_frame.width)
                    ).astype(np.uint16).copy()
            else:
                print(f"Failed to get transformed depth frame. Ret: {ret}")
        elif frameready.depth:
            print("Transformed depth not ready, attempting to get raw depth.")
            ret, depth_frame = self.camera.scGetFrame(ScFrameType.SC_DEPTH_FRAME)
            if ret == 0:
                if depth_frame.pFrameData:
                    print("Warning: Got raw depth, not aligned depth.")
            else:
                print(f"Failed to get raw depth frame. Ret: {ret}")

        return rgb_image, aligned_depth_image

    def get_calibration_params(self) -> dict | None:
        """
        获取相机的标定参数，包括RGB和深度相机的内参，以及深度到RGB的外参。
        Returns:
            dict | None: 包含内外参的字典
        """
        if not self.is_connected:
            print("Camera not connected. Cannot get calibration parameters.")
            return None

        # Get RGB Sensor Intrinsic Parameters
        ret_rgb_intr, self.rgb_intrinsic_params = self.camera.scGetSensorIntrinsicParameters(ScSensorType.SC_COLOR_SENSOR)
        if ret_rgb_intr != 0:
            print(
                f"Failed to get RGB sensor intrinsic parameters. Ret: {ret_rgb_intr}")
            self.rgb_intrinsic_params = None
        else:
            print("RGB Sensor Intrinsics:", self.rgb_intrinsic_params.fx,
                  self.rgb_intrinsic_params.cx, "...")  # etc.

        # Get Depth (TOF) Sensor Intrinsic Parameters
        ret_depth_intr, self.depth_intrinsic_params = self.camera.scGetSensorIntrinsicParameters(
            ScSensorType.SC_TOF_SENSOR)
        if ret_depth_intr != 0:
            print(
                f"Failed to get Depth sensor intrinsic parameters. Ret: {ret_depth_intr}")
            self.depth_intrinsic_params = None
        else:
            print("Depth Sensor Intrinsics:", self.depth_intrinsic_params.fx,
                  self.depth_intrinsic_params.cx, "...")  # etc.

        # Get Extrinsic Parameters (e.g., Depth to RGB sensor)
        ret_extr, self.rgb_to_depth_extrinsic_params = self.camera.scGetSensorExtrinsicParameters()
        if ret_extr != 0:
            print(f"Failed to get extrinsic parameters. Ret: {ret_extr}")
            self.rgb_to_depth_extrinsic_params = None
        else:
            print("Extrinsics (Depth to Color): Rotation ", self.rgb_to_depth_extrinsic_params.rotation,
                  "Translation", self.rgb_to_depth_extrinsic_params.translation)

        return {
            "rgb_intrinsics": self.rgb_intrinsic_params,
            "depth_intrinsics": self.depth_intrinsic_params,
            "extrinsics_depth_to_color": self.rgb_to_depth_extrinsic_params
        }

    def __enter__(self) -> "NYXCamera":
        """
        支持with语句自动连接相机。
        Returns:
            NYXCamera: 当前对象
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        支持with语句自动断开相机连接。
        """
        self.disconnect()


if __name__ == '__main__':
    nyx_cam = NYXCamera()
    if nyx_cam.connect():
        calib_params = nyx_cam.get_calibration_params()
        if calib_params:
            if calib_params["rgb_intrinsics"]:
                print(
                    f"RGB fx: {calib_params['rgb_intrinsics'].fx}, fy: {calib_params['rgb_intrinsics'].fy}")

        if nyx_cam.start_streaming():
            try:
                for i in range(10):
                    print(f"Attempting to get frame {i+1}")
                    rgb, depth = nyx_cam.get_frame()
                    if rgb is not None:
                        print(
                            f"RGB frame received: shape {rgb.shape}, dtype {rgb.dtype}")
                    if depth is not None:
                        print(
                            f"Aligned Depth frame received: shape {depth.shape}, dtype {depth.dtype}")

                    time.sleep(0.1)

            finally:
                nyx_cam.stop_streaming()

        nyx_cam.disconnect()
    else:
        print("Could not connect to camera.")
