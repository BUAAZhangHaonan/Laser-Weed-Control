from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TargetCoordinateMsg:
    """杂草目标坐标消息"""
    timestamp: float  # 时间戳
    target_positions: List[Tuple[float, float, float]]  # 3D坐标列表 [(x,y,z), ...]
    confidence_scores: List[float]  # 每个目标的置信度


@dataclass
class CameraStatusMsg:
    """相机状态消息"""
    is_connected: bool
    frame_rate: float
    error_code: int = 0
    error_message: str = ""


@dataclass
class LaserStatusMsg:
    """激光器状态消息"""
    is_connected: bool
    is_ready: bool
    error_code: int = 0
    error_message: str = ""
