# TCP/IP 通信模块详细说明

本项目中的 Jetson AGX Orin 和 Windows 上位机之间的通信将通过 TCP/IP Socket 实现，以替代原有的 HTTP 文件传输模式。本模块负责 Jetson 端发送目标坐标等信息，以及 Windows 端接收并可能发送控制命令。

## 1. 通信架构

*   **角色:**
    *   **Jetson AGX Orin:** TCP 服务器端 (`jetson/communication/tcp/server.py`)，监听特定端口，接收 Windows 连接，并发送目标坐标信息。
    *   **Windows 上位机:** TCP 客户端端 (`windows/communication/tcp/client.py`)，主动连接到 Jetson 服务器，接收目标坐标信息，并可能发送控制命令。
*   **连接方式:** 点对点 TCP 连接。Windows 客户端连接到 Jetson 服务器后，建立一个持久连接用于双向数据交换。
*   **网络:** Jetson 和 Windows 连接到同一个交换机，位于同一局域网内。确保 Jetson 的 IP 地址对 Windows 可见且可访问。

## 2. 消息格式 (`common/messages.py`)

为了确保 Jetson 和 Windows 能够正确理解对方发送的数据，我们将使用结构化的消息格式，并采用 JSON 进行序列化。

*   **消息定义:** 在 `common/messages.py` 中定义消息类，例如：

    ```python
    # common/messages.py
    import json
    import time # 导入 time 模块，用于时间戳

    class TargetCoordinateMsg:
        def __init__(self, target_id, laser_x, laser_y, laser_z, confidence, timestamp):
            self.target_id = target_id          # 目标唯一ID (例如，用于追踪或去重)
            self.laser_x = laser_x              # 目标在激光器坐标系下的 X 坐标
            self.laser_y = laser_y              # 目标在激光器坐标系下的 Y 坐标
            self.laser_z = laser_z              # 目标在激光器坐标系下的 Z 坐标
            self.confidence = confidence        # YOLO 检测的置信度
            self.timestamp = timestamp          # 消息生成的时间戳 (例如，time.time() 或 datetime)

        def to_dict(self):
            return {
                "target_id": self.target_id,
                "laser_x": self.laser_x,
                "laser_y": self.laser_y,
                "laser_z": self.laser_z,
                "confidence": self.confidence,
                "timestamp": self.timestamp
            }

        @staticmethod
        def from_dict(data):
            return TargetCoordinateMsg(
                data.get("target_id"),
                data.get("laser_x"),
                data.get("laser_y"),
                data.get("laser_z"),
                data.get("confidence"),
                data.get("timestamp")
            )

    class CommandMsg:
        def __init__(self, command, data=None, command_id=None):
            self.command = command          # 命令类型 (字符串，如 "START_PROCESS", "STOP_PROCESS", "GET_STATUS")
            self.data = data                # 与命令相关的额外数据 (例如，设置参数时的参数值字典)
            self.command_id = command_id    # 可选: 命令的唯一 ID (用于请求-响应模式)

        def to_dict(self):
             return {"command": self.command, "data": self.data, "command_id": self.command_id}

        @staticmethod
        def from_dict(data):
             return CommandMsg(data.get("command"), data.get("data"), data.get("command_id"))

    # --- 序列化与反序列化函数 ---

    def serialize_message(message):
        """将消息对象序列化为 JSON 格式的字节串"""
        if isinstance(message, (TargetCoordinateMsg, CommandMsg)):
            # 将字典转换为 JSON 字符串，然后编码为 UTF-8 字节串
            return json.dumps(message.to_dict()).encode('utf-8')
        # 可以扩展支持其他消息类型
        raise TypeError(f"Unsupported message type: {type(message)}")

    def deserialize_message(data_bytes):
        """将 JSON 格式的字节串反序列化为消息对象"""
        try:
            # 将字节串解码为 UTF-8 字符串，然后解析为 Python 字典
            data_dict = json.loads(data_bytes.decode('utf-8'))
            # 根据字典内容判断消息类型并反序列化
            if "laser_x" in data_dict and "laser_y" in data_dict:
                return TargetCoordinateMsg.from_dict(data_dict)
            elif "command" in data_dict:
                 return CommandMsg.from_dict(data_dict)
            # 可以扩展支持其他消息类型
            print(f"Warning: Received unknown message format: {data_dict}")
            return None # 未知消息类型
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Error deserializing message: {e}")
            return None # 解码或解析失败
    ```

*   **序列化:** 使用 `json.dumps().encode('utf-8')` 将 Python 对象（通过 `to_dict()` 转换）转换为字节串。
*   **反序列化:** 使用 `data_bytes.decode('utf-8')` 将字节串解码为字符串，再使用 `json.loads()` 解析为 Python 字典，最后通过 `from_dict()` 转换为消息对象。

## 3. 消息分帧

TCP 是流协议，没有内置的消息边界。我们需要手动实现分帧机制，以确保接收端能够准确地解析出每一条完整的消息。这里采用 **长度前缀法**。

*   **发送端:**
    1.  将消息序列化为字节串 `message_bytes`。
    2.  获取字节串的长度 `length = len(message_bytes)`。
    3.  将长度 `length` 打包成一个固定长度的二进制表示，例如 4 字节的无符号大端整数 (`>I`)。使用 Python 的 `struct` 库：`length_prefix = struct.pack('>I', length)`。
    4.  将长度前缀和消息体拼接后发送：`socket.sendall(length_prefix + message_bytes)`。
*   **接收端:**
    1.  维护一个接收缓冲区 (`buffer = b''`)。
    2.  持续从 socket 接收数据并追加到缓冲区。
    3.  检查缓冲区：
        *   如果缓冲区长度小于 4 字节，等待更多数据。
        *   如果缓冲区长度大于等于 4 字节，读取前 4 字节 (`length_prefix_bytes = buffer[:4]`)，使用 `struct.unpack('>I', length_prefix_bytes)` 解包得到消息长度 `length`。
        *   检查缓冲区剩余长度：
            *   如果缓冲区剩余长度 (`len(buffer) - 4`) 小于 `length`，说明消息未接收完整，等待更多数据。
            *   如果缓冲区剩余长度大于等于 `length`，说明缓冲区包含至少一条完整消息。读取消息体 (`message_bytes = buffer[4 : 4 + length]`)。
            *   从缓冲区移除已处理的部分 (`buffer = buffer[4 + length:]`)。
            *   对 `message_bytes` 进行反序列化。
            *   重复检查缓冲区，看是否包含下一条消息。

## 4. Jetson TCP 服务器 (`jetson/communication/tcp/server.py`)

*   **职责:**
    *   创建并绑定 Socket，监听指定 IP 和端口。
    *   接受 Windows 客户端的连接。
    *   在一个独立的线程或异步任务中处理客户端连接，负责消息的接收和发送。
    *   提供公共接口供 Jetson 主程序调用，以发送目标坐标消息。
    *   处理连接管理（断开、重连）。
*   **实现要点:**
    *   使用 `socket.socket(socket.AF_INET, socket.SOCK_STREAM)` 创建 TCP Socket。
    *   使用 `socket.bind((ip, port))` 绑定地址。
    *   使用 `socket.listen(1)` 开始监听（只接受一个客户端连接）。
    *   使用 `conn, addr = socket.accept()` 接受连接。
    *   启动一个线程（例如，`ClientHandlerThread`）来处理 `conn` 连接。
    *   在 `ClientHandlerThread` 中实现接收循环：使用 `conn.recv()` 接收数据，进行分帧和反序列化，将收到的 `CommandMsg` 传递给 Jetson 主程序（例如，通过队列或回调函数）。
    *   在 `ClientHandlerThread` 中实现发送方法 `send_message(message)`：接收消息对象，进行序列化和分帧，使用 `conn.sendall()` 发送。
    *   Jetson 主程序通过调用 `server_instance.send_coordinates(target_coord_msg)` 来发送坐标。这个调用会在 `ClientHandlerThread` 的上下文或通过线程间通信（如队列）转发到发送方法。
    *   处理 `socket.error` 等异常，确保连接中断时能有适当的响应或日志记录。

### Jetson TCP 服务器流程图 (Mermaid)

```mermaid
graph TD
    A[Jetson 主程序] --> B{初始化 TCP Server};
    B --> C[创建 Socket];
    C --> D[绑定 IP:Port];
    D --> E[开始监听];
    E --> F{等待客户端连接};
    F --> G[客户端连接成功];
    G --> H[创建 ClientHandler 线程];
    H --> I[ClientHandler 线程循环];
    I --> J{接收数据};
    J -- 长度前缀/消息体 --> K[分帧 & 反序列化];
    K -- CommandMsg --> L[传递命令给主程序];
    I --> M{主程序需要发送坐标?};
    M -- TargetCoordinateMsg --> N[调用 Server 发送方法];
    N --> O[序列化 & 长度前缀打包];
    O --> P[通过 Socket 发送数据];
    J -- 错误/连接关闭 --> Q[处理连接断开];
    H -- 线程结束 --> R[Server 清理];

    %% 交互流
    L -- 执行操作 --> A;
    A -- 计算出坐标 --> M;
    P -- 通过网络 --> Windows 客户端;
    Windows_Client --> J;
```

## 5. Windows TCP 客户端 (`windows/communication/tcp/client.py`)

*   **职责:**
    *   创建 Socket，连接到 Jetson 服务器。
    *   在一个独立的线程或异步任务中处理接收循环，负责消息的接收和反序列化。
    *   提供公共接口以发送控制命令。
    *   处理连接管理（断开、自动重连）。
*   **实现要点:**
    *   使用 `socket.socket(socket.AF_INET, socket.SOCK_STREAM)` 创建 TCP Socket。
    *   使用 `socket.connect((jetson_ip, jetson_port))` 连接服务器。通常在一个循环中尝试连接直到成功，实现自动重连。
    *   启动一个线程（例如，`ServerListenerThread`）来处理接收逻辑。
    *   在 `ServerListenerThread` 中实现接收循环：使用 `socket.recv()` 接收数据，进行分帧和反序列化，将收到的 `TargetCoordinateMsg` 传递给 Windows 主程序（例如，通过队列或回调函数）。
    *   提供公共方法 `send_command(command_msg)`：接收 `CommandMsg` 对象，进行序列化和分帧，使用 `socket.sendall()` 发送给服务器。
    *   处理 `socket.error` 等异常，实现断线后的重连逻辑。

### Windows TCP 客户端流程图 (Mermaid)

```mermaid
graph TD
    A[Windows 主程序] --> B{初始化 TCP Client};
    B --> C[创建 Socket];
    C --> D{尝试连接 Jetson};
    D -- 连接失败 --> D; % 自动重连
    D -- 连接成功 --> E[连接建立];
    E --> F[创建 ServerListener 线程];
    F --> G[ServerListener 线程循环];
    G --> H{接收数据};
    H -- 长度前缀/消息体 --> I[分帧 & 反序列化];
    I -- TargetCoordinateMsg --> J[传递坐标给主程序];
    A -- 需要发送命令 --> K[调用 Client 发送方法];
    K -- CommandMsg --> L[序列化 & 长度前缀打包];
    L --> M[通过 Socket 发送数据];
    H -- 错误/连接关闭 --> N[处理连接断开 & 重连];
    F -- 线程结束 --> O[Client 清理];

    %% 交互流
    J -- 处理坐标 --> A;
    A -- 根据逻辑/用户输入 --> K;
    M -- 通过网络 --> Jetson 服务器;
    Jetson_Server --> H;
```

## 6. 集成到主程序

*   **Jetson (`jetson/main.py`):**
    *   创建 `NYXCamera`, `ImageStabilizer`, `CoordinateTransformer` 实例。
    *   创建 `TcpServer` 实例，并启动其处理连接的线程。
    *   主循环中获取相机帧 -> 稳像 -> 检测 -> 坐标转换。
    *   计算出坐标后，调用 `TcpServer` 实例的发送方法发送消息。
    *   处理 `TcpServer` 接收线程传过来的命令消息。
*   **Windows (`windows/main.py`):**
    *   创建 `TcpClient`, `SafetyChecker`, `LaserController` 实例。
    *   创建 `TcpClient` 实例，并启动其处理接收和重连的线程。
    *   `TcpClient` 接收线程收到坐标消息后，通过回调或其他机制通知 Windows 主程序。
    *   Windows 主程序接收到坐标后，调用 `SafetyChecker`，然后调用 `LaserController`。
    *   通过 UI 或其他方式调用 `TcpClient` 实例的发送方法发送命令给 Jetson。

## 7. 关键实现库

*   `socket`: Python 内置库，用于创建和管理 TCP Socket。
*   `struct`: Python 内置库，用于在 Python 对象和 C 结构体之间进行转换，实现长度前缀打包和解包。
*   `json`: Python 内置库，用于 JSON 序列化和反序列化。
*   `threading` 或 `asyncio`: 用于在主程序不被通信阻塞的情况下进行并发处理。

## 8. 开发建议

*   **逐步实现:** 先实现最简单的连接和收发固定字符串，再加入消息格式和分帧。
*   **错误处理:** 在每一步都要考虑可能的网络错误、连接断开、数据格式错误，并实现健壮的错误处理和日志记录。
*   **单元测试:** 对消息序列化/反序列化、分帧逻辑进行单元测试。
*   **独立测试:** 先独立运行 Jetson Server 和 Windows Client，使用测试消息进行充分测试，确保通信层稳定可靠，再集成到各自的主程序中。
