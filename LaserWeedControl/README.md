# LaserWeedControl 包使用说明

推荐一：用包的方式运行脚本，在项目根目录（workspace）下运行：

```bash
python -m LaserWeedControl.camera.coordinate_transform_2d_simple
```

这样 Python 会自动把根目录加入 sys.path，不用每个脚本都加路径代码。

推荐二：写一个统一的入口脚本，在项目根目录下写一个 run.py，里面统一调用你要运行的功能。

推荐三：在每个脚本中手动添加路径代码，确保可以正确导入包内模块。
这段代码适用于LaserWeedControl包内的所有脚本，前提是这些脚本都位于LaserWeedControl的子目录（如camera、camera/API等）下。

```python
import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
```

它的作用是：将LaserWeedControl的上一级目录（即你的项目根目录，如workspace）加入sys.path，从而保证from LaserWeedControl.xxx的导入在任何运行位置都不会出错。
适用范围：

- 适用于LaserWeedControl包内的所有脚本（如LaserWeedControl/camera/xxx.py）。
- 如果脚本在更深层目录（如LaserWeedControl/camera/API/xxx.py），可以适当增加'..'的数量。
- 如果脚本在项目根目录（如workspace），则不适用，应调整为```os.path.dirname(__file__)```。

建议：
对于包内脚本，推荐用这种方式。
对于不同目录层级的脚本，建议根据实际目录结构调整'..'的数量。
