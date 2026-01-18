# 🚗 Limo 机器人 TD3-STL 算法实物部署操作指南

本文档详细说明了如何将训练好的 **TD3-STL 强化学习导航模型** 部署到 **松灵 Limo 实车** 上。

⚠️ **核心提示**：实车部署不仅仅是运行代码，最关键的是 **“坐标系对齐”** 和 **“传感器一致性”**。请务必严格按照以下步骤操作。

---

## 📋 第一阶段：环境与代码核对 (Checklist)

在开始搬运小车之前，请确保您的代码库已经完成了 **Sim-to-Real 的对齐修正**。

### 1.1 仿真模型对齐

* **文件**: `src/limo_description/urdf/limo_gazebo.gazebo`
* **检查项**:
* [ ] `samples` 是否已改为 **720**？
* [ ] `max angle` 是否为 **1.57** (90度，总FOV 180度)？
* [ ] `range max` 是否为 **6.0**？


* **操作**: 修改后必须重新编译工作空间：
```bash
catkin_make
source devel/setup.bash

```



### 1.2 训练参数对齐

* **文件**: `src/limoRL/scripts/STL-TD3/params.py`
* **检查项**:
* [ ] `MAX_V` 是否已降至 **0.5** (实车安全速度)？
* [ ] `ACTION_REPEAT` 是否设为 **2** (对应 5Hz 控制频率)？



### 1.3 部署脚本确认

* **文件**: `deploy_limo.py`
* **检查项**: 是否移除了 `env.pose_odom += offset` 这种错误的累加代码？（应由 Env 类内部处理或只在读取时处理）。

---

## 📍 第二阶段：物理场地坐标标定 (Calibration)

**这是最关键的一步。** 仿真里的任务点（例如 `(6, -6)`）在现实世界中可能在墙壁里。我们必须测量现实场景的坐标，并反算回代码中。

### 2.1 确定“物理原点”

1. 在办公室/实验室地面贴一个胶带十字，作为 **实车启动点 (Real Origin, 0,0)**。
2. 规定 **车头朝向**（例如正对走廊前方）。此方向即为 **X轴正方向**。
3. **注意**：每次实验，小车必须严格摆放在此位置、此朝向开机。

### 2.2 采集现实任务坐标

1. **启动 Limo 底盘** (在 Limo 端):
```bash
roslaunch limo_bringup limo_start.launch pub_odom_tf:=false

```


2. **启动键盘控制** (在 PC 端):
```bash
roslaunch limo_bringup limo_teletop_keyboard.launch

```


3. **测量任务点 A (例如门口)**:
* 遥控小车开到门口中心。
* 查看里程计坐标:
```bash
rostopic echo /odom/pose/pose/position -n 1

```


* 记下输出的 `x` 和 `y` (例如 `x=3.5, y=-1.2`)。


4. **测量任务点 B (例如打印机旁)**:
* 遥控开过去，记下坐标 (例如 `x=6.0, y=2.0`)。



### 2.3 反算并更新配置

我们需要将“现实坐标”转换为模型认为的“仿真坐标”。

* **公式**: `Sim_Coord = Real_Coord + Offset`
* **默认 Offset**: `(-7.0, 0.0)` (因为仿真里车出生在 -7.0)

**计算示例**:

* **任务 A (门口)**:
* Real: `(3.5, -1.2)`
* Sim: `3.5 + (-7.0) = -3.5`, `-1.2 + 0 = -1.2`
* **填入 `params.py**`: `pos: [-3.5, -1.2]`


* **任务 B (打印机)**:
* Real: `(6.0, 2.0)`
* Sim: `6.0 + (-7.0) = -1.0`, `2.0 + 0 = 2.0`
* **填入 `params.py**`: `pos: [-1.0, 2.0]`



**操作**: 修改 `src/limoRL/scripts/STL-TD3/params.py` 中的 `TASK_CONFIG`，填入计算后的坐标。

---

## 🚀 第三阶段：实车实验操作流程

### 3.1 网络配置

确保 PC 和 Limo 连接同一 WiFi。

* **Limo 端 (`~/.bashrc`)**:
```bash
export ROS_MASTER_URI=http://<PC_IP>:11311
export ROS_IP=<LIMO_IP>

```


* **PC 端 (`~/.bashrc`)**:
```bash
export ROS_MASTER_URI=http://<PC_IP>:11311
export ROS_IP=<PC_IP>

```


* **测试**: 在两端互相 `ping` 对方 IP。

### 3.2 启动步骤

1. **PC 端**: 启动 ROS Master
```bash
roscore

```


2. **Limo 端**: 启动底层驱动
* 将车摆放在 **物理原点**。
* SSH 连接 Limo 并运行:


```bash
roslaunch limo_bringup limo_start.launch pub_odom_tf:=false

```


* *检查雷达是否旋转。*


3. **PC 端**: 检查数据链路
```bash
rostopic list  # 应该能看到 /scan 和 /odom
rostopic echo /scan -n 1 # 确认有数据且不为空

```


4. **PC 端**: 运行部署脚本
```bash
cd ~/STL-Projects/limo_RL/limo_RL-main/src/limoRL/scripts/STL-TD3/
python3 deploy_limo.py

```


* 终端会显示: `Alignment Offset: [-7.0, 0.0]`
* **再次确认**车头方向正确。
* 按 **Enter** 键开始。



### 3.3 实验监控

* **观察终端**:
* `Task`: 当前进行到第几个任务。
* `Dist`: 距离当前目标的距离（如果距离在变小，说明导航正常）。
* `Act`: 输出的线速度和角速度。


* **紧急停止**:
* 如果小车即将撞墙或失控，请立即在运行 Python 的终端按 **`Ctrl + C`**。脚本逻辑会发送 `(0,0)` 速度指令强制停车。



---

## ❓ 常见问题 (Troubleshooting)

1. **小车原地转圈 / 倒车**
* **原因**: 坐标系定义反了，或者电机驱动方向定义反了。
* **解决**: 在 `stl_real_env.py` 的 `step` 函数中，尝试给 `action[1]` (角速度) 加负号测试。


2. **小车直冲墙壁**
* **原因**: 这里的墙壁在“仿真坐标系”里可能是一片空地。
* **解决**: 说明你的 **第 2.3 步 (坐标反算)** 没做对，或者物理摆放的朝向歪了。请重新标定。


3. **雷达数据报错 / 无法避障**
* **原因**: 实车雷达可能扫到了车体上的天线或外壳，导致它以为只有 1cm 距离。
* **解决**: 打开 PC 端的 `rviz`，添加 `LaserScan`，查看是否有固定的噪点在车身周围。如果有，需要在代码中把这部分角度的数据 mask 掉 (设为 5.0)。


4. **报错 "Model not found"**
* **原因**: `models` 文件夹里没有你指定的模型。
* **解决**: 检查 `deploy_limo.py` 里的 `model_name` 是否与文件夹里的文件名一致（注意不要带 `_actor` 后缀）。