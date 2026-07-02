# 任务日志

## 项目

Franka Panda 机械臂抓取放置 (pick-and-place) 强化学习控制

## 目标

在 `nn_controller.py` 中实现 PPO 强化学习控制器，根据 5 阶段任务（接近→下降抓取→上升→运输→放置）设计阶段奖励函数。

---

### 2026-07-02

#### 1. 项目结构分析

**发现问题：**
- `nn_controller.py` 是空文件，需要从头实现
- `train.py` 原本是行为克隆 (BC/IL) 的训练代码，不是 RL
- `controllers/__init__.py` 未导入 NNController
- `test.py` 只支持 IKController

**解决方案：**
- 完整重写 `nn_controller.py`：包含 ActorCritic 网络、PPO 控制器、阶段检测、奖励函数
- 重写 `train.py`：改为 PPO 训练循环
- 修改 `controllers/__init__.py` 和 `test.py`

#### 2. 环境接口探索

**发现问题：**
- 无法直接运行环境测试（无 MuJoCo 或 display 依赖）
- `part.txt` 信息不完整：缺少 `is_grasping` 等重要属性
- 关节限位信息不在代码中，在 XML 模型里

**解决方案：**
- 通过阅读 `franka_env.py` 源码找到 `is_grasping` 属性、`set_arm_target` / `set_gripper` 接口
- 通过 `mjmodel.xml` 查到了关节限位：
  - joint1: `[-2.8973, 2.8973]`
  - joint2: `[-1.7628, 1.7628]`
  - joint4: `[-3.0718, -0.0698]`
  - joint6: `[-0.0175, 3.7525]`
  - 夹爪 (slide): `[0, 0.04]m`，控制值 `[0, 255]`

#### 3. 观测空间设计

**参考 `collect.py` 的观测模板，扩展为 21 维：**
- `arm_q` (4)：关节角度
- `arm_dq` (4)：关节速度
- `ee_pos` (3)：末端执行器位置
- `blk_pos` (3)：方块位置
- `tgt_pos` (3)：目标位置
- `finger_open` (1)：夹爪开度
- `dist_ee_block` (1)：末端到方块距离
- `is_grasping` (1)：是否抓取到方块
- `dist_block_target` (1)：方块到目标距离

#### 4. 动作空间设计

**5 维连续动作：**
- 4 个关节目标位置（tanh → 缩放至关节限位）
- 1 个夹爪命令（tanh → [0, 1]，0=闭合，1=张开）

#### 5. 阶段检测实现

**关键设计：基于距离的有限状态机，非硬编码步数**

- Stage 0 (Approach)：末端 xy 离方块较远 → 移动到方块上方
- Stage 1 (Descend)：末端在方块上方 → 下降抓取
- Stage 2 (Lift)：已抓取 → 上升到搬运高度
- Stage 3 (Transport)：上升到足够高度 → 移向目标
- Stage 4 (Place)：在目标上方 → 下降放置
- Stage 5 (Done)：完成

**遇到的问题：** FSM 需要防止倒退（stage 下降），所以加了 `if self.stage < self._prev_stage: self.stage = self._prev_stage`

#### 6. 奖励函数设计

**分阶段奖励，遵循"路线、偏移、靠近目的地"思路：**

- 进度奖励：`(prev_dist - curr_dist) * 5.0`，鼓励靠近阶段子目标
- 阶段完成奖励：+10（正常阶段）、+50（完成）
- 时间惩罚：-0.01/步，鼓励效率
- 偏离惩罚：距离增大时额外惩罚
- 掉落惩罚：-20（抓取后又丢失方块）
- 成功奖励：+100（任务完成）

#### 7. PPO 算法实现

**标准裁剪 PPO + GAE：**
- GAE(lambda=0.95) 计算优势函数
- clip_epsilon=0.2
- 10 epoch 内更新，mini-batch size=64
- 小网络：Actor(21→64→64→5) + Critic(21→64→64→1)

#### 8. 训练脚本设计

**train.py 主要特征：**
- 支持 `--resume` 从 checkpoint 续训
- 支持 `--rand` 随机方块位置增强泛化
- 每 `save_interval` 步保存 checkpoint + 评估
- 训练日志保存为 JSON

#### 9. 测试脚本修改

**test.py 新增 `--controller nn` 模式：**
- 加载 .pt checkpoint 恢复策略网络
- deterministic 模式运行（无探索噪声）
- 实时显示阶段、距离、抓取状态
