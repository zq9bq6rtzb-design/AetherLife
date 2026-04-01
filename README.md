# AetherLife
自主进化AI训练系统
AetherLife v1.0 - 最终完美版

作者: Marshall
状态: 开源开发中 | 支持 DPO 对齐 + 进化式架构搜索

核心特性

• 🧮 纯 NumPy 自动微分框架：无第三方深度学习框架依赖，轻量化部署

• 🤖 多架构支持：Transformer、InfiniAttention、MoE、GRU、多模态图像编码

• 📚 训练范式全覆盖：DPO 对齐训练、SFT 预热、离线/在线进化优化

• 🛡️ 自适应硬件层：实时监控 GPU/CPU 状态，自动降频/避峰，保障训练稳定性

• 🧠 生命化系统架构：元学习模块 + 意识核心决策 + 自适应体进化

• ⚖️ 中央仲裁器：多目标优化协调（训练损失、收敛速度、硬件能耗）

• 🔍 多目标进化搜索：自动优化超参数与模型架构

• 💾 断点智能保存：自动检查点（checkpoint）管理，支持断点续训

• 📊 结构化日志：进度条 + 训练曲线 + 指标可视化

• ✅ 全链路自检：单元测试 + 算法有效性校验，保障训练稳定性

快速开始

1. 环境依赖

本项目基于纯 NumPy 实现，核心依赖仅需 numpy，额外依赖用于可视化与工具链：
pip install numpy tqdm matplotlib psutil
# 或通过 requirements.txt 安装
pip install -r requirements.txt
requirements.txt 内容参考：
numpy>=1.24.0
tqdm>=4.65.0
matplotlib>=3.7.0
psutil>=5.9.0

2. 数据准备

• 数据格式：三行一组（prompt | chosen | rejected），文件名为 train.ids

• 存放路径：./bpe_data/train.ids（默认）

• 自定义路径：通过 --data-dir 指定

• 示例格式（token ids 示例）：
101, 202, 303  # prompt token ids
404, 505, 606  # chosen token ids
707, 808, 909  # rejected token ids

3. 运行指令

直接运行（自动模式）
python aether_life_v1.py
• 自动检测数据/检查点

• 先执行单元测试 + 算法自检

• 再启动训练

指定数据目录
python aether_life_v1.py --data-dir ./my_custom_data
恢复训练（断点续训）
python aether_life_v1.py --resume ./checkpoints/checkpoint_001000
目录结构

AetherLife/
├── aether_life_v1.py          # 主训练脚本
├── requirements.txt            # 依赖配置
├── README.md                   # 项目说明（本文件）
├── LICENSE                     # 开源许可证（必选）
├── bpe_data/                   # 数据目录（默认）
│   └── train.ids               # 训练数据
├── checkpoints/                # 检查点目录
│   ├── checkpoint_001000
│   └── ...
├── logs/                       # 日志目录
│   ├── train.log
│   └── metrics.png
└── tests/                      # 单元测试目录
├── test_autograd.py
└── test_architectures.py

配置说明
参数名 类型 默认值 说明 
--data-dir str ./bpe_data 训练数据目录 
--resume str None 断点续训路径（如 ./checkpoints/checkpoint_001000） 
--epochs int 100 训练轮数 
--batch-size int 32 批次大小 
--lr float 1e-4 学习率 
--hardware-monitor bool True 开启硬件监控与自适应降频 

常见问题 (FAQ)

Q1: 运行时提示缺少依赖？

A: 执行 pip install -r requirements.txt 安装所有依赖，确保 Python 版本 ≥ 3.9。

Q2: 训练时显存不足？

A: 1. 减小 --batch-size；2. 开启硬件监控自动降频；3. 拆分大批次为小批次累加梯度。

Q3: 如何查看训练日志？

A: 日志自动保存于 ./logs/train.log，同时终端会输出结构化进度条与关键指标。

开源协议

本项目采用 MIT License 开源协议，详见 LICENSE 文件。

• 允许自由使用、修改、分发

• 保留原作者版权声明即可

• 禁止将本项目作为恶意工具使用

贡献指南

欢迎提交 PR/Issue 参与项目优化：

1. Fork 本仓库

2. 创建功能分支（feature/xxx）

3. 提交代码前执行单元测试

4. 发起 PR，描述修改内容与测试结果

联系与支持

• 问题反馈：提交 Issue

• 技术交流：欢迎参与讨论，共同完善 AetherLife 生态