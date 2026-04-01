# AetherLife
自主进化AI训练系统
AetherLife v1.0 - 最终完美版


# AetherLife

**自主进化 AI 训练系统**  
**AetherLife v1.0 - 最终完美版（单文件）**

**作者**: Marshall  
**许可证**: MIT License (2026)

一个**纯 NumPy 实现**、**可在手机/iPad 上直接运行**的轻量级自主进化 AI 训练框架。

它不依赖 PyTorch 或 TensorFlow，却支持 DPO 对齐、InfiniAttention、MoE 等先进架构，并内置了**生命化自适应系统** —— 模型会像生命体一样感知硬件、自我诊断、动态进化。

### 核心亮点

- 🧮 **纯 NumPy 自动微分**：从零实现的 Tensor 系统，轻量到极致
- 🤖 **多架构支持**：Transformer、InfiniAttention、MoE、GRU、多模态图像编码
- 📚 **训练全覆盖**：DPO 对齐 + SFT 预热 + 在线/离线进化优化
- 🛡️ **自适应硬件层**：实时监控内存、电量、温度，自动降频避峰
- 🧠 **生命化系统**：元学习 + 意识核心决策 + 自适应体动态重构
- ⚖️ **中央仲裁器**：多目标协调（损失、速度、能耗）
- 🔍 **多目标进化搜索**：自动优化超参数与模型架构
- 💾 **智能断点续训** + 结构化日志 + 训练可视化

**特别适合**：移动端开发者、轻量级 AI 实验、从零实现爱好者、Pythonista 用户。

### 快速开始

#### 1. 安装依赖
```bash
pip install numpy tqdm matplotlib psutil
2. 数据准备

将训练数据放在 ./bpe_data/train.ids，格式为三行一组（prompt | chosen | rejected），每行是 token id 用空格分隔。

3. 运行训练

# 自动模式（推荐）
python AetherLife_LifeCore.py

# 指定数据目录
python AetherLife_LifeCore.py --data-dir ./my_data

# 断点续训
python AetherLife_LifeCore.py --resume ./checkpoints/checkpoint_001000
项目结构

AetherLife/
├── AetherLife_LifeCore.py     # 主训练脚本（单文件）
├── README.md
├── LICENSE
├── requirements.txt
├── bpe_data/train.ids         # 训练数据
├── checkpoints/               # 自动保存的检查点
└── logs/                      # 训练日志
更多配置参数请运行 python AetherLife_LifeCore.py --help 查看。

常见问题

Q：能在 iPhone/iPad 上跑吗？
A：完全可以！本项目专门适配 Pythonista 等移动端环境，资源占用低。

Q：训练速度慢怎么办？
A：降低 batch size，或开启硬件监控自动降频。SFT 预热阶段可使用较短序列加速启动。

Q：如何查看训练曲线？
A：训练完成后 checkpoints 目录会生成报告，logs 目录保存详细日志。

开源协议

本项目采用 MIT License，欢迎自由使用、修改和分发。

贡献指南

欢迎提交 Issue 和 Pull Request，一起把这个自主进化 AI 项目做得更好！