---
name: 添加智谱清影 (CogVideoX) 视频生成适配器
overview: 为 videoclaw 项目添加智谱清影 (CogVideoX) 视频生成适配器，使用 zhipuai SDK 实现文生视频和图生视频功能。用户已提供有效的 API Key。
todos:
  - id: add-zhipuai-dependency
    content: 添加 zhipuai SDK 依赖到 pyproject.toml
    status: completed
  - id: create-zhipu-adapter
    content: 创建 ZhipuVideoAdapter 适配器实现
    status: completed
    dependencies:
      - add-zhipuai-dependency
  - id: update-config
    content: 添加 zhipu_api_key 配置到 config.py
    status: completed
    dependencies:
      - create-zhipu-adapter
  - id: register-entry-point
    content: 在 pyproject.toml 注册 entry point
    status: completed
    dependencies:
      - create-zhipu-adapter
  - id: update-env-files
    content: 更新 .env 和 .env.example 配置
    status: completed
    dependencies:
      - update-config
  - id: update-router-profile
    content: 更新 router.py 中的 MODEL_PROFILES
    status: completed
    dependencies:
      - create-zhipu-adapter
  - id: test-adapter
    content: 测试智谱清影适配器
    status: completed
    dependencies:
      - update-env-files
      - register-entry-point
      - update-router-profile
---

## 用户需求

用户提供了智谱清影 API Key，希望将其集成到 videoclaw 项目中作为视频生成适配器，替代需要充值的 Kling 和无效的 MiniMax API Key。

## 产品概述

为 videoclaw 项目添加智谱清影 (CogVideoX) 视频生成适配器，支持文生视频和图生视频功能，利用智谱清影的免费额度进行视频生成。

## 核心功能

- 文生视频（Text-to-Video）：根据文本描述生成视频
- 图生视频（Image-to-Video）：根据图片和文本描述生成视频
- 异步任务轮询：创建任务 → 轮询状态 → 下载视频
- 成本估算与健康检查

## 用户提供的 API Key

`36477ee2ab364f228c543c900c9ff358.qhWzYGGNsgYCyF8R`

## 技术栈

- Python 3.12+（现有项目）
- zhipuai SDK（智谱官方 SDK）
- httpx（异步HTTP客户端，用于下载视频）

## 实现方案

### 智谱清影 API 关键信息

- **SDK**: `pip install zhipuai`
- **模型**: `cogvideox-flash`（快速版）、`cogvideox`（高品质版）
- **认证方式**: API Key 直接传入 ZhipuAI 客户端

### 工作流程

1. 初始化 ZhipuAI 客户端
2. 提交视频生成任务 → 获取 `task_id`
3. 轮询任务状态（建议间隔5-10秒）→ 获取 `video_result`
4. 下载视频文件

### API 调用示例

```python
from zhipuai import ZhipuAI
client = ZhipuAI(api_key="YOUR_API_KEY")

# 提交任务
response = client.videos.generations(
    model="cogvideox-flash",
    prompt="一只可爱的柯基犬在阳光下的草地上奔跑",
    size="1024x576",
    duration=5,
    quality="speed",
)

# 轮询状态
result = client.videos.retrieve_videos_result(id=response.id)
# result.task_status: "SUCCESS", "FAILED", "PROCESSING"
# result.video_result: 视频URL列表
```

## 目录结构

```
project-root/
├── src/videoclaw/
│   ├── config.py                          # [MODIFY] 添加 zhipu_api_key 配置项
│   ├── models/
│   │   ├── adapters/
│   │   │   ├── zhipu.py                   # [NEW] 智谱清影视频生成适配器
│   │   │   └── __init__.py                # 无需修改
│   │   └── router.py                      # [MODIFY] 更新 MODEL_PROFILES 评分
│   └── ...
├── pyproject.toml                         # [MODIFY] 注册 entry point，添加 zhipuai 依赖
├── .env                                   # [MODIFY] 添加智谱 API Key
└── .env.example                           # [MODIFY] 添加配置示例
```

## 实现要点

1. **使用官方 SDK**: 智谱提供 `zhipuai` SDK，比直接 HTTP 调用更稳定
2. **异步任务处理**: 视频生成为异步模式，需要轮询任务状态
3. **错误处理**: 处理任务失败、超时、网络错误等异常情况
4. **配置管理**: 支持 `ZHIPU_API_KEY` 和 `VIDEOCLAW_ZHIPU_API_KEY` 环境变量