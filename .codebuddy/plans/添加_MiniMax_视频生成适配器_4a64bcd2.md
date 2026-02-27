---
name: 添加 MiniMax 视频生成适配器
overview: 为 videoclaw 项目添加 MiniMax (海螺AI) 视频生成适配器，支持文生视频和图生视频功能。MiniMax 提供免费额度，可替代需要充值的 Kling。
todos:
  - id: create-minimax-adapter
    content: 创建 MiniMaxVideoAdapter 适配器实现
    status: completed
  - id: update-config
    content: 添加 minimax_api_key 配置到 config.py
    status: completed
    dependencies:
      - create-minimax-adapter
  - id: register-entry-point
    content: 在 pyproject.toml 注册 entry point
    status: completed
    dependencies:
      - create-minimax-adapter
  - id: update-env-files
    content: 更新 .env 和 .env.example 配置
    status: completed
    dependencies:
      - update-config
  - id: update-router-profile
    content: 更新 router.py 中的 MODEL_PROFILES
    status: completed
    dependencies:
      - create-minimax-adapter
---

## 用户需求

用户提供了 MiniMax API Key，希望将其集成到 videoclaw 项目中作为视频生成适配器，替代需要充值的 Kling 平台。

## 产品概述

为 videoclaw 项目添加 MiniMax 视频生成适配器，支持文生视频和图生视频功能，利用 MiniMax 的免费额度进行视频生成。

## 核心功能

- 文生视频（Text-to-Video）：根据文本描述生成视频
- 图生视频（Image-to-Video）：根据首帧图片和文本描述生成视频
- 异步任务轮询：创建任务 → 轮询状态 → 下载视频
- 成本估算与健康检查

## 技术栈

- Python 3.12+（现有项目）
- httpx（异步HTTP客户端，已有依赖）
- pydantic-settings（配置管理，已有依赖）

## 实现方案

基于现有 KlingVideoAdapter 的架构模式，实现 MiniMaxVideoAdapter，遵循 VideoModelAdapter 协议。

### MiniMax API 关键信息

- **创建任务**: `POST https://api.minimax.io/v1/video_generation`
- **查询状态**: `GET https://api.minimax.io/v1/query/video_generation?task_id=xxx`
- **下载视频**: `GET https://api.minimax.io/v1/files/retrieve?file_id=xxx`
- **认证方式**: `Authorization: Bearer {api_key}`
- **支持模型**: `MiniMax-Hailuo-2.3`（推荐）、`MiniMax-Hailuo-02`

### 工作流程

1. 创建视频生成任务 → 获取 `task_id`
2. 轮询任务状态（建议间隔10秒）→ 获取 `file_id`
3. 获取视频下载链接 → 下载视频文件

## 目录结构

```
project-root/
├── src/videoclaw/
│   ├── config.py                          # [MODIFY] 添加 minimax_api_key 配置项
│   ├── models/
│   │   ├── adapters/
│   │   │   ├── minimax.py                 # [NEW] MiniMax视频生成适配器
│   │   │   └── __init__.py                # 无需修改
│   │   └── router.py                      # [MODIFY] 更新 MODEL_PROFILES 评分
│   └── ...
├── pyproject.toml                         # [MODIFY] 注册 entry point
├── .env                                   # [MODIFY] 添加 MiniMax API Key
└── .env.example                           # [MODIFY] 添加配置示例
```

## 实现要点

1. **异步任务处理**：MiniMax API 为异步模式，需要轮询任务状态
2. **错误处理**：处理任务失败、超时、网络错误等异常情况
3. **配置管理**：支持 `MINIMAX_API_KEY` 和 `VIDEOCLAW_MINIMAX_API_KEY` 环境变量
4. **Entry Point 注册**：在 pyproject.toml 中注册适配器工厂函数