---
name: 添加 Kling 视频生成适配器
overview: 实现 Kling (可灵) 视频生成 API 适配器，支持 Access Key / Secret Key 认证
todos:
  - id: add-kling-config
    content: 修改 config.py 添加 Kling API 密钥配置字段
    status: completed
  - id: create-kling-adapter
    content: 创建 adapters/kling.py 实现 KlingVideoAdapter
    status: completed
    dependencies:
      - add-kling-config
  - id: register-kling-entry
    content: 修改 pyproject.toml 注册 Kling entry point
    status: completed
    dependencies:
      - create-kling-adapter
  - id: update-env-example
    content: 更新 .env.example 添加 Kling 配置示例
    status: completed
    dependencies:
      - add-kling-config
  - id: update-user-env
    content: 更新 .env 添加用户的 Kling 密钥
    status: completed
    dependencies:
      - add-kling-config
---

## 产品概述

为 VideoClaw 项目添加 Kling（可灵）视频生成模型适配器，使用户能够通过配置 Access Key 和 Secret Key 来调用快手可灵 API 进行视频生成。

## 核心功能

- 配置 Kling API 密钥（Access Key + Secret Key）
- 实现 KlingVideoAdapter 适配器，支持文生视频和图生视频
- 支持异步 API 调用和任务轮询机制
- 与现有模型路由和成本追踪系统集成

## 技术栈选择

- 语言: Python 3.12+
- HTTP 客户端: httpx（复用现有依赖）
- 配置管理: pydantic-settings（复用现有模式）
- 认证方式: Kling API 标准签名认证（HMAC-SHA256）

## 实现方案

### Kling API 认证机制

Kling（快手可灵）API 使用 Access Key + Secret Key 进行签名认证：

- 签名算法: HMAC-SHA256
- 请求头: 包含 timestamp、signature 等字段
- API 流程: 提交任务 -> 轮询状态 -> 下载视频

### 架构设计

复用现有 `VideoModelAdapter` 协议，参考 `OpenAIVideoAdapter` 实现模式：

- 提交生成任务获取 task_id
- 轮询任务状态直到完成
- 下载生成的视频文件

### 目录结构

```
src/videoclaw/
├── config.py                    # [MODIFY] 添加 kling_access_key 和 kling_secret_key 配置字段
└── models/adapters/
    └── kling.py                 # [NEW] KlingVideoAdapter 实现，包含签名认证、API 调用、任务轮询逻辑

pyproject.toml                   # [MODIFY] 添加 kling entry point
.env.example                     # [MODIFY] 添加 Kling 配置示例
.env                             # [MODIFY] 添加用户的 Kling 密钥
```

### 实现要点

1. **config.py**: 添加 `kling_access_key` 和 `kling_secret_key` 配置字段
2. **kling.py**: 

- 实现 HMAC-SHA256 签名认证
- 支持 TEXT_TO_VIDEO 和 IMAGE_TO_VIDEO 能力
- 实现异步任务提交和状态轮询
- 复用 httpx.AsyncClient 进行 HTTP 调用

3. **pyproject.toml**: 在 `videoclaw.adapters` entry points 中注册 Kling 适配器