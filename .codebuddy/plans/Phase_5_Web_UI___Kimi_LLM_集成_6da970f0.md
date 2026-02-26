---
name: Phase 5 Web UI + Kimi LLM 集成
overview: 实现拖拽式 DAG 可视化编辑器（React Flow）和 Kimi LLM 配置支持
design:
  architecture:
    framework: react
    component: shadcn
  styleKeywords:
    - Modern
    - Tech
    - Dark-mode
  fontSystem:
    fontFamily: Inter
    heading:
      size: 18px
      weight: 600
    subheading:
      size: 14px
      weight: 500
    body:
      size: 14px
      weight: 400
  colorSystem:
    primary:
      - "#3b82f6"
      - "#8b5cf6"
      - "#22c55e"
    background:
      - "#0a0a0a"
      - "#171717"
      - "#262626"
    text:
      - "#fafafa"
      - "#a3a3a3"
    functional:
      - "#22c55e"
      - "#ef4444"
      - "#eab308"
todos:
  - id: config-kimi
    content: 添加 Kimi API 配置到 .env.example 和 config.py
    status: completed
  - id: install-deps
    content: 安装 React Flow 前端依赖 (@xyflow/react)
    status: completed
  - id: flow-types
    content: 创建 Flow 编辑器类型定义 (types.ts)
    status: completed
    dependencies:
      - install-deps
  - id: flow-utils
    content: 实现 YAML
    status: completed
    dependencies:
      - flow-types
---

## 产品概述

为 VideoClaw 项目完成 Phase 5 Web UI 可视化编辑器，并配置支持 Kimi (月之暗面) LLM。

## 核心功能

### 1. DAG 可视化编辑器

- 拖拽式节点创建与编辑
- 可视化连线管理依赖关系
- 属性面板配置节点参数
- YAML 与可视化双向同步
- 实时预览与验证

### 2. Kimi LLM 集成

- 支持 Kimi API Key 配置
- Director Agent 可使用 Kimi 模型生成视频计划
- 支持模型切换 (moonshot-v1-8k/32k/128k)

## 技术栈

- **前端**: Next.js 16, React 19, TypeScript, Tailwind CSS, shadcn/ui
- **DAG 编辑器**: React Flow (@xyflow/react v12)
- **后端**: Python 3.12+, LiteLLM (已集成)
- **LLM**: Kimi (OpenAI 兼容 API)

## 技术架构

### Kimi LLM 集成

Kimi 使用 OpenAI 兼容 API，LiteLLM 直接支持：

- 模型字符串: `openai/moonshot-v1-8k`
- API Base: `https://api.moonshot.cn/v1`
- 认证: `MOONSHOT_API_KEY`

### DAG 编辑器架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Flow Editor Page                         │
├──────────────┬──────────────────────────┬──────────────────┤
│  NodePalette │      FlowCanvas          │  NodeProperties  │
│  (左侧工具栏)  │      (React Flow 画布)    │  (右侧属性面板)   │
│              │                          │                  │
│  - script_gen│   ┌─────┐    ┌─────┐    │  - 节点 ID       │
│  - storyboard│   │node1│───▶│node2│    │  - 参数配置      │
│  - video_gen │   └─────┘    └─────┘    │  - 依赖管理      │
│  - tts       │                          │                  │
│  - music     │   支持拖拽、缩放、平移      │                  │
│  - compose   │                          │                  │
│  - render    │                          │                  │
├──────────────┴──────────────────────────┴──────────────────┤
│                    YAML Editor (Tab 切换)                   │
│                    双向同步: Flow JSON <-> YAML              │
└─────────────────────────────────────────────────────────────┘
```

## 核心目录结构

```
videoclaw/
├── .env.example                    # [MODIFY] 添加 MOONSHOT_API_KEY
├── src/videoclaw/
│   └── config.py                   # [MODIFY] 添加 moonshot_api_key
└── web/
    ├── package.json                # [MODIFY] 添加 @xyflow/react
    └── src/
        ├── app/flow/page.tsx       # [MODIFY] 集成可视化编辑器
        └── components/flow-editor/ # [NEW] DAG 编辑器组件
            ├── FlowCanvas.tsx      # React Flow 画布
            ├── NodePalette.tsx     # 左侧节点工具栏
            ├── NodeProperties.tsx  # 右侧属性面板
            ├── CustomNode.tsx      # 自定义节点组件
            ├── types.ts            # 类型定义
            ├── utils.ts            # YAML <-> Flow 转换
            └── index.ts            # 导出
```

## 实现要点

- React Flow v12 使用 `@xyflow/react` 包名
- 节点类型与后端 `TaskType` 枚举对齐
- 连线验证：禁止循环依赖、确保依赖节点存在
- YAML 同步：使用 `yaml` 库解析/生成

## 设计风格

采用现代科技感设计风格，与现有 VideoClaw Web UI 保持一致。

## 布局设计

- **三栏布局**: 左侧工具栏 (200px) + 中间画布 (自适应) + 右侧属性面板 (280px)
- **顶部 Tab**: YAML Editor / Visual Editor 切换
- **底部状态栏**: 验证状态、节点数量、同步状态

## 节点配色

按任务类型区分节点颜色：

- `script_gen`: 蓝色 (#3b82f6)
- `storyboard`: 紫色 (#8b5cf6)
- `video_gen`: 绿色 (#22c55e)
- `tts`: 橙色 (#f97316)
- `music`: 粉色 (#ec4899)
- `compose`: 黄色 (#eab308)
- `render`: 青色 (#06b6d4)

## 交互设计

- 从左侧工具栏拖拽节点到画布创建
- 点击节点显示属性面板
- 拖拽连接点创建依赖关系
- 双击节点编辑标签