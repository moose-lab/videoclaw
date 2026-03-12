# 这个王妃太狂野 — 交付物说明

**类型:** 穿越/古装/爽剧
**集数:** 5集 x 60秒
**画面比例:** 9:16 竖屏
**LLM:** Kimi K2 Thinking (via Evolink API)
**生成日期:** 2026-03-12

---

## 目录结构

```
这个王妃太狂野/
├── README.md                          ← 你正在看的文件
│
├── 这个王妃太狂野_完整剧本.md          ← 人可读的完整剧本（编剧/导演审阅用）
│
├── images/
│   └── characters/                    ← 角色参考图（Seedream 5.0 生成）
│       ├── 林薇.png                   ← 女主角 — 穿越女特工/废物王妃
│       ├── 萧衍.png                   ← 男主角 — 冷面靖王
│       ├── 慕容雪.png                 ← 反派 — 阴毒侧妃
│       └── 老太君.png                 ← 配角 — 萧家老太君
│
├── audio/
│   └── voice_samples/                 ← 角色声线样本（Edge TTS 生成）
│       ├── 林薇_voice_sample.mp3      ← 冷静知性女声
│       ├── 萧衍_voice_sample.mp3      ← 低沉威严男声
│       ├── 慕容雪_voice_sample.mp3    ← 甜美活泼女声
│       ├── 老太君_voice_sample.mp3    ← 沉稳老练女声
│       └── voice_mapping.json         ← 声线映射元数据
│
├── executor/                          ← 结构化数据（交付给执行器管线）
│   ├── characters.json                ← 角色完整数据（含 voice_profile + 图片路径）
│   ├── ep01_executor_data.json        ← 第1集执行器数据
│   ├── ep02_executor_data.json        ← 第2集执行器数据
│   ├── ep03_executor_data.json        ← 第3集执行器数据
│   ├── ep04_executor_data.json        ← 第4集执行器数据
│   └── ep05_executor_data.json        ← 第5集执行器数据
│
├── final_series.json                  ← 完整序列化数据（DramaSeries 全量）
├── 00_series_outline.json             ← 剧集大纲原始数据
├── ep02_当众打脸_script.json           ← 第2集 LLM 原始输出
├── ep03_深夜试探_script.json           ← 第3集 LLM 原始输出
├── ep04_至暗时刻_script.json           ← 第4集 LLM 原始输出
└── ep05_王妃驾到_script.json           ← 第5集 LLM 原始输出
```

---

## 怎么看？按角色分工

### 编剧 / 导演 审阅

打开 **`这个王妃太狂野_完整剧本.md`** — 包含：
- 角色表（人设 + 视觉 + TTS 配置）
- 每集每场景的完整分镜（景别、运镜、情绪、台词、旁白、转场）
- 悬念和配乐方案

### 执行器 / 开发者 对接

打开 **`executor/`** 目录 — 每集一个 JSON 文件，字段直接对应下游执行器：

```json
{
  "episode_number": 2,
  "episode_title": "当众打脸",
  "duration_seconds": 60.0,
  "scenes": [
    {
      "scene_id": "ep02_s01",
      "visual_prompt": "...",        → 视频生成器 (video_gen)
      "shot_scale": "close_up",      → 构图控制
      "shot_type": "detail",         → 镜头类型
      "camera_movement": "static",   → 运镜指令
      "duration_seconds": 5.0,       → 时长控制
      "dialogue": "...",             → TTS 对话合成
      "narration": "...",            → TTS 旁白合成
      "speaking_character": "林薇",   → 角色声音路由
      "emotion": "tense",            → 氛围/音乐匹配
      "characters_present": ["林薇"], → 角色一致性检查
      "transition": "fade_in"        → 转场效果
    }
  ],
  "characters": [...],               → 含 voice_profile 的完整角色数据
  "music": {"style": "...", "mood": "...", "tempo": 85},
  "voice_over": {"text": "...", "tone": "...", "language": "zh"},
  "cliffhanger": "..."
}
```

### 角色参考图

打开 **`images/characters/`** — 4 张角色全身参考图（Seedream 5.0 生成）：
- 用于视频生成时保持角色视觉一致性
- 每个角色的 `visual_prompt` 已嵌入每个场景的 JSON 中

### 角色声线样本

打开 **`audio/voice_samples/`** — 4 个角色声线 MP3 样本（Edge TTS 生成）：
- 每个角色一段代表性台词，展示该角色的语音风格
- `voice_mapping.json` 记录每个角色的 TTS 声线映射
- 生产环境使用 WaveSpeed MiniMax speech-02-hd（`voice_profile` 中配置）

### 角色数据（含语音配置）

打开 **`executor/characters.json`** — 每个角色包含：

| 字段 | 用途 |
|------|------|
| `name` | 角色名 |
| `visual_prompt` | AI 视频生成的角色外貌描述（英文） |
| `voice_style` | 语义化声音风格 |
| `voice_profile.voice_id` | MiniMax speech-02-hd 声音 ID |
| `voice_profile.speed` | 语速 |
| `voice_profile.pitch` | 音调 |
| `reference_image` | 角色参考图路径 |
| `voice_sample` | 角色声线样本路径 |

---

## 字段 → 执行器映射

| 字段 | 执行器 / 任务 |
|------|--------------|
| `visual_prompt` | video_gen — AI 视频生成 |
| `shot_scale` / `shot_type` | video_gen — 构图和镜头类型 |
| `camera_movement` | video_gen — 运镜指令 |
| `dialogue` + `speaking_character` | TTS (WaveSpeed/MiniMax speech-02-hd) — 角色对话 |
| `narration` | TTS — 旁白合成 |
| `voice_profile` | TTS — 声音参数（voice_id/speed/pitch/emotion） |
| `emotion` | PromptEnhancer — 视觉氛围增强 + 配乐匹配 |
| `transition` | Compose — 转场效果 |
| `music` | Music Generator — 背景音乐 |
| `cliffhanger` | 跨集叙事 — 下一集开场衔接 |
| `characters_present` | 一致性检查 — 确保角色外貌跨场景一致 |
| `reference_image` | 角色参考图注入 — 视频生成角色一致性 |
| `duration_seconds` | 时长控制 — 场景和全集时长校准 |

---

## 快速开始

```bash
# 1. 查看完整剧本
cat docs/deliverables/这个王妃太狂野/这个王妃太狂野_完整剧本.md

# 2. 查看第2集执行器数据
python3 -m json.tool docs/deliverables/这个王妃太狂野/executor/ep02_executor_data.json

# 3. 查看角色数据和语音配置
python3 -m json.tool docs/deliverables/这个王妃太狂野/executor/characters.json

# 4. 在 Python 中加载完整数据
python3 -c "
from videoclaw.drama.models import DramaSeries
import json
series = DramaSeries.from_dict(json.load(open('docs/deliverables/这个王妃太狂野/final_series.json')))
print(f'{series.title}: {len(series.characters)} characters, {len(series.episodes)} episodes')
for ep in series.episodes:
    print(f'  Ep{ep.number} {ep.title}: {len(ep.scenes)} scenes')
"
```

---

## 数据来源

| 集数 | 生成方式 | 模型 |
|------|---------|------|
| 第1集 废妃重生 | Mock 数据（E2E 测试验证用） | — |
| 第2集 当众打脸 | LLM 生成 | kimi-k2-thinking-turbo |
| 第3集 深夜试探 | LLM 生成 | kimi-k2-thinking-turbo |
| 第4集 至暗时刻 | LLM 生成 | kimi-k2-thinking (streaming) |
| 第5集 王妃驾到 | LLM 生成 | kimi-k2-thinking (streaming) |
| 角色参考图 | AI 生成 | doubao-seedream-5.0-lite |
| 角色声线样本 | TTS 合成 | Edge TTS (demo) / MiniMax speech-02-hd (prod) |
