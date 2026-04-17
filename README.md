# 🚇 南京地铁乘客需求分析

基于微博数据的马斯洛需求层次挖掘与情感分析

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## 📌 项目简介

本项目基于 **14,088 条**南京地铁相关的微博数据，通过大模型驱动的文本分析，构建了一套符合地铁场景的**马斯洛需求层次映射**，并结合情感分析与主题聚类，系统性地定位乘客满意点与痛点，为地铁运营优化提供数据支撑。

### 🎯 核心目标

| 层级 | 目标 | 产出 |
|------|------|------|
| 基础层 | 乘客需求结构化分类 | 五层需求体系（基础→保障→舒适→尊重→共鸣） |
| 中层 | 情感与需求的关联分析 | 正负面情感分布 + 高频主题识别 |
| 终极层 | 问题诊断与优先级排序 | 可落地的运营优化建议 |

## ✨ 项目亮点

- 🤖 **LLM 驱动的需求分类**：使用 DeepSeek / OpenAI API，通过精细化提示词将乘客反馈映射至五层需求体系
- 🔍 **情感因素精准提取**：从每条反馈中提取 3-7 字的核心服务要素（如"列车空调""安检服务""暖心播报"）
- 📊 **自动阈值聚类**：基于 Sentence-BERT 文本向量化，通过轮廓系数自动寻找最优聚类阈值
- 🏷️ **语义对齐**：使用 LLM 为每个聚类簇生成标准化表述
- 🕸️ **共现网络分析**：构建主题共现网络，使用 Louvain 社区检测发现"共发性问题"


## 📝 系列文章

本项目配套了五篇技术博客，从零讲解完整分析流程：

| 序号 | 文章标题 | 核心内容 |
|:---:|---------|---------|
| 1 | [抛弃繁琐微调！用 DeepSeek 零样本搞定 1.4 万条客诉分类](https://juejin.cn/post/7629293953226162222) | 提示词三版迭代，准确率从60%到91.2% |
| 2 | [LLM 结构化抽取实战：如何逼迫大模型严格输出"3-7字"核心要素](https://juejin.cn/post/7629602077840769075) | 要素提取提示词设计，成功率95% |
| 3 | [拒绝拍脑袋！Sentence-BERT 文本聚类如何用轮廓系数自动寻优](https://juejin.cn/post/7629359863177494574) | 层次聚类 + 轮廓系数自动选最优阈值 |
| 4 | [AI 帮你做总结：文本聚类后，如何用 DeepSeek 批量实现语义对齐](https://juejin.cn/post/7629503574843621426) | 聚类结果标准化表述 |
| 5 | [马斯洛理论重构地铁服务：挖掘 14,088 条文本背后的真实诉求](https://juejin.cn/post/7629598336643874862) | 完整业务洞察与运营建议 |

> 💻 **全套源码已开源**，欢迎 Star ⭐ 和交流讨论！


## 📁 项目结构

```
nanjing-metro-analysis/
├── README.md                          # 项目主页
├── LICENSE                            # 开源协议
├── .gitignore                         # Git 忽略配置
├── .env.example                       # 环境变量示例文件
├── requirements.txt                   # Python 依赖
├── data/
│   ├── raw/                           # 原始微博数据
│   └── processed/                     # 处理后的结果
├── scripts/
│   ├── 01_demand_classification/      # 需求层次分类
│   │   ├── classify_demand.py         # DeepSeek 版本（基础版）
│   │   ├── classify_demand_deepseek.py # DeepSeek 版本（带评价功能）
│   │   └── classify_demand_openai.py   # OpenAI 版本（带评价功能）
│   ├── 02_sentiment_analysis/         # 情感分析
│   │   └── analyze_sentiment.py
│   ├── 03_topic_extraction/           # 情感因素提取
│   │   └── extract_factors.py
│   ├── 04_clustering/                 # 聚类分析（待添加）
│   ├── 05_semantic_alignment/         # 语义对齐（待添加）
│   └── 06_network_analysis/           # 网络分析（待添加）
├── notebooks/                         # Jupyter Notebook 分析文件
│   ├── 01_optimal_threshold_clustering.ipynb
│   └── 02_semantic_alignment.ipynb
├── docs/                              # 详细文档
├── figures/                           # 图表输出
└── outputs/                           # 最终输出文件
```

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/Ula6/nanjing-metro-analysis.git
cd nanjing-metro-analysis
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置 API 密钥
```bash
# 复制示例配置文件
cp .env.example .env

# 编辑 .env 文件，填入你的真实 API 密钥
# DEEPSEEK_API_KEY=你的DeepSeek密钥
# OPENAI_API_KEY=你的OpenAI密钥
```

### 4. 准备数据
将你的微博 CSV 文件放入 `data/raw/` 目录，并确保包含 `微博正文` 列。

### 5. 运行分析

#### 第一步：需求层次分类
```bash
# 使用 DeepSeek API（基础版）
python scripts/01_demand_classification/classify_demand.py

# 或使用带评价功能的版本（需要 ground_truth.csv）
python scripts/01_demand_classification/classify_demand_deepseek.py
```

#### 第二步：情感分析
```bash
python scripts/02_sentiment_analysis/analyze_sentiment.py
```

#### 第三步：情感因素提取
```bash
python scripts/03_topic_extraction/extract_factors.py
```

#### 第四步：聚类与语义对齐
在 Jupyter Notebook 中运行：
- `notebooks/01_optimal_threshold_clustering.ipynb`
- `notebooks/02_semantic_alignment.ipynb`

> ⚠️ 注意：运行前请检查脚本中的文件路径配置，确保与你的本地路径一致。


## 📈 核心发现摘要

### 需求层次分布
- **基础层（安全与时效）**：列车故障、延误问题是最核心痛点
- **保障层（设施便利）**：电梯/扶梯缺失是突出负面因素
- **舒适层（环境体验）**：温度不适、拥挤问题最为集中
- **尊重层 & 共鸣层**：暖心播报、毕业祝福等活动获广泛好评

### 高频主题 Top 5
| 排名 | 主题 | 频次 | 情感倾向 |
|------|------|------|----------|
| 1 | 列车空调 | 584 | 负面 |
| 2 | 暖心祝福 | 546 | 正面 |
| 3 | 车厢拥挤 | 390 | 负面 |
| 4 | 设施不足 | 295 | 负面 |
| 5 | 服务贴心 | 161 | 正面 |

## 🔧 技术栈

- **语言**：Python 3.9+
- **LLM API**：DeepSeek Chat、OpenAI GPT-3.5-turbo
- **文本向量化**：Sentence-BERT (paraphrase-multilingual-mpnet-base-v2)
- **聚类算法**：层次聚类 (Ward's method)
- **网络分析**：NetworkX + Louvain 社区检测
- **数据处理**：Pandas、NumPy
- **可视化**：Matplotlib、Seaborn

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 📧 联系方式

- 作者：lsy
- GitHub：[Ula6](https://github.com/Ula6)

## ⭐ Star History

如果你觉得这个项目有用，请给一个 Star ⭐ 支持一下！
