# 领域小模型项目模板

教学目标：把法律、医学项目中的共通工程结构抽象成可复用模板，迁移到金融、教育、客服、企业知识库等领域。

## 标准目录

```text
domain_model_template/
├── configs/
├── data/
├── scripts/
├── src/
├── tests/
├── reports/
└── README.md
```

## 必备契约

- 数据版本、清洗脚本、切分规则可追踪。
- 训练配置、模型版本、RAG index 版本写入 `run_manifest.json`。
- 评测报告、失败案例、风险报告和 model card 互相引用。
- 发布前必须通过 regression eval、安全 eval 和 rollback 检查。

## 学习入口

对应课程章节：`lessons/19_domain_model_template.md`。
