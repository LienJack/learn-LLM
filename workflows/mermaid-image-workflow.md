# Mermaid 图片工作流

参考：`/Users/lienli/Documents/GitHub/learn-agent/images/workflows/mermaid-to-png.md`

目标：文章里的 Mermaid 不直接停留在正文中。处理后应同时保留可编辑提示词、Mermaid 源码、PNG 图片和 manifest 记录。

## 命令

先查看计划：

```bash
python -m scripts.article_mermaid_images --dry-run lessons/01_pytorch_training_intuition.md
```

确认后执行：

```bash
python -m scripts.article_mermaid_images --yes lessons/01_pytorch_training_intuition.md
```

批量处理：

```bash
python -m scripts.article_mermaid_images --yes lessons/*.md
```

## 产物

每篇文章的 Mermaid 产物写到文章旁：

```text
lessons/01_pytorch_training_intuition.md
lessons/assets/01_pytorch_training_intuition/
  mermaid-01.mmd
  mermaid-01.prompt.md
  mermaid-01.png
  mermaid-manifest.json
```

## 规则

- `.prompt.md` 是提示词事实来源，包含文章路径、目标路径和 Mermaid source。
- `.mmd` 是可编辑图表源码，不能只保留 PNG。
- 正文中的 fenced Mermaid block 会替换成 PNG Markdown 引用；紧邻 block 的 `mermaid-prompt` 注释会进入 `.prompt.md`，不保留在正文。
- Mermaid 渲染使用 `npx -y @mermaid-js/mermaid-cli`。
- PNG 默认继续走 ImageMagick `magick` 压缩；本地没有 ImageMagick 时可临时加 `--no-compress`。
- 调试或只想留下提示词/源码时可加 `--no-render`，脚本会避免改正文。
