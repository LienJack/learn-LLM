from pathlib import Path

from scripts.article_mermaid_images import (
    extract_mermaid_blocks,
    plan_article,
    render_prompt,
    update_markdown,
)


def test_extract_mermaid_blocks_keeps_prompt_comment() -> None:
    markdown = """# Title

<!-- mermaid-prompt: Show the training loop with a feedback arrow. -->
```mermaid
graph TD
  A[Data] --> B[Loss]
```
"""

    blocks = extract_mermaid_blocks(markdown)

    assert len(blocks) == 1
    assert blocks[0].prompt == "Show the training loop with a feedback arrow."
    assert blocks[0].prompt_start is not None
    assert blocks[0].source == "graph TD\n  A[Data] --> B[Loss]\n"


def test_plan_article_uses_article_asset_workspace(tmp_path: Path) -> None:
    article = tmp_path / "lessons" / "01_intro.md"
    article.parent.mkdir()
    article.write_text(
        """# Intro

```mermaid
flowchart LR
  A --> B
```
""",
        encoding="utf-8",
    )

    [asset] = plan_article(article, repo_root=tmp_path)

    assert asset.source_path == (
        tmp_path / "lessons" / "assets" / "01_intro" / "mermaid-01.mmd"
    )
    assert asset.prompt_path == (
        tmp_path / "lessons" / "assets" / "01_intro" / "mermaid-01.prompt.md"
    )
    assert asset.target_path == (
        tmp_path / "lessons" / "assets" / "01_intro" / "mermaid-01.png"
    )
    assert asset.markdown_ref == "assets/01_intro/mermaid-01.png"


def test_render_prompt_contains_mermaid_source_and_paths(tmp_path: Path) -> None:
    article = tmp_path / "lesson.md"
    article.write_text(
        """# Lesson

```mermaid
flowchart LR
  A --> B
```
""",
        encoding="utf-8",
    )
    [asset] = plan_article(article, repo_root=tmp_path)

    prompt = render_prompt(asset, repo_root=tmp_path)

    assert "## Prompt" in prompt
    assert "`lesson.md`" in prompt
    assert "```mermaid\nflowchart LR\n  A --> B\n```" in prompt


def test_update_markdown_replaces_each_mermaid_block(tmp_path: Path) -> None:
    article = tmp_path / "lesson.md"
    article.write_text(
        """# Lesson

Before.

```mermaid
flowchart LR
  A --> B
```

After.
""",
        encoding="utf-8",
    )
    assets = plan_article(article, repo_root=tmp_path)

    update_markdown(article, assets)

    updated = article.read_text(encoding="utf-8")
    assert "```mermaid" not in updated
    assert "![lesson diagram 1](assets/lesson/mermaid-01.png)" in updated
    assert "Before." in updated
    assert "After." in updated


def test_update_markdown_removes_prompt_comment_after_extracting_it(tmp_path: Path) -> None:
    article = tmp_path / "lesson.md"
    article.write_text(
        """# Lesson

<!-- mermaid-prompt: Draw a compact training loop. -->
```mermaid
flowchart LR
  A --> B
```
""",
        encoding="utf-8",
    )
    assets = plan_article(article, repo_root=tmp_path)

    update_markdown(article, assets)

    updated = article.read_text(encoding="utf-8")
    assert "mermaid-prompt" not in updated
    assert "![lesson diagram 1](assets/lesson/mermaid-01.png)" in updated
