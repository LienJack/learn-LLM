from pathlib import Path

from scripts.article_formula_images import (
    extract_formula_blocks,
    plan_article,
    strip_formula_images,
    update_markdown,
)


def test_extract_formula_blocks_selects_math_text_blocks() -> None:
    markdown = """# Lesson

```text
y = f(x; theta)
```

```text
真实困惑
  -> 生活类比
```
"""

    blocks = extract_formula_blocks(markdown)

    assert len(blocks) == 1
    assert blocks[0].source == "y = f(x; theta)"


def test_plan_article_uses_article_asset_workspace(tmp_path: Path) -> None:
    article = tmp_path / "lessons" / "math.md"
    article.parent.mkdir()
    article.write_text(
        """# Math

```text
loss = -log p_correct
```
""",
        encoding="utf-8",
    )

    [asset] = plan_article(article, repo_root=tmp_path)

    assert asset.source_path == tmp_path / "lessons" / "assets" / "math" / "formula-001.txt"
    assert asset.target_path == tmp_path / "lessons" / "assets" / "math" / "formula-001.png"
    assert asset.markdown_ref == "assets/math/formula-001.png"


def test_update_markdown_replaces_formula_block_with_image_and_source_comment(
    tmp_path: Path,
) -> None:
    article = tmp_path / "math.md"
    article.write_text(
        """# Math

Before.

```text
KL(P || Q) = H(P, Q) - H(P)
```

After.
""",
        encoding="utf-8",
    )
    assets = plan_article(article, repo_root=tmp_path)

    update_markdown(article, assets)

    updated = article.read_text(encoding="utf-8")
    assert "```text" not in updated
    assert "<!-- formula-source: formula-001" in updated
    assert "KL(P || Q) = H(P, Q) - H(P)" in updated
    assert "![math formula 1](assets/math/formula-001.png)" in updated
    assert "Before." in updated
    assert "After." in updated


def test_strip_formula_images_restores_plaintext_block() -> None:
    markdown = """# Math

<!-- formula-source: formula-001 sha256=abc123
y = wx + b
-->
![math formula 1](assets/math/formula-001.png)
"""

    stripped = strip_formula_images(markdown)

    assert "![math formula" not in stripped
    assert "```text\ny = wx + b\n```" in stripped
