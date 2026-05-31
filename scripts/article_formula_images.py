"""Extract plaintext formula blocks from course articles and render PNG assets."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_BLOCK_RE = re.compile(r"^```text[^\n]*\n(?P<source>[\s\S]*?)\n```", re.MULTILINE)
FORMULA_IMAGE_RE = re.compile(
    r"\n*<!--\s*formula-source: (?P<id>formula-\d+) sha256=(?P<sha>[0-9a-f]+)\n"
    r"(?P<source>[\s\S]*?)\n-->\n"
    r"!\[(?P<alt>[^\]]*)\]\((?P<ref>[^)]*)\)\n*",
    re.MULTILINE,
)

MATH_MARKERS = (
    "=",
    "<-",
    "@",
    "Σ",
    "Π",
    "√",
    "sqrt",
    "softmax",
    "KL(",
    "H(",
    "P(",
    "rank(",
    "||",
    "argmax",
    "exp(",
    "log",
    "LayerNorm",
    "RMSNorm",
    "dL/d",
    "theta",
    "Δ",
    "τ",
    "·",
    "≈",
    "CE(",
)

TEXTUAL_EXCLUSIONS = (
    "-> 需要",
    "-> 数学对象",
    "[必读]",
    "loss 是否对齐正确位置",
    "相似 !=",
    "rerank",
    "引用",
    "review gate",
)


@dataclasses.dataclass(frozen=True)
class FormulaBlock:
    index: int
    source: str
    raw: str
    start: int
    end: int
    source_hash: str


@dataclasses.dataclass(frozen=True)
class FormulaAsset:
    id: str
    block: FormulaBlock
    article_path: Path
    workspace_dir: Path
    source_path: Path
    target_path: Path
    markdown_ref: str
    alt: str


def extract_formula_blocks(markdown: str) -> list[FormulaBlock]:
    blocks: list[FormulaBlock] = []
    for match in TEXT_BLOCK_RE.finditer(strip_formula_images(markdown)):
        source = match.group("source").strip()
        if not is_formula_source(source):
            continue
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        blocks.append(
            FormulaBlock(
                index=len(blocks) + 1,
                source=source,
                raw=match.group(0),
                start=match.start(),
                end=match.end(),
                source_hash=source_hash,
            )
        )
    return blocks


def strip_formula_images(markdown: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        return f"\n```text\n{match.group('source').strip()}\n```\n"

    return FORMULA_IMAGE_RE.sub(replacement, markdown)


def is_formula_source(source: str) -> bool:
    compact = source.strip()
    if not compact:
        return False
    if any(exclusion in compact for exclusion in TEXTUAL_EXCLUSIONS):
        return False
    if not any(marker in compact for marker in MATH_MARKERS):
        return False
    lines = [line for line in compact.splitlines() if line.strip()]
    if len(lines) > 12:
        return False
    chinese_chars = sum(1 for char in compact if "\u4e00" <= char <= "\u9fff")
    math_chars = sum(1 for char in compact if char in "=<>+-*/@[]()_{}|·ΣΠΔτ≈%^")
    ascii_letters = sum(1 for char in compact if char.isascii() and char.isalpha())
    return math_chars + ascii_letters >= max(3, chinese_chars // 2)


def plan_article(article_path: Path, repo_root: Path = REPO_ROOT) -> list[FormulaAsset]:
    article_path = article_path.resolve()
    markdown = article_path.read_text(encoding="utf-8")
    blocks = extract_formula_blocks(markdown)
    workspace_dir = article_path.parent / "assets" / article_path.stem
    return [asset_for_block(article_path, workspace_dir, block, repo_root) for block in blocks]


def asset_for_block(
    article_path: Path,
    workspace_dir: Path,
    block: FormulaBlock,
    repo_root: Path = REPO_ROOT,
) -> FormulaAsset:
    asset_id = f"formula-{block.index:03d}"
    source_path = workspace_dir / f"{asset_id}.txt"
    target_path = workspace_dir / f"{asset_id}.png"
    return FormulaAsset(
        id=asset_id,
        block=block,
        article_path=article_path,
        workspace_dir=workspace_dir,
        source_path=source_path,
        target_path=target_path,
        markdown_ref=relative_markdown_path(article_path.parent, target_path),
        alt=f"{article_path.stem} formula {block.index}",
    )


def relative_markdown_path(from_dir: Path, target_path: Path) -> str:
    return Path(os.path.relpath(target_path, from_dir)).as_posix()


def write_source(asset: FormulaAsset) -> None:
    asset.workspace_dir.mkdir(parents=True, exist_ok=True)
    asset.source_path.write_text(f"{asset.block.source}\n", encoding="utf-8")


def render_formula(asset: FormulaAsset) -> dict[str, object]:
    asset.target_path.parent.mkdir(parents=True, exist_ok=True)
    source = asset.block.source
    regular_font = load_font(34, bold=False)
    mono_font = load_mono_font(32)
    font = mono_font if should_use_mono(source) else regular_font
    lines = source.splitlines()
    line_spacing = 12
    padding_x = 42
    padding_y = 32
    dummy = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bboxes = [draw.textbbox((0, 0), line or " ", font=font) for line in lines]
    widths = [bbox[2] - bbox[0] for bbox in bboxes]
    heights = [bbox[3] - bbox[1] for bbox in bboxes]
    width = max(widths, default=1) + padding_x * 2
    height = sum(heights) + line_spacing * max(0, len(lines) - 1) + padding_y * 2
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    y = padding_y
    for line, bbox, line_height in zip(lines, bboxes, heights, strict=True):
        x = padding_x
        draw.text((x, y - bbox[1]), line, fill=(20, 24, 31, 255), font=font)
        y += line_height + line_spacing
    image.save(asset.target_path, optimize=True)
    return {
        "renderStatus": "generated",
        "width": width,
        "height": height,
        "bytes": asset.target_path.stat().st_size,
    }


def should_use_mono(source: str) -> bool:
    if any("\u4e00" <= char <= "\u9fff" for char in source):
        return False
    return any(marker in source for marker in ("[", "]", ":", "@", "[:,", "W:"))


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    if bold:
        candidates.insert(0, "/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    return first_available_font(candidates, size)


def load_mono_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    return first_available_font(candidates, size)


def first_available_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def update_markdown(article_path: Path, assets: list[FormulaAsset]) -> None:
    markdown = strip_formula_images(article_path.read_text(encoding="utf-8"))
    pieces: list[str] = []
    cursor = 0
    for asset in sorted(assets, key=lambda item: item.block.start):
        pieces.append(markdown[cursor : asset.block.start])
        if pieces and not pieces[-1].endswith("\n\n"):
            pieces[-1] = pieces[-1].rstrip() + "\n\n"
        pieces.append(markdown_formula_ref(asset))
        cursor = asset.block.end
    pieces.append(markdown[cursor:])
    article_path.write_text("".join(pieces), encoding="utf-8")


def markdown_formula_ref(asset: FormulaAsset) -> str:
    source = asset.block.source.strip()
    return "\n".join(
        [
            f"<!-- formula-source: {asset.id} sha256={asset.block.source_hash}",
            source,
            "-->",
            f"![{asset.alt}]({asset.markdown_ref})",
            "",
        ]
    )


def write_manifest(
    article_path: Path,
    assets: list[FormulaAsset],
    results: list[dict[str, object]],
    repo_root: Path,
) -> None:
    if not assets:
        return
    manifest_path = assets[0].workspace_dir / "formula-manifest.json"
    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "article": repo_relative(article_path, repo_root),
        "generatedAt": generated_at,
        "count": len(assets),
        "assets": [
            {
                "id": asset.id,
                "sourceHash": asset.block.source_hash,
                "source": repo_relative(asset.source_path, repo_root),
                "target": repo_relative(asset.target_path, repo_root),
                "markdownRef": asset.markdown_ref,
                **result,
            }
            for asset, result in zip(assets, results, strict=True)
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def repo_relative(path: Path, repo_root: Path = REPO_ROOT) -> str:
    return Path(os.path.relpath(path, repo_root)).as_posix()


def run(article_path: Path, repo_root: Path = REPO_ROOT) -> list[dict[str, object]]:
    assets = plan_article(article_path, repo_root)
    results: list[dict[str, object]] = []
    for asset in assets:
        write_source(asset)
        results.append(render_formula(asset))
    update_markdown(article_path, assets)
    write_manifest(article_path, assets, results, repo_root)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render plaintext formula blocks from a course article as PNG assets."
    )
    parser.add_argument("article", type=Path, help="Markdown article to process.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run(args.article)
    generated = sum(1 for result in results if result.get("renderStatus") == "generated")
    print(f"Generated {generated} formula image(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
