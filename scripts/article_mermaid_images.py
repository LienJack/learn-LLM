"""Extract Mermaid blocks from course articles and render them as PNG assets."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MERMAID_BLOCK_RE = re.compile(r"^```mermaid[^\n]*\n(?P<source>[\s\S]*?)\n```", re.MULTILINE)
PROMPT_COMMENT_RE = re.compile(r"<!--\s*mermaid-prompt:\s*(?P<prompt>.*?)\s*-->\s*$", re.DOTALL)


@dataclasses.dataclass(frozen=True)
class MermaidBlock:
    index: int
    source: str
    raw: str
    start: int
    end: int
    prompt: str
    prompt_start: int | None = None


@dataclasses.dataclass(frozen=True)
class MermaidAsset:
    id: str
    block: MermaidBlock
    article_path: Path
    workspace_dir: Path
    source_path: Path
    prompt_path: Path
    target_path: Path
    markdown_ref: str
    alt: str


def extract_mermaid_blocks(markdown: str) -> list[MermaidBlock]:
    blocks: list[MermaidBlock] = []
    for match in MERMAID_BLOCK_RE.finditer(markdown):
        prompt, prompt_start = extract_prompt_comment(markdown[: match.start()])
        blocks.append(
            MermaidBlock(
                index=len(blocks) + 1,
                source=clean_mermaid_source(match.group("source")),
                raw=match.group(0),
                start=match.start(),
                end=match.end(),
                prompt=prompt,
                prompt_start=prompt_start,
            )
        )
    return blocks


def extract_prompt_comment(prefix: str) -> tuple[str, int | None]:
    tail = prefix.rstrip()
    lines = tail.splitlines()
    window = "\n".join(lines[-4:])
    match = PROMPT_COMMENT_RE.search(window)
    if not match:
        return "", None
    prompt_start = len(prefix) - len(window) + match.start()
    return match.group("prompt").strip(), prompt_start


def clean_mermaid_source(source: str) -> str:
    cleaned = source.strip()
    fenced = re.match(r"^```(?:mermaid)?\n([\s\S]*?)\n```$", cleaned, flags=re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    return f"{cleaned}\n"


def plan_article(article_path: Path, repo_root: Path = REPO_ROOT) -> list[MermaidAsset]:
    article_path = article_path.resolve()
    markdown = article_path.read_text(encoding="utf-8")
    blocks = extract_mermaid_blocks(markdown)
    workspace_dir = article_path.parent / "assets" / article_path.stem
    return [asset_for_block(article_path, workspace_dir, block, repo_root) for block in blocks]


def asset_for_block(
    article_path: Path,
    workspace_dir: Path,
    block: MermaidBlock,
    repo_root: Path = REPO_ROOT,
) -> MermaidAsset:
    asset_id = f"mermaid-{block.index:02d}"
    source_path = workspace_dir / f"{asset_id}.mmd"
    prompt_path = workspace_dir / f"{asset_id}.prompt.md"
    target_path = workspace_dir / f"{asset_id}.png"
    return MermaidAsset(
        id=asset_id,
        block=block,
        article_path=article_path,
        workspace_dir=workspace_dir,
        source_path=source_path,
        prompt_path=prompt_path,
        target_path=target_path,
        markdown_ref=relative_markdown_path(article_path.parent, target_path),
        alt=f"{article_path.stem} diagram {block.index}",
    )


def relative_markdown_path(from_dir: Path, target_path: Path) -> str:
    return Path(os.path.relpath(target_path, from_dir)).as_posix()


def render_prompt(asset: MermaidAsset, repo_root: Path = REPO_ROOT) -> str:
    custom_prompt = asset.block.prompt or (
        "Render this Mermaid diagram as a clear in-article technical PNG. "
        "Keep the Mermaid source as the editable prompt of record."
    )
    return "\n".join(
        [
            f"# {asset.id} Mermaid render prompt",
            "",
            f"- Article: `{repo_relative(asset.article_path, repo_root)}`",
            f"- Source: `{repo_relative(asset.source_path, repo_root)}`",
            f"- Target: `{repo_relative(asset.target_path, repo_root)}`",
            "",
            "## Prompt",
            "",
            custom_prompt,
            "",
            "## Mermaid Source",
            "",
            "```mermaid",
            asset.block.source.rstrip(),
            "```",
            "",
        ]
    )


def write_sources(asset: MermaidAsset, repo_root: Path = REPO_ROOT) -> None:
    asset.workspace_dir.mkdir(parents=True, exist_ok=True)
    asset.source_path.write_text(asset.block.source, encoding="utf-8")
    asset.prompt_path.write_text(render_prompt(asset, repo_root), encoding="utf-8")


def render_mermaid(
    asset: MermaidAsset, mermaid_package: str, timeout_seconds: int
) -> dict[str, str]:
    asset.target_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "npx",
        "-y",
        mermaid_package,
        "-i",
        str(asset.source_path),
        "-o",
        str(asset.target_path),
        "-b",
        "transparent",
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return {"renderStatus": "generated"}
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        return {"renderStatus": "failed", "error": command_error(error)}


def compress_png(asset: MermaidAsset, imagemagick_bin: str) -> dict[str, object]:
    if not asset.target_path.exists():
        return {"compressionStatus": "missing"}
    before = asset.target_path.stat().st_size
    temp_path = asset.target_path.with_suffix(".compressed.png")
    command = [
        imagemagick_bin,
        str(asset.target_path),
        "-strip",
        "-define",
        "png:compression-level=9",
        "-define",
        "png:compression-filter=5",
        "-define",
        "png:compression-strategy=1",
        str(temp_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        after_candidate = temp_path.stat().st_size
        if after_candidate < before:
            temp_path.replace(asset.target_path)
        else:
            temp_path.unlink(missing_ok=True)
    except (OSError, subprocess.CalledProcessError) as error:
        temp_path.unlink(missing_ok=True)
        return {"compressionStatus": "failed", "compressionError": command_error(error)}
    after = asset.target_path.stat().st_size
    return {
        "compressionStatus": "compressed" if after < before else "unchanged",
        "compression": {
            "beforeBytes": before,
            "afterBytes": after,
            "savedBytes": max(0, before - after),
        },
    }


def update_markdown(article_path: Path, assets: list[MermaidAsset]) -> None:
    if not assets:
        return
    markdown = article_path.read_text(encoding="utf-8")
    pieces: list[str] = []
    cursor = 0
    for asset in sorted(assets, key=lambda item: item.block.start):
        replacement_start = asset.block.prompt_start or asset.block.start
        pieces.append(markdown[cursor:replacement_start])
        if pieces and not pieces[-1].endswith("\n\n"):
            pieces[-1] = pieces[-1].rstrip() + "\n\n"
        pieces.append(f"![{asset.alt}]({asset.markdown_ref})")
        cursor = asset.block.end
    pieces.append(markdown[cursor:])
    article_path.write_text("".join(pieces), encoding="utf-8")


def write_manifest(
    article_path: Path,
    assets: list[MermaidAsset],
    results: list[dict[str, object]],
    repo_root: Path,
) -> None:
    if not assets:
        return
    manifest_path = assets[0].workspace_dir / "mermaid-manifest.json"
    manifest = {
        "article": repo_relative(article_path, repo_root),
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "assets": [
            manifest_asset(asset, result, repo_root)
            for asset, result in zip(assets, results, strict=True)
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def manifest_asset(
    asset: MermaidAsset, result: dict[str, object], repo_root: Path
) -> dict[str, object]:
    item: dict[str, object] = {
        "id": asset.id,
        "kind": "mermaid",
        "block": asset.block.index,
        "sourcePath": repo_relative(asset.source_path, repo_root),
        "promptPath": repo_relative(asset.prompt_path, repo_root),
        "targetPath": repo_relative(asset.target_path, repo_root),
        "markdownRef": asset.markdown_ref,
        "sourceSha256": sha256_text(asset.block.source),
    }
    if asset.target_path.exists():
        item["bytes"] = asset.target_path.stat().st_size
        item["sha256"] = sha256_file(asset.target_path)
    item.update(result)
    return item


def repo_relative(path: Path, repo_root: Path = REPO_ROOT) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_error(error: BaseException) -> str:
    if isinstance(error, subprocess.CalledProcessError):
        detail = (error.stderr or error.stdout or "").strip()
        return detail or str(error)
    if isinstance(error, subprocess.TimeoutExpired):
        return f"Command timed out after {error.timeout} seconds."
    return str(error)


def run(args: argparse.Namespace, repo_root: Path = REPO_ROOT) -> int:
    if args.no_render:
        args.no_markdown = True
    article_paths = [Path(path).resolve() for path in args.articles]
    plans = [
        (article_path, plan_article(article_path, repo_root)) for article_path in article_paths
    ]
    total = sum(len(assets) for _, assets in plans)
    print(f"Mermaid articles: {len(plans)}")
    print(f"Mermaid blocks: {total}")
    for article_path, assets in plans:
        print(f"- {repo_relative(article_path, repo_root)}: {len(assets)}")
        for asset in assets:
            print(
                f"  {asset.id}: {repo_relative(asset.source_path, repo_root)} -> "
                f"{repo_relative(asset.target_path, repo_root)}"
            )
    if args.dry_run or not total:
        return 0
    if not args.yes:
        print(
            "Refusing to write without --yes. "
            "Re-run with --dry-run to inspect or --yes to execute.",
            file=sys.stderr,
        )
        return 2

    exit_code = 0
    for article_path, assets in plans:
        results: list[dict[str, object]] = []
        generated_assets: list[MermaidAsset] = []
        for asset in assets:
            write_sources(asset, repo_root)
            result: dict[str, object] = {"renderStatus": "skipped", "compressionStatus": "skipped"}
            if not args.no_render:
                result.update(render_mermaid(asset, args.mermaid_package, args.timeout))
            if result.get("renderStatus") == "generated" and not args.no_compress:
                result.update(compress_png(asset, args.imagemagick_bin))
            failed = (
                result.get("renderStatus") == "failed"
                or result.get("compressionStatus") == "failed"
            )
            if failed:
                exit_code = 1
            if result.get("renderStatus") == "generated":
                generated_assets.append(asset)
            results.append(result)
        if not args.no_markdown:
            update_markdown(article_path, generated_assets)
        write_manifest(article_path, assets, results, repo_root)
    return exit_code


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("articles", nargs="+", help="Markdown article files to process.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the Mermaid asset plan.")
    parser.add_argument("--yes", action="store_true", help="Write files and update Markdown.")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Write .mmd/.prompt.md/manifest without rendering PNG.",
    )
    parser.add_argument("--no-compress", action="store_true", help="Skip ImageMagick compression.")
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Do not replace Mermaid blocks with image refs.",
    )
    parser.add_argument(
        "--mermaid-package",
        default="@mermaid-js/mermaid-cli",
        help="Package used by npx.",
    )
    parser.add_argument("--imagemagick-bin", default="magick", help="ImageMagick executable.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Mermaid render timeout in seconds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
