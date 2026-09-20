"""Sample CLI demonstrating SAM 3.1 media segmentation (Meta Model API).

Usage:
    sam3 "glasses" ./photo.png
    sam3 --prompt "glasses" --image ./photo.png --overlay ./out.png

Sends a text concept prompt + an image to the ``sam-3.1`` model over the
Responses API (``https://api.meta.ai/v1``) and prints the returned
boxes / raw special-token output. Optionally draws box overlays.

Auth: create a ``.env`` file in the project folder containing::

    MODEL_API_KEY=your-key-here

Docs: https://dev.meta.ai/docs/media-segmentation
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import re
import sys
from pathlib import Path

API_BASE_URL = "https://api.meta.ai/v1"
MODEL_ID = "sam-3.1"

# <0f>0<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..|><|mask;x=0;y=0;data=H,W,~payload|>,1...
BOX_RE = re.compile(
    r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>"
)
RECORD_RE = re.compile(
    r"(\d+)"  # object id
    r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>"
    r"<\|mask;x=0;y=0;data=(\d+),(\d+),([~!])"
)


def load_api_key() -> str:
    """Load MODEL_API_KEY from the environment (and .env if present)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None  # type: ignore[assignment]
    if load_dotenv is not None:
        load_dotenv()  # picks up ./.env by default
    api_key = os.environ.get("MODEL_API_KEY", "").strip()
    if not api_key:
        print(
            "error: MODEL_API_KEY is not set.\n"
            "Create a .env file in this project folder with:\n"
            "    MODEL_API_KEY=your-key-here",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return api_key


def image_to_data_url(image: str) -> str:
    """Accept an http(s) URL (passed through) or a local file (base64 data URL)."""
    if image.startswith(("http://", "https://", "data:")):
        return image
    path = Path(image)
    if not path.is_file():
        print(f"error: image file not found: {image}", file=sys.stderr)
        raise SystemExit(2)
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None or not mime.startswith("image/"):
        # Default to png; the API sniffs content anyway.
        mime = "image/png"
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def call_sam31(prompt: str, image_url: str, mask_encoding: str, stream: bool) -> str:
    """Call sam-3.1 and return the accumulated output_text (special-token line)."""
    try:
        from openai import OpenAI
    except ImportError:
        print(
            "error: the 'openai' package is not installed. Run: uv sync",
            file=sys.stderr,
        )
        raise SystemExit(2)

    client = OpenAI(base_url=API_BASE_URL, api_key=load_api_key())
    payload: dict = {
        "model": MODEL_ID,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
    }
    if mask_encoding != "lossless":
        payload["metadata"] = {"mask_encoding": mask_encoding}

    if stream:
        chunks: list[str] = []
        stream_obj = client.responses.create(stream=True, **payload)  # type: ignore[arg-type]
        for event in stream_obj:
            if getattr(event, "type", "") == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                chunks.append(delta)
                print(delta, end="", flush=True)
        print()
        return "".join(chunks)

    response = client.responses.create(**payload)  # type: ignore[arg-type]
    # SDK convenience property; fall back to walking the output list.
    text = getattr(response, "output_text", "") or ""
    if not text:
        parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for block in getattr(item, "content", []) or []:
                if getattr(block, "type", "") == "output_text":
                    parts.append(getattr(block, "text", "") or "")
        text = "".join(parts)
    return text


def parse_records(output_text: str) -> list[dict]:
    """Parse the SAM wire format into [{object_id, box, mask_meta}]."""
    records: list[dict] = []
    for m in RECORD_RE.finditer(output_text):
        (obj_id, x1, y1, x2, y2, w, h, mh, mw, enc) = m.groups()
        records.append(
            {
                "object_id": int(obj_id),
                "box": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "w": int(w),
                    "h": int(h),
                },
                "mask": {"h": int(mh), "w": int(mw), "encoding": enc},
            }
        )
    return records


def draw_box_overlay(image_path: str, records: list[dict], out_path: Path) -> None:
    """Draw boxes on a copy of the source image and save it."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print(
            "warning: Pillow is not installed, skipping overlay "
            "(run: uv sync).",
            file=sys.stderr,
        )
        return
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for rec in records:
        b = rec["box"]
        # Cycle a few outline colours so multiple instances are distinguishable.
        color = ["red", "lime", "cyan", "yellow", "magenta", "orange"][
            rec["object_id"] % 6
        ]
        draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline=color, width=3)
        draw.text((b["x1"] + 4, b["y1"] + 4), f"id={rec['object_id']}", fill=color)
    img.save(out_path)
    print(f"overlay saved to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sam3",
        description="Segment an image with SAM 3.1 (Meta Model API).",
    )
    p.add_argument("prompt", nargs="?", help='concept noun phrase, e.g. "glasses"')
    p.add_argument("image", nargs="?", help="local image file or http(s) image URL")
    p.add_argument("--prompt", dest="prompt_opt", help="same as positional prompt")
    p.add_argument("--image", dest="image_opt", help="same as positional image")
    p.add_argument(
        "--mask-encoding",
        choices=["lossless", "one_bit"],
        default="lossless",
        help="mask payload encoding (default: lossless; one_bit is smaller)",
    )
    p.add_argument(
        "--stream",
        action="store_true",
        help="stream response deltas instead of a single request",
    )
    p.add_argument(
        "--overlay",
        help="where to save the box-overlay PNG "
        "(default: <image-stem>_segmented.png; use --no-overlay to skip)",
    )
    p.add_argument(
        "--no-overlay",
        action="store_true",
        help="do not write an overlay image",
    )
    p.add_argument(
        "--raw-output",
        help="write the raw special-token output_text to this file",
    )
    p.add_argument(
        "--svg-dir",
        help="export per-object non-rectangular SVG outlines to this directory",
    )
    return p


def export_svg_outlines(output_text: str, records: list[dict], svg_dir: str) -> None:
    """Decode masks to per-object SVG outline files (pure Python).

    Uses the native ``meta-sam-parser`` package (same contract as the
    TypeScript ``@meta-sam/parser``): each mask becomes a polygonal
    ``M...L...Z`` path scaled from raster coords into source pixels via
    the record's ``bounds``, drawn over a ``viewBox`` of the source image.
    """
    try:
        from meta_sam_parser import (
            CompletedOutcome,
            decode_mask_to_svg_path,
            image_segmentation_format,
        )
    except ImportError:
        print(
            "error: --svg-dir needs the 'meta-sam-parser' package. "
            "Run: uv sync",
            file=sys.stderr,
        )
        raise SystemExit(2)

    out = Path(svg_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = output_text.strip()

    parser = image_segmentation_format().create_parser()
    parser.push(raw)
    finished = parser.finish(CompletedOutcome())
    result = finished["result"] if isinstance(finished, dict) else finished.result

    for d in result.diagnostics or ():
        print(f"parser {d.severity} [{d.code}] line {d.line}: {d.message}")

    boxes = {b.object_id: b for b in result.records if b.kind == "box"}
    masks = [r for r in result.records if r.kind == "mask"]
    if not masks:
        print("no mask records parsed; nothing to export.")
        return

    if records:
        frame_w, frame_h = records[0]["box"]["w"], records[0]["box"]["h"]
    else:
        frame_w = max(
            [b.right for b in boxes.values()]
            + [m.bounds.right for m in masks]
        )
        frame_h = max(
            [b.bottom for b in boxes.values()]
            + [m.bounds.bottom for m in masks]
        )

    palette = ["#ff0000", "#00c000", "#00bfff", "#ffd400", "#ff00ff", "#ff8000"]
    for m in masks:
        try:
            oid = int(m.object_id)
        except (TypeError, ValueError):
            oid = m.object_id
        color = palette[int(m.object_id) % len(palette)] if str(
            m.object_id
        ).lstrip("-").isdigit() else palette[0]
        path = decode_mask_to_svg_path(m.mask)  # "" when the mask is empty
        bw = m.bounds.right - m.bounds.left
        bh = m.bounds.bottom - m.bounds.top
        sx = bw / m.mask.width if m.mask.width else 1
        sy = bh / m.mask.height if m.mask.height else 1
        box = boxes.get(m.object_id)
        svg_file = out / f"mask_{oid}.svg"
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {frame_w} {frame_h}" '
            f'width="{frame_w}" height="{frame_h}">'
        ]
        if box is not None:
            lines.append(
                f'  <rect x="{box.left}" y="{box.top}" '
                f'width="{box.right - box.left}" height="{box.bottom - box.top}" '
                f'fill="none" stroke="{color}" stroke-width="3"/>'
            )
        if path:
            stroke_w = 2 / min(sx, sy) if min(sx, sy) else 2
            lines.append(
                f'  <path d="{path}" '
                f'transform="translate({m.bounds.left} {m.bounds.top}) '
                f'scale({sx} {sy})" fill="{color}" fill-opacity="0.35" '
                f'stroke="{color}" stroke-width="{stroke_w:.2f}" '
                f'fill-rule="evenodd"/>'
            )
        else:
            lines.append(f"  <!-- empty mask for object {oid} -->")
        lines.append("</svg>")
        svg_file.write_text("\n".join(lines) + "\n")
        print(
            f"id={oid} outline -> {svg_file} "
            f"(mask {m.mask.height}x{m.mask.width})"
        )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    prompt = args.prompt_opt or args.prompt
    image = args.image_opt or args.image
    if not prompt or not image:
        parser.print_usage(sys.stderr)
        print("error: a prompt and an image are required.", file=sys.stderr)
        print('example: sam3 "glasses" ./photo.png', file=sys.stderr)
        raise SystemExit(2)

    image_url = image_to_data_url(image)
    print(f'model: {MODEL_ID}\nprompt: "{prompt}"\nimage: {image}')

    output_text = call_sam31(prompt, image_url, args.mask_encoding, args.stream)

    if not output_text.strip():
        print("no matches: the model returned empty output_text.")
        return

    print("\n--- raw output_text ---")
    print(output_text.strip())

    if args.raw_output:
        Path(args.raw_output).write_text(output_text.strip() + "\n")
        print(f"raw output saved to {args.raw_output}")

    records = parse_records(output_text)
    print(f"\n--- {len(records)} match(es) ---")
    for rec in records:
        b = rec["box"]
        m = rec["mask"]
        print(
            f"id={rec['object_id']} "
            f"box=({b['x1']},{b['y1']})-({b['x2']},{b['y2']}) "
            f"on {b['w']}x{b['h']} "
            f"mask={m['h']}x{m['w']} "
            f"encoding={'lossless' if m['encoding'] == '~' else 'one_bit'}"
        )
    if not records:
        print("(could not parse boxes from output; raw text above is authoritative)")

    if args.svg_dir:
        if not records:
            print("skipping SVG export: no records parsed.")
        else:
            export_svg_outlines(output_text, records, args.svg_dir)

    if args.no_overlay or image.startswith(("http://", "https://", "data:")):
        if not args.no_overlay and not Path(image).is_file():
            print("skipping overlay: --overlay needs a local image file.")
        return
    out_path = (
        Path(args.overlay)
        if args.overlay
        else Path(image).with_name(f"{Path(image).stem}_segmented.png")
    )
    draw_box_overlay(image, records, out_path)


if __name__ == "__main__":
    main()
