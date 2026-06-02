#!/usr/bin/env python3
"""Render a Python-vs-Rust benchmark comparison as SVG."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


DEFAULT_NAMES = [
    "media_jpeg",
    "encode",
    "dynamics",
    "score",
    "full",
    "policy_rollout",
    "policy_sample_fixed",
    "policy_sample_generated",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True, type=Path, dest="python_json")
    parser.add_argument("--rust", required=True, type=Path, dest="rust_json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metric", default="p50_ms", choices=("mean_ms", "p50_ms", "p95_ms", "p99_ms"))
    parser.add_argument("--title", default="TD-MPC2 CUDA Runtime Latency")
    parser.add_argument("--names", nargs="*", default=DEFAULT_NAMES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = load_stats(args.python_json)
    rust = load_stats(args.rust_json)
    rows = []
    for name in args.names:
        if name in python and name in rust:
            py = float(python[name][args.metric])
            rs = float(rust[name][args.metric])
            rows.append((name, py, rs, py / rs if rs > 0 else 0.0))
    if not rows:
        raise RuntimeError("no common benchmark rows to plot")

    svg = render_svg(args.title, args.metric, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(f"wrote {args.output}")


def load_stats(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["name"]: row for row in payload["stats"]}


def render_svg(title: str, metric: str, rows: list[tuple[str, float, float, float]]) -> str:
    width = 1120
    left = 230
    right = 240
    top = 88
    row_h = 62
    chart_w = width - left - right
    height = top + 40 + row_h * len(rows)
    max_value = max(max(py, rs) for _, py, rs, _ in rows)
    scale = chart_w / max_value if max_value > 0 else 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<style>",
        "text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#172026}",
        ".title{font-size:28px;font-weight:760}.sub{font-size:14px;fill:#52606d}.label{font-size:15px;font-weight:650}",
        ".value{font-size:13px;fill:#303b45}.axis{stroke:#d7dde3;stroke-width:1}.py{fill:#c2410c}.rs{fill:#0f766e}",
        ".grid{stroke:#edf1f5;stroke-width:1}.speed{font-size:13px;font-weight:700;fill:#172026}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="28" y="38">{html.escape(title)}</text>',
        f'<text class="sub" x="28" y="62">Metric: {html.escape(metric)}. Lower is faster. TD-MPC2 CUDA; media_jpeg includes decode and model-tensor preprocessing.</text>',
        f'<text class="sub" x="{left}" y="82">Python / PyTorch</text>',
        f'<rect class="py" x="{left + 125}" y="70" width="14" height="14" rx="2"/>',
        f'<text class="sub" x="{left + 155}" y="82">Rust / Candle</text>',
        f'<rect class="rs" x="{left + 250}" y="70" width="14" height="14" rx="2"/>',
    ]

    ticks = 4
    for tick in range(ticks + 1):
        value = max_value * tick / ticks
        x = left + value * scale
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - 28}"/>')
        parts.append(f'<text class="sub" x="{x:.1f}" y="{height - 10}" text-anchor="middle">{value:.1f}ms</text>')
    parts.append(
        f'<line class="axis" x1="{left}" y1="{height - 28}" x2="{left + chart_w}" y2="{height - 28}"/>'
    )

    for idx, (name, py, rs, speedup) in enumerate(rows):
        y = top + idx * row_h
        py_w = py * scale
        rs_w = rs * scale
        py_value_x, py_anchor = value_label(left, chart_w, py_w)
        rs_value_x, rs_anchor = value_label(left, chart_w, rs_w)
        parts.extend(
            [
                f'<text class="label" x="28" y="{y + 27}">{html.escape(name)}</text>',
                f'<rect class="py" x="{left}" y="{y + 8}" width="{py_w:.1f}" height="18" rx="4"/>',
                f'<rect class="rs" x="{left}" y="{y + 32}" width="{rs_w:.1f}" height="18" rx="4"/>',
                f'<text class="value" x="{py_value_x:.1f}" y="{y + 22}" text-anchor="{py_anchor}">{py:.3f} ms</text>',
                f'<text class="value" x="{rs_value_x:.1f}" y="{y + 46}" text-anchor="{rs_anchor}">{rs:.3f} ms</text>',
                f'<text class="speed" x="{left + chart_w + 24}" y="{y + 35}">{speedup:.2f}x</text>',
            ]
        )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def value_label(left: int, chart_w: int, bar_w: float) -> tuple[float, str]:
    outside_x = left + bar_w + 8
    if outside_x <= left + chart_w - 74:
        return outside_x, "start"
    return left + bar_w - 8, "end"


if __name__ == "__main__":
    main()
