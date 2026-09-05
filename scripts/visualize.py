"""Render one scene to a standalone HTML file.

    python scripts/visualize.py --ckpt /path/ckpt_step3000.pt --out /content/scene.html

Runs in its own interpreter, which is why this exists alongside `reify.viz`:
importing a newly added module inside a long-lived notebook kernel fails when an
editable install's finder was built before that module existed. A subprocess has
no such state.

The output is self-contained, so it opens in any browser and can be dropped into
a notebook with IPython.display.HTML.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reify.viz.plotly_view import show_scene  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--scene", default=None, help="scene id; omit to pick at random")
    parser.add_argument("--seed", type=int, default=None, help="seed for the random pick")
    parser.add_argument("--out", default="/content/scene.html")
    parser.add_argument("--max-points", type=int, default=30_000)
    parser.add_argument("--point-size", type=float, default=1.8)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--cdn", action="store_true",
                        help="link plotly.js from a CDN instead of inlining it, "
                             "for a much smaller file that needs network access")
    args = parser.parse_args()

    fig, scene_id, info = show_scene(
        ckpt=args.ckpt,
        scene_id=args.scene,
        split=args.split,
        config=args.config,
        max_points=args.max_points,
        seed=args.seed,
        point_size=args.point_size,
        height=args.height,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn" if args.cdn else "inline",
                   full_html=True)

    print(json.dumps(info, indent=2, default=str))
    print(f"[viz] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
