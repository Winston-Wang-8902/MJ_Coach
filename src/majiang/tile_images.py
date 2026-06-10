"""
Load pre-cropped tile images for all 27 Sichuan Mahjong tile types.

    from majiang.tile_images import build_tile_images
    images = build_tile_images()   # {tile_str: PIL.Image}

Call build_tile_images_tk() inside a live Tk root to get ImageTk.PhotoImage objects.
"""

from __future__ import annotations
from pathlib import Path
from PIL import Image

TILE_W = 46
TILE_H = 64

_ASSETS = Path(__file__).parent / "tile_assets"


def build_tile_images() -> dict[str, Image.Image]:
    """Return {tile_str: PIL.Image} for all 27 tile types, resized to TILE_W×TILE_H."""
    imgs: dict[str, Image.Image] = {}
    for suit in "mps":
        for n in range(1, 10):
            key = f"{n}{suit}"
            img = Image.open(_ASSETS / f"{key}.png").convert("RGB")
            if img.size != (TILE_W, TILE_H):
                img = img.resize((TILE_W, TILE_H), Image.LANCZOS)
            imgs[key] = img
    return imgs


def build_tile_images_tk() -> dict[str, object]:
    """Return {tile_str: ImageTk.PhotoImage}.  Must be called inside a live Tk root."""
    from PIL import ImageTk
    return {k: ImageTk.PhotoImage(v) for k, v in build_tile_images().items()}
