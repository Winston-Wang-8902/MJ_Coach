"""
四川麻将 Coach — Flask web server
Run locally:   python3 app.py
Production:    gunicorn app:app --bind 0.0.0.0:$PORT
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, jsonify, request, send_from_directory
from majiang import (
    Tile, random_hand, analyze_hand, check_tenpai, check_tenpai_any,
    random_no_win_p14, random_no_win_p_n,
    random_one_suit_hand, random_two_suit_tenpai_hand,
    find_tenpai_improvements,
)

app = Flask(__name__, static_folder="static", template_folder="templates")

_TILE_ASSETS = os.path.join(os.path.dirname(__file__), "src", "majiang", "tile_assets")

_VALID_STUDY_N = {5, 8, 11, 14}


# ── pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def hub():
    return send_from_directory("templates", "hub.html")

@app.route("/coach")
def coach():
    return send_from_directory("templates", "index.html")

@app.route("/study")
def study_select():
    return send_from_directory("templates", "study_select.html")

@app.route("/study/<int:n>")
def study(n):
    if n not in _VALID_STUDY_N:
        return "Invalid difficulty", 400
    return send_from_directory("templates", "study.html")

@app.route("/tiles/<path:filename>")
def tile_image(filename):
    return send_from_directory(_TILE_ASSETS, filename)


# ── random hand generators ────────────────────────────────────────────────────

@app.route("/api/random")
def api_random():
    return jsonify({"tiles": list(random_hand(13))})

@app.route("/api/random/one-suit")
def api_random_one_suit():
    return jsonify({"tiles": list(random_one_suit_hand())})

@app.route("/api/random/tenpai")
def api_random_tenpai():
    return jsonify({"tiles": list(random_two_suit_tenpai_hand())})

@app.route("/api/random/no-win-p/<int:n>")
def api_no_win_p_n(n):
    if n not in _VALID_STUDY_N:
        return jsonify({"error": f"n must be one of {sorted(_VALID_STUDY_N)}"}), 400
    return jsonify({"tiles": list(random_no_win_p_n(n))})

# Backward-compat alias
@app.route("/api/random/no-win-p14")
def api_no_win_p14():
    return jsonify({"tiles": list(random_no_win_p14())})


# ── analysis ──────────────────────────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    tiles = [Tile(t) for t in request.json["tiles"]]
    n = len(tiles)

    if n == 14:
        r = analyze_hand(tiles)
        if r.is_winner:
            return jsonify({
                "type": "win",
                "labels": [ht.label for ht in r.hand_types],
                "multiplier": r.multiplier,
            })
        return jsonify({"type": "not_win"})

    if n == 13:
        r = check_tenpai(tiles)
        if r.is_tenpai:
            return jsonify({
                "type": "tenpai",
                "multiplier_sum": r.multiplier_sum,
                "waits": [
                    {
                        "tile": str(w.tile),
                        "labels": [ht.label for ht in w.result.hand_types],
                        "multiplier": w.result.multiplier,
                        "remaining": w.remaining,
                    }
                    for w in r.waits
                ],
            })
        return jsonify({"type": "not_tenpai"})

    return jsonify({"type": "invalid", "count": n})


def _wait_to_dict(w, multiplier_scale: int = 1) -> dict:
    return {
        "tile": str(w.tile),
        "labels": [ht.label for ht in w.result.hand_types],
        "multiplier": w.result.multiplier * multiplier_scale,
        "remaining": w.remaining,
    }


def _discard_analysis(tiles: list[Tile]) -> dict:
    """
    For each tile in an N-tile hand, return tenpai info after discarding it.
    Also returns gang (暗杠) options for tiles with 4 copies.
    """
    from collections import Counter
    counts: Counter = Counter(str(t) for t in tiles)

    # ── discard options ───────────────────────────────────────────────────────
    cache: dict[str, dict] = {}
    results = []
    for i, tile in enumerate(tiles):
        key = str(tile)
        if key not in cache:
            remaining = [t for j, t in enumerate(tiles) if j != i]
            r = check_tenpai_any(remaining)
            cache[key] = {
                "tile": key,
                "multiplier_sum": r.multiplier_sum if r.is_tenpai else 0,
                "waits": [_wait_to_dict(w) for w in r.waits],
            }
        results.append(cache[key])

    # ── 暗杠 options: tiles with exactly 4 copies ──────────────────────────
    gangs = []
    for tile_str, cnt in sorted(counts.items()):
        if cnt != 4:
            continue
        # Remove all 4 copies; remaining N-4 tiles are checked for tenpai.
        # N-4 = 3k+1 for all valid N (5→1, 8→4, 11→7, 14→10), so
        # check_tenpai_any handles them correctly.
        remaining = [t for t in tiles if str(t) != tile_str]
        r = check_tenpai_any(remaining)
        # Tile X is no longer in the deck after kong — exclude it from waits.
        valid_waits = [w for w in r.waits if str(w.tile) != tile_str]
        adjusted_ms = sum(w.remaining * w.result.multiplier * 2
                          for w in valid_waits)
        gangs.append({
            "tile": tile_str,
            "multiplier_sum": adjusted_ms,
            "waits": [_wait_to_dict(w, multiplier_scale=2) for w in valid_waits],
        })

    return {"results": results, "gangs": gangs}


@app.route("/api/analyze14", methods=["POST"])
def api_analyze14():
    """Per-discard analysis for a 14-tile hand (kept for backward compat)."""
    tiles = [Tile(t) for t in request.json["tiles"]]
    if len(tiles) != 14:
        return jsonify({"error": "Need 14 tiles"}), 400
    return jsonify(_discard_analysis(tiles))


@app.route("/api/analyzeN", methods=["POST"])
def api_analyze_n():
    """Per-discard analysis for any N-tile hand (N ∈ {5, 8, 11, 14})."""
    tiles = [Tile(t) for t in request.json["tiles"]]
    if len(tiles) not in _VALID_STUDY_N:
        return jsonify({"error": f"tile count must be one of {sorted(_VALID_STUDY_N)}"}), 400
    return jsonify(_discard_analysis(tiles))


@app.route("/api/swap1", methods=["POST"])
def api_swap1():
    tiles = [Tile(t) for t in request.json["tiles"]]
    if len(tiles) != 13:
        return jsonify({"error": "Need 13 tiles"}), 400
    r = find_tenpai_improvements(tiles)
    if not r.can_improve:
        return jsonify({"swaps": []})
    return jsonify({
        "swaps": [
            {
                "discard": str(s.discard),
                "draw": str(s.draw),
                "waits": [str(w.tile) for w in s.tenpai.waits],
                "multiplier_sum": s.tenpai.multiplier_sum,
            }
            for s in r.swaps
        ]
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
