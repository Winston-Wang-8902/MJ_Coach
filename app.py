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
    Tile, random_hand, analyze_hand, check_tenpai,
    random_no_win_p14, random_one_suit_hand, random_two_suit_tenpai_hand,
    find_tenpai_improvements,
)

app = Flask(__name__, static_folder="static", template_folder="templates")

_TILE_ASSETS = os.path.join(os.path.dirname(__file__), "src", "majiang", "tile_assets")


# ── pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def hub():
    return send_from_directory("templates", "hub.html")

@app.route("/coach")
def coach():
    return send_from_directory("templates", "index.html")

@app.route("/study")
def study():
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


@app.route("/api/analyze14", methods=["POST"])
def api_analyze14():
    """For each tile in a 14-tile hand, return the multiplier_sum after discarding it."""
    tiles = [Tile(t) for t in request.json["tiles"]]
    if len(tiles) != 14:
        return jsonify({"error": "Need 14 tiles"}), 400

    # Cache results by tile value to avoid redundant tenpai checks
    cache: dict[str, int] = {}
    results = []
    for i, tile in enumerate(tiles):
        key = str(tile)
        if key not in cache:
            remaining = [t for j, t in enumerate(tiles) if j != i]
            r = check_tenpai(remaining)
            cache[key] = r.multiplier_sum if r.is_tenpai else 0
        results.append({"tile": key, "multiplier_sum": cache[key]})

    return jsonify({"results": results})


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
