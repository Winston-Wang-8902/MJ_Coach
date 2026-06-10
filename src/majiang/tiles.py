"""
Tile representation and parsing for Sichuan Mahjong (四川麻将).

Tile notation: "<number><suit>"
  Suits:   m = 万 (characters), p = 饼 (circles), s = 条 (bamboo)
  Numbers: 1–9

Compact hand notation: numbers run before their suit letter, e.g.
  "123m456p789s11p"  ->  1m 2m 3m 4p 5p 6p 7s 8s 9s 1p 1p
"""

from __future__ import annotations
import random

SUITS = frozenset("mps")
SUIT_NAMES = {"m": "万", "p": "饼", "s": "条"}


class Tile(str):
    """
    Immutable string subclass representing a single tile, e.g. "3m", "9s".
    Inherits all str behaviour so it works directly as a dict / Counter key.
    """

    __slots__ = ()

    def __new__(cls, value: str) -> "Tile":
        value = value.strip().lower()
        if len(value) != 2 or value[0] not in "123456789" or value[1] not in SUITS:
            raise ValueError(
                f"Invalid tile {value!r}. Expected <number><suit>, e.g. '3m', '7p', '9s'."
            )
        return super().__new__(cls, value)

    @property
    def number(self) -> int:
        return int(self[0])

    @property
    def suit(self) -> str:
        return self[1]

    def __repr__(self) -> str:
        return f"Tile({str(self)!r})"

    def chinese(self) -> str:
        return f"{self.number}{SUIT_NAMES[self.suit]}"


ALL_TILES: list["Tile"] = [Tile(f"{n}{s}") for s in "mps" for n in "123456789"]
"""All 27 distinct tile types in Sichuan Mahjong, in suit-then-number order."""

_DECK: list["Tile"] = ALL_TILES * 4   # full 108-tile deck (4 copies of each type)


def random_hand(n: int = 13) -> list["Tile"]:
    """Return n tiles drawn at random (without replacement) from a full 108-tile deck."""
    if not 1 <= n <= 14:
        raise ValueError(f"n must be between 1 and 14, got {n}")
    return random.sample(_DECK, n)


def parse_hand(notation: str) -> list[Tile]:
    """
    Parse compact mahjong hand notation into a list of Tiles.

    Examples
    --------
    >>> parse_hand("123m456p789s11p22s")
    [Tile('1m'), Tile('2m'), Tile('3m'), Tile('4p'), Tile('5p'), Tile('6p'),
     Tile('7s'), Tile('8s'), Tile('9s'), Tile('1p'), Tile('1p'), Tile('2s'), Tile('2s')]

    >>> parse_hand("1m 2m 3m 4p 5p 6p 7s 8s 9s 1p 1p 2s 2s 3s")  # space-separated
    [...]
    """
    notation = notation.replace(" ", "").lower()
    tiles: list[Tile] = []
    pending_digits: list[str] = []

    for ch in notation:
        if ch.isdigit():
            pending_digits.append(ch)
        elif ch in SUITS:
            if not pending_digits:
                raise ValueError(f"Suit '{ch}' with no preceding number in {notation!r}")
            for d in pending_digits:
                tiles.append(Tile(d + ch))
            pending_digits.clear()
        else:
            raise ValueError(f"Unexpected character {ch!r} in {notation!r}")

    if pending_digits:
        raise ValueError(f"Trailing digits with no suit in {notation!r}")

    return tiles
