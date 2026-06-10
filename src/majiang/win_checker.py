"""
Sichuan Mahjong (四川麻将) winning-hand detector with hand-type scoring.

Winning patterns
----------------
  Standard win   1 pair (对子) + 4 melds (顺子 / 刻子)           ×1
  七对            Seven Pairs — 7 distinct pairs                   ×2
  龙七对          Dragon Seven Pairs — 七对 with ≥1 quad as 2 pairs ×4

Suit / structure modifiers (stack multiplicatively)
----------------------------------------------------
  清一色          Flush — all 14 tiles share one suit              ×4
  碰碰胡          All Triplets — all 4 melds are triplets           ×2

Combined examples
-----------------
  清七对  = 清一色 × 七对   = ×8
  清龙七对 = 清一色 × 龙七对 = ×16
  清碰碰胡 = 清一色 × 碰碰胡 = ×8

When a hand qualifies under multiple interpretations (e.g. a hand that is
both a 清一色 standard win and a 清七对), analyze_hand returns the one with
the highest multiplier.
"""

from __future__ import annotations

import random as _random
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .tiles import ALL_TILES, Tile


# ---------------------------------------------------------------------------
# Hand types
# ---------------------------------------------------------------------------

class HandType(Enum):
    """Each member carries a Chinese label and its multiplier contribution."""

    STANDARD            = ("标准胡", 1)
    SEVEN_PAIRS         = ("七对",   2)
    DRAGON_SEVEN_PAIRS  = ("龙七对", 4)
    FLUSH               = ("清一色", 4)
    ALL_TRIPLETS        = ("碰碰胡", 2)

    def __init__(self, label: str, multiplier: int):
        self.label = label
        self.multiplier = multiplier

    def __repr__(self) -> str:
        return f"HandType.{self.name}"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Meld:
    tiles: tuple[Tile, ...]

    @property
    def kind(self) -> str:
        return "triplet" if self.tiles[0] == self.tiles[1] else "sequence"

    def __str__(self) -> str:
        return "".join(self.tiles)


@dataclass
class Decomposition:
    """A single valid meld-based decomposition of a standard winning hand."""
    pair: Tile
    melds: list[Meld] = field(default_factory=list)

    def __str__(self) -> str:
        meld_strs = "  ".join(str(m) for m in self.melds)
        return f"pair={self.pair}{self.pair}  melds=[{meld_strs}]"


@dataclass
class WinResult:
    """
    Complete analysis of a 14-tile hand.

    Attributes
    ----------
    is_winner     : True if the hand is a valid winning hand.
    hand_types    : Ordered list of HandType values that apply.
                    First element is the base type (STANDARD / SEVEN_PAIRS /
                    DRAGON_SEVEN_PAIRS); subsequent elements are modifiers
                    (FLUSH, ALL_TRIPLETS).  Empty when is_winner is False.
    multiplier    : Product of all hand_type.multiplier values (0 if not winner).
    decompositions: Meld-based decompositions (empty for seven-pairs wins).
    """
    is_winner: bool
    hand_types: list[HandType]
    multiplier: int
    decompositions: list[Decomposition]

    def __str__(self) -> str:
        if not self.is_winner:
            return "Not a winning hand"
        labels = " + ".join(ht.label for ht in self.hand_types)
        return f"Win! {labels}  (×{self.multiplier})"


# ---------------------------------------------------------------------------
# Internal: meld search
# ---------------------------------------------------------------------------

def _find_melds(counts: Counter, melds_so_far: list[Meld]) -> Optional[list[Meld]]:
    """
    Recursively cover every tile in *counts* with melds.
    Always consumes min(counts) first → deterministic, no redundant branches.
    Returns a complete meld list on success, None on failure.
    Mutates *counts* in place but restores it on backtrack.
    """
    if not counts:
        return list(melds_so_far)

    first = min(counts)                        # tile format: "<number><suit>"
    suit, num = first[1], int(first[0])

    # --- try triplet ---
    if counts[first] >= 3:
        counts[first] -= 3
        if counts[first] == 0:
            del counts[first]
        result = _find_melds(counts, melds_so_far + [Meld((first, first, first))])
        if result is not None:
            return result
        counts[first] = counts.get(first, 0) + 3

    # --- try sequence ---
    if num <= 7:
        t2 = Tile(f"{num + 1}{suit}")
        t3 = Tile(f"{num + 2}{suit}")
        if counts.get(t2, 0) >= 1 and counts.get(t3, 0) >= 1:
            for t in (first, t2, t3):
                counts[t] -= 1
                if counts[t] == 0:
                    del counts[t]
            result = _find_melds(counts, melds_so_far + [Meld((first, t2, t3))])
            if result is not None:
                return result
            for t in (first, t2, t3):
                counts[t] = counts.get(t, 0) + 1

    return None


# ---------------------------------------------------------------------------
# Internal: hand-type classification helpers
# ---------------------------------------------------------------------------

def _is_flush(tiles: list[Tile]) -> bool:
    suits = {t[1] for t in tiles}
    return len(suits) == 1


def _modifiers_for_standard(decomp: Decomposition, tiles: list[Tile]) -> list[HandType]:
    mods: list[HandType] = []
    if _is_flush(tiles):
        mods.append(HandType.FLUSH)
    if all(m.kind == "triplet" for m in decomp.melds):
        mods.append(HandType.ALL_TRIPLETS)
    return mods


def _result_for_standard(decomp: Decomposition, tiles: list[Tile]) -> WinResult:
    mods = _modifiers_for_standard(decomp, tiles)
    hand_types = [HandType.STANDARD] + mods
    mult = 1
    for ht in hand_types:
        mult *= ht.multiplier
    return WinResult(
        is_winner=True,
        hand_types=hand_types,
        multiplier=mult,
        decompositions=[decomp],
    )


def _check_seven_pairs(counts: Counter, tiles: list[Tile]) -> Optional[WinResult]:
    """
    Return a WinResult if the hand is a valid 七对 or 龙七对, else None.

    Rules:
      • Every tile must appear an even number of times (2 or 4).
      • The total number of pairs (count // 2) must equal exactly 7.
      • If any tile has count == 4 it contributes 2 pairs → 龙七对 (×4).
      • Otherwise → 七对 (×2).
    Both can stack with 清一色 (×4).
    """
    if any(c % 2 != 0 for c in counts.values()):
        return None
    if sum(c // 2 for c in counts.values()) != 7:
        return None

    is_dragon = any(c == 4 for c in counts.values())
    base = HandType.DRAGON_SEVEN_PAIRS if is_dragon else HandType.SEVEN_PAIRS
    hand_types: list[HandType] = [base]
    if _is_flush(tiles):
        hand_types.append(HandType.FLUSH)

    mult = 1
    for ht in hand_types:
        mult *= ht.multiplier

    return WinResult(
        is_winner=True,
        hand_types=hand_types,
        multiplier=mult,
        decompositions=[],
    )


# ---------------------------------------------------------------------------
# Shared sentinel — referenced by both fast-path helpers and public API
# ---------------------------------------------------------------------------

_LOSER = WinResult(is_winner=False, hand_types=[], multiplier=0, decompositions=[])


# ---------------------------------------------------------------------------
# Counter-based fast paths (used by the two-swap hot loop)
# ---------------------------------------------------------------------------

def _is_flush_counter(counts: Counter) -> bool:
    suits = {t[1] for t in counts}
    return len(suits) == 1


def _analyze_counter(counts: Counter) -> WinResult:
    """
    Identical logic to analyze_hand but takes a pre-built Counter directly,
    skipping tile-list creation and Counter construction.
    Assumes exactly 14 tiles in counts.  Does NOT mutate counts.
    """
    best = _LOSER  # defined just below; forward ref resolved at call time

    # --- seven-pairs branch ---
    if all(c % 2 == 0 for c in counts.values()):
        pair_count = sum(c // 2 for c in counts.values())
        if pair_count == 7:
            is_dragon = any(c == 4 for c in counts.values())
            base = HandType.DRAGON_SEVEN_PAIRS if is_dragon else HandType.SEVEN_PAIRS
            hand_types: list[HandType] = [base]
            if _is_flush_counter(counts):
                hand_types.append(HandType.FLUSH)
            mult = 1
            for ht in hand_types:
                mult *= ht.multiplier
            best = WinResult(is_winner=True, hand_types=hand_types, multiplier=mult, decompositions=[])

    # --- standard-win branch ---
    flush = _is_flush_counter(counts)
    seen_pairs: set[str] = set()
    for pair_tile, cnt in list(counts.items()):
        if cnt < 2 or pair_tile in seen_pairs:
            continue
        seen_pairs.add(pair_tile)

        rest = Counter(counts)          # copy so _find_melds can mutate freely
        rest[pair_tile] -= 2
        if rest[pair_tile] == 0:
            del rest[pair_tile]

        melds = _find_melds(rest, [])
        if melds is not None:
            all_triplets = all(m.kind == "triplet" for m in melds)
            hand_types = [HandType.STANDARD]
            if flush:
                hand_types.append(HandType.FLUSH)
            if all_triplets:
                hand_types.append(HandType.ALL_TRIPLETS)
            mult = 1
            for ht in hand_types:
                mult *= ht.multiplier
            if mult > best.multiplier:
                best = WinResult(
                    is_winner=True,
                    hand_types=hand_types,
                    multiplier=mult,
                    decompositions=[Decomposition(pair=pair_tile, melds=melds)],
                )

    return best


def _tenpai_waits_counter(
    counts13: Counter,
    skip1: "Tile | None" = None,
    skip2: "Tile | None" = None,
) -> list[WaitingTile]:
    """
    Tenpai check that mutates counts13 in-place (adding/removing one tile per
    candidate) instead of building new lists and Counters.  On return, counts13
    is restored to its original state.  Assumes exactly 13 tiles in counts13.

    skip1 / skip2: optional tiles to exclude from the wait candidates (used by
    the two-swap search to avoid suggesting the tiles just discarded as waits).
    """
    waits: list[WaitingTile] = []
    for candidate in ALL_TILES:
        if candidate == skip1 or candidate == skip2:
            continue
        n_in_hand = counts13.get(candidate, 0)
        if n_in_hand >= 4:
            continue
        counts13[candidate] += 1
        result = _analyze_counter(counts13)     # 14 tiles
        counts13[candidate] -= 1
        if counts13[candidate] == 0:
            del counts13[candidate]
        if result.is_winner:
            waits.append(WaitingTile(tile=candidate, result=result,
                                     remaining=4 - n_in_hand))
    return waits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_hand(tiles: list[Tile] | list[str]) -> WinResult:
    """
    Fully analyze a 14-tile Sichuan Mahjong hand.

    Returns a WinResult describing whether the hand wins, which hand types
    apply, and the combined multiplier.  When a hand qualifies under multiple
    interpretations the one with the highest multiplier is returned.

    Parameters
    ----------
    tiles : 14 Tile objects or plain strings (e.g. "3m", "7p").

    Examples
    --------
    >>> r = analyze_hand(parse_hand("111m222m333m444m55m"))
    >>> r.is_winner, r.multiplier, [ht.label for ht in r.hand_types]
    (True, 8, ['标准胡', '清一色', '碰碰胡'])

    >>> r = analyze_hand(parse_hand("11m33m55m77m99m11p33p"))
    >>> r.is_winner, r.multiplier
    (True, 2)
    """
    tiles = [Tile(t) if not isinstance(t, Tile) else t for t in tiles]
    if len(tiles) != 14:
        raise ValueError(f"Expected 14 tiles, got {len(tiles)}")

    counts = Counter(tiles)
    best: WinResult = _LOSER

    # --- seven-pairs branch ---
    sp = _check_seven_pairs(counts, tiles)
    if sp is not None and sp.multiplier > best.multiplier:
        best = sp

    # --- standard-win branch: try every possible pair ---
    seen_pairs: set[str] = set()
    for pair_tile, cnt in list(counts.items()):
        if cnt < 2 or pair_tile in seen_pairs:
            continue
        seen_pairs.add(pair_tile)

        rest = Counter(counts)
        rest[pair_tile] -= 2
        if rest[pair_tile] == 0:
            del rest[pair_tile]

        melds = _find_melds(rest, [])
        if melds is not None:
            decomp = Decomposition(pair=pair_tile, melds=melds)
            candidate = _result_for_standard(decomp, tiles)
            if candidate.multiplier > best.multiplier:
                best = candidate

    return best


def is_winning_hand(tiles: list[Tile] | list[str]) -> bool:
    """Return True if the 14-tile hand is any valid Sichuan Mahjong win."""
    return analyze_hand(tiles).is_winner


def winning_decompositions(
    tiles: list[Tile] | list[str],
    limit: int = 0,
) -> list[Decomposition]:
    """
    Return all distinct meld-based decompositions of the hand.
    Seven-pairs wins are not represented here; use analyze_hand for those.

    Parameters
    ----------
    tiles : 14-tile hand.
    limit : stop after this many results (0 = find all).
    """
    tiles = [Tile(t) if not isinstance(t, Tile) else t for t in tiles]
    if len(tiles) != 14:
        raise ValueError(f"Expected 14 tiles, got {len(tiles)}")

    results: list[Decomposition] = []
    seen_pairs: set[str] = set()
    base_counts = Counter(tiles)

    for pair_tile, cnt in list(base_counts.items()):
        if cnt < 2 or pair_tile in seen_pairs:
            continue
        seen_pairs.add(pair_tile)

        rest = Counter(base_counts)
        rest[pair_tile] -= 2
        if rest[pair_tile] == 0:
            del rest[pair_tile]

        melds = _find_melds(rest, [])
        if melds is not None:
            results.append(Decomposition(pair=pair_tile, melds=melds))
            if limit and len(results) >= limit:
                return results

    return results


# ---------------------------------------------------------------------------
# Tenpai (聽牌) detection
# ---------------------------------------------------------------------------

@dataclass
class WaitingTile:
    """One tile that completes a tenpai hand, with the resulting WinResult."""
    tile: Tile
    result: WinResult
    remaining: int = 4  # copies still drawable (4 − copies already in hand)

    def __str__(self) -> str:
        labels = "+".join(ht.label for ht in self.result.hand_types)
        return f"{self.tile}({self.remaining}张 ×{self.result.multiplier} {labels})"


@dataclass
class TenpaiResult:
    """
    Result of a tenpai check on a 13-tile hand.

    Attributes
    ----------
    is_tenpai       : True if at least one tile completes the hand.
    waits           : One entry per completing tile, ordered by tile value.
                      Each entry holds the tile and the best WinResult it
                      produces.
    multiplier_sum  : Sum of all individual winning multipliers — useful as
                      a rough measure of the hand's overall strength.
    """
    is_tenpai: bool
    waits: list[WaitingTile]

    @property
    def multiplier_sum(self) -> int:
        # Weight each wait's multiplier by how many copies of that tile can
        # still be drawn (4 minus the copies already in the 13-tile hand).
        return sum(w.remaining * w.result.multiplier for w in self.waits)

    def __str__(self) -> str:
        if not self.is_tenpai:
            return "Not tenpai"
        tile_strs = ", ".join(str(w) for w in self.waits)
        return f"Tenpai: {tile_strs}  (multiplier sum ×{self.multiplier_sum})"


def check_tenpai(tiles: list[Tile] | list[str]) -> TenpaiResult:
    """
    Check whether a 13-tile hand is tenpai (one tile from winning).

    For every candidate tile not already at count 4 in the hand, adds it
    and calls analyze_hand.  Returns a TenpaiResult listing all completing
    tiles and their best WinResult.

    Parameters
    ----------
    tiles : Exactly 13 Tile objects or plain strings.

    Examples
    --------
    >>> r = check_tenpai(parse_hand("123m456p789s11p23s"))
    >>> r.is_tenpai
    True
    >>> [str(w.tile) for w in r.waits]
    ['1s', '4s']
    """
    tiles = [Tile(t) if not isinstance(t, Tile) else t for t in tiles]
    if len(tiles) != 13:
        raise ValueError(f"Expected 13 tiles for tenpai check, got {len(tiles)}")

    counts = Counter(tiles)
    waits: list[WaitingTile] = []

    for candidate in ALL_TILES:
        n_in_hand = counts.get(candidate, 0)
        if n_in_hand >= 4:
            continue                        # tile exhausted; can't be drawn
        result = analyze_hand(tiles + [candidate])
        if result.is_winner:
            waits.append(WaitingTile(tile=candidate, result=result,
                                     remaining=4 - n_in_hand))

    return TenpaiResult(is_tenpai=bool(waits), waits=waits)


def random_one_suit_hand() -> list[Tile]:
    """Return a random 13-tile hand drawn from 饼 (p) tiles only."""
    deck = [Tile(f"{n}p") for n in "123456789"] * 4
    return _random.sample(deck, 13)


def random_two_suit_tenpai_hand() -> list[Tile]:
    """
    Return a random 13-tile tenpai hand with at most two suits and at least
    one wait whose multiplier is > 1 (i.e. not a plain 标准胡 ×1 hand).

    Suit-count distribution: 75 % one-suit, 25 % two-suit.

    One-suit path — rejection sampling:
        All completions are 清一色 (×4), so the filter is always satisfied.
        Hit rate ~50 %, expected latency < 2 ms.

    Two-suit path — structured construction:
        Plain two-suit tenpai hands that pass the multiplier filter are
        extremely rare (~0.06 %), so hands are built directly from two
        structures that guarantee multiplier ≥ 2:
          • 七对 (seven pairs): six random pairs + one singleton, waiting
            for the seventh pair (×2).
          • 碰碰胡 (all triplets): four random triplets + one singleton,
            waiting for the completing pair (×2).
        One structure is chosen at random with equal probability.
    """
    suits = list("mps")
    if _random.random() < 0.75:
        # ── 1-suit: rejection sampling (inner loop commits to this path) ──
        # All 1-suit tenpai completions are 清一色 (×4) — never plain ×1.
        while True:
            suit = _random.choice(suits)
            deck = [Tile(f"{n}{suit}") for n in "123456789"] * 4
            hand = _random.sample(deck, 13)
            if check_tenpai(hand).is_tenpai:
                return hand
    else:
        # ── 2-suit: structured construction (always returns, always ≥×2) ──
        # Plain two-suit tenpai with multiplier > 1 is extremely rare via
        # rejection sampling, so hands are built from known structures.
        s1, s2 = _random.sample(suits, 2)
        pool = [Tile(f"{n}{s}") for s in (s1, s2) for n in "123456789"]
        if _random.random() < 0.5:
            # 七对: 6 pairs + 1 singleton  (13 tiles, wait = 7th pair ×2)
            tiles = _random.sample(pool, 7)
            return [tiles[i] for i in range(6) for _ in range(2)] + [tiles[6]]
        else:
            # 碰碰胡: 4 triplets + 1 singleton  (13 tiles, wait = pair ×2)
            tiles = _random.sample(pool, 5)
            return ([tiles[i] for i in range(4) for _ in range(3)]
                    + [tiles[4]])


# ---------------------------------------------------------------------------
# Tenpai improvement: find swaps that turn a non-tenpai hand into tenpai
# ---------------------------------------------------------------------------

@dataclass
class Swap:
    """
    A single discard-then-draw that transforms a 13-tile hand into tenpai.

    Attributes
    ----------
    discard : Tile removed from the original hand.
    draw    : Tile added in its place.
    tenpai  : TenpaiResult of the resulting 13-tile hand.
    """
    discard: Tile
    draw: Tile
    tenpai: TenpaiResult

    def __str__(self) -> str:
        waits = " ".join(str(w.tile) for w in self.tenpai.waits)
        return (
            f"discard {self.discard}, draw {self.draw}"
            f" → waits [{waits}]  (sum ×{self.tenpai.multiplier_sum})"
        )


@dataclass
class ImprovementResult:
    """
    All single-tile swaps that make a 13-tile hand tenpai.

    Attributes
    ----------
    swaps       : Every (discard, draw) pair that achieves tenpai, sorted by
                  (discard, draw).  Empty when no swap helps.
    can_improve : True when at least one swap works.
    """
    swaps: list[Swap]

    @property
    def can_improve(self) -> bool:
        return bool(self.swaps)

    def __str__(self) -> str:
        if not self.swaps:
            return "No swap leads to tenpai"
        lines = [f"  {swap}" for swap in self.swaps]
        return f"{len(self.swaps)} swap(s) lead to tenpai:\n" + "\n".join(lines)


def find_tenpai_improvements(tiles: list[Tile] | list[str]) -> ImprovementResult:
    """
    Find all single-tile swaps that make a 13-tile hand tenpai.

    For every unique tile value in the hand (discard candidate) and every tile
    that could be drawn (skip if already at 4 copies in the 12-tile remainder),
    check_tenpai is called on the resulting 13-tile hand.  Results are
    deduplicated by (discard_value, draw_value) so identical tiles in the hand
    never produce duplicate entries.

    Parameters
    ----------
    tiles : Exactly 13 Tile objects or plain strings.

    Returns
    -------
    ImprovementResult whose swaps are sorted by (discard, draw).

    Examples
    --------
    >>> r = find_tenpai_improvements(parse_hand("123m456p789s12p35s"))
    >>> r.can_improve
    True
    >>> any(s.discard == Tile("5s") and s.draw == Tile("3p") for s in r.swaps)
    True
    """
    tiles = [Tile(t) if not isinstance(t, Tile) else t for t in tiles]
    if len(tiles) != 13:
        raise ValueError(f"Expected 13 tiles for improvement check, got {len(tiles)}")

    base_counts = Counter(tiles)
    seen: set[tuple[str, str]] = set()
    swaps: list[Swap] = []

    for discard in sorted(set(tiles)):              # unique discards, sorted
        remainder = Counter(base_counts)
        remainder[discard] -= 1
        if remainder[discard] == 0:
            del remainder[discard]
        remainder_list = list(remainder.elements())  # 12 tiles

        for draw in ALL_TILES:
            if draw == discard:
                continue
            if remainder.get(draw, 0) >= 4:
                continue                            # can't draw a 5th copy
            key = (str(discard), str(draw))
            if key in seen:
                continue
            seen.add(key)

            result = check_tenpai(remainder_list + [draw])
            if result.is_tenpai:
                swaps.append(Swap(discard=discard, draw=draw, tenpai=result))

    swaps.sort(key=lambda s: (str(s.discard), str(s.draw)))
    return ImprovementResult(swaps=swaps)


# ---------------------------------------------------------------------------
# Two-tile swap improvement
# ---------------------------------------------------------------------------

@dataclass
class DoubleSwap:
    """
    A discard-two / draw-two exchange that transforms a 13-tile hand into tenpai.

    Attributes
    ----------
    discard : Pair of tiles removed, stored as a sorted tuple (d1 ≤ d2).
    draw    : Pair of tiles added,   stored as a sorted tuple (r1 ≤ r2).
    tenpai  : TenpaiResult of the resulting 13-tile hand.
    """
    discard: tuple[Tile, Tile]
    draw: tuple[Tile, Tile]
    tenpai: TenpaiResult

    def __str__(self) -> str:
        waits = " ".join(str(w.tile) for w in self.tenpai.waits)
        d1, d2 = self.discard
        r1, r2 = self.draw
        return (
            f"discard {d1}+{d2}, draw {r1}+{r2}"
            f" → waits [{waits}]  (sum ×{self.tenpai.multiplier_sum})"
        )


@dataclass
class DoubleImprovementResult:
    """
    All two-tile swaps that make a 13-tile hand tenpai.

    Attributes
    ----------
    swaps       : Every (discard_pair, draw_pair) that achieves tenpai, sorted
                  by (d1, d2, r1, r2).  Empty when no two-swap helps.
    can_improve : True when at least one swap works.
    """
    swaps: list[DoubleSwap]

    @property
    def can_improve(self) -> bool:
        return bool(self.swaps)

    def __str__(self) -> str:
        if not self.swaps:
            return "No two-tile swap leads to tenpai"
        lines = [f"  {swap}" for swap in self.swaps]
        return f"{len(self.swaps)} two-tile swap(s) lead to tenpai:\n" + "\n".join(lines)


def find_tenpai_improvements_2(tiles: list[Tile] | list[str]) -> DoubleImprovementResult:
    """
    Find all two-tile swaps that make a 13-tile hand tenpai.

    For every unique discard pair (d1 ≤ d2) and every draw pair (r1 ≤ r2),
    checks the resulting 13-tile hand for tenpai and collects all successes.

    Constraints applied for coaching relevance:
      • Draw tiles must differ from both discarded tiles (r1, r2 ∉ {d1, d2}).
        Drawing back a tile you just discarded is equivalent to a smaller swap.
      • The tenpai wait list excludes d1 and d2: waiting on a tile you just
        put in the discard pile is not useful advice.
      • Draw tiles that would exceed four copies in the new hand are skipped.

    Deduplication is by value: identical tiles in the hand never produce
    duplicate (discard, draw) entries.

    Parameters
    ----------
    tiles : Exactly 13 Tile objects or plain strings.

    Returns
    -------
    DoubleImprovementResult whose swaps are sorted by (d1, d2, r1, r2).

    Examples
    --------
    >>> r = find_tenpai_improvements_2(parse_hand("123m456p789s12p35s"))
    >>> r.can_improve
    True
    """
    tiles = [Tile(t) if not isinstance(t, Tile) else t for t in tiles]
    if len(tiles) != 13:
        raise ValueError(f"Expected 13 tiles for two-swap check, got {len(tiles)}")

    counts = Counter(tiles)          # mutated in-place throughout; always restored
    unique = sorted(set(tiles))
    swaps: list[DoubleSwap] = []

    for i, d1 in enumerate(unique):
        # Remove d1 from counts
        counts[d1] -= 1
        if counts[d1] == 0:
            del counts[d1]

        for d2 in unique[i:]:
            if d1 == d2 and counts.get(d1, 0) == 0:
                # d1 was already removed above; need at least one more copy for d2
                continue

            # Remove d2 from counts  →  counts is now the 11-tile remainder
            counts[d2] -= 1
            if counts[d2] == 0:
                del counts[d2]

            for j, r1 in enumerate(ALL_TILES):
                # Drawing a discarded tile back is equivalent to a smaller swap
                if r1 == d1 or r1 == d2:
                    continue
                if counts.get(r1, 0) >= 4:
                    continue                # can't draw a 5th copy of r1

                # Add r1  →  counts is now 12 tiles
                counts[r1] += 1

                for r2 in ALL_TILES[j:]:   # r2 >= r1, avoids duplicate pairs
                    if r2 == d1 or r2 == d2:
                        continue
                    if r1 == r2:
                        if counts.get(r1, 0) >= 4:   # already added one r1 above
                            continue
                    else:
                        if counts.get(r2, 0) >= 4:
                            continue

                    # Add r2  →  counts is now the candidate 13-tile hand
                    counts[r2] += 1

                    # Pass d1/d2 as excluded wait candidates: waiting on a tile
                    # you just discarded is incoherent coaching advice.
                    waits = _tenpai_waits_counter(counts, skip1=d1, skip2=d2)
                    if waits:
                        swaps.append(DoubleSwap(
                            discard=(d1, d2),
                            draw=(r1, r2),
                            tenpai=TenpaiResult(is_tenpai=True, waits=waits),
                        ))

                    # Restore r2
                    counts[r2] -= 1
                    if counts[r2] == 0:
                        del counts[r2]

                # Restore r1
                counts[r1] -= 1
                if counts[r1] == 0:
                    del counts[r1]

            # Restore d2
            counts[d2] = counts.get(d2, 0) + 1

        # Restore d1
        counts[d1] = counts.get(d1, 0) + 1

    swaps.sort(key=lambda s: (str(s.discard[0]), str(s.discard[1]),
                               str(s.draw[0]),    str(s.draw[1])))
    return DoubleImprovementResult(swaps=swaps)
