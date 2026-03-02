"""
Frame wrapper around arcengine.FrameData with grid inspection helpers.
"""

from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
from arcengine import FrameData, GameAction, GameState

__all__ = ["DiffRegion", "Frame"]


@dataclass(slots=True)
class DiffRegion:
    """
    A contiguous region of changed cells between two frames.

    Attributes:
        x0, y0, x1, y1: Bounding box (inclusive start, exclusive end).
        changes: [(x, y, old_val, new_val), ...] within this region.
    """

    x0: int
    y0: int
    x1: int
    y1: int
    changes: list[tuple[int, int, int, int]]

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def count(self) -> int:
        return len(self.changes)

    def __repr__(self) -> str:
        return (
            f"DiffRegion(x=[{self.x0},{self.x1}) y=[{self.y0},{self.y1}), "
            f"{self.count} changes)"
        )


def _cluster_changes(
    changes: list[tuple[int, int, int, int]],
    margin: int = 2,
) -> list[DiffRegion]:
    """
    Group changes into contiguous regions.

    Two changes belong to the same region if their bounding boxes
    (expanded by *margin* pixels) overlap.
    """
    if not changes:
        return []

    # Sort by (y, x) for spatial locality
    sorted_changes = sorted(changes, key=lambda c: (c[1], c[0]))

    regions: list[list[tuple[int, int, int, int]]] = []
    boxes: list[list[int]] = []  # [min_x, min_y, max_x, max_y] per region

    for change in sorted_changes:
        x, y = change[0], change[1]
        merged = False
        for i, box in enumerate(boxes):
            if (
                x >= box[0] - margin
                and x <= box[2] + margin
                and y >= box[1] - margin
                and y <= box[3] + margin
            ):
                regions[i].append(change)
                box[0] = min(box[0], x)
                box[1] = min(box[1], y)
                box[2] = max(box[2], x)
                box[3] = max(box[3], y)
                merged = True
                break
        if not merged:
            regions.append([change])
            boxes.append([x, y, x, y])

    # Merge any regions whose expanded boxes now overlap
    merged_any = True
    while merged_any:
        merged_any = False
        i = 0
        while i < len(boxes):
            j = i + 1
            while j < len(boxes):
                bi, bj = boxes[i], boxes[j]
                if (
                    bi[0] - margin <= bj[2] + margin
                    and bi[2] + margin >= bj[0] - margin
                    and bi[1] - margin <= bj[3] + margin
                    and bi[3] + margin >= bj[1] - margin
                ):
                    # Merge j into i
                    regions[i].extend(regions[j])
                    bi[0] = min(bi[0], bj[0])
                    bi[1] = min(bi[1], bj[1])
                    bi[2] = max(bi[2], bj[2])
                    bi[3] = max(bi[3], bj[3])
                    regions.pop(j)
                    boxes.pop(j)
                    merged_any = True
                else:
                    j += 1
            i += 1

    return [
        DiffRegion(x0=box[0], y0=box[1], x1=box[2] + 1, y1=box[3] + 1, changes=region)
        for region, box in zip(regions, boxes)
    ]


class Frame:
    """
    Wrapper around FrameData with grid inspection helpers.

    Attributes:
        grid: Immutable 2D tuple of ints — the current level's grid.
        grid_np: Read-only numpy int8 array view of the grid (cached on first access).
        winning_frame: A full Frame of the just-completed level when a level
            transition occurred on this action, otherwise None. Has all the
            same helpers (render, diff, find, etc.) so you can inspect what
            the winning state looked like.
        state: Current GameState (NOT_FINISHED, WIN, GAME_OVER).
        levels_completed: Levels beaten so far.
        win_levels: Total levels required to win the game.
        game_id: Game identifier string.
        available_actions: Action name strings that can be passed to submit_action.
        width, height: Grid dimensions.
    """

    grid: tuple[tuple[int, ...], ...]
    winning_frame: "Frame | None"
    state: GameState
    levels_completed: int
    win_levels: int
    game_id: str

    _data: FrameData
    _grid_array: np.ndarray | None
    _frozen: bool

    __slots__ = (
        "grid",
        "winning_frame",
        "state",
        "levels_completed",
        "win_levels",
        "game_id",
        "_data",
        "_grid_array",
        "_frozen",
    )

    def __init__(
        self, data: FrameData, *, prev_levels_completed: int | None = None
    ) -> None:
        object.__setattr__(self, "_frozen", False)
        self._data = data
        raw_grid = data.frame[-1]
        self.grid = tuple(tuple(row) for row in raw_grid)
        self._grid_array: np.ndarray | None = None
        self.state = data.state
        self.levels_completed = data.levels_completed
        self.win_levels = data.win_levels
        self.game_id = data.game_id
        level_transition = (
            prev_levels_completed is not None
            and data.levels_completed > prev_levels_completed
            and len(data.frame) > 1
        )
        if level_transition:
            win: Frame = object.__new__(Frame)
            object.__setattr__(win, "_frozen", False)
            win._data = data
            win_grid = data.frame[0]
            win.grid = tuple(tuple(row) for row in win_grid)
            win._grid_array = None
            win.winning_frame = None
            win.state = data.state
            win.levels_completed = prev_levels_completed  # type: ignore[assignment]
            win.win_levels = data.win_levels
            win.game_id = data.game_id
            object.__setattr__(win, "_frozen", True)
            self.winning_frame = win
        else:
            self.winning_frame = None
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: object) -> None:
        if self._frozen:
            raise AttributeError(f"Frame is immutable, cannot set '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"Frame is immutable, cannot delete '{name}'")

    @property
    def grid_np(self) -> np.ndarray:
        """Cached numpy view of the grid (int8, read-only)."""
        if self._grid_array is None:
            arr = np.array(self.grid, dtype=np.int8)
            arr.flags.writeable = False
            object.__setattr__(self, "_grid_array", arr)
        assert self._grid_array is not None
        return self._grid_array

    @property
    def width(self) -> int:
        """Number of columns in the grid."""
        return len(self.grid[0]) if self.grid else 0

    @property
    def height(self) -> int:
        """Number of rows in the grid."""
        return len(self.grid)

    @property
    def available_actions(self) -> list[str]:
        """
        Action names (e.g. 'ACTION1') that can be passed to submit_action.
        """
        actions = [GameAction.from_id(a).name for a in self._data.available_actions]
        if "RESET" not in actions:
            actions.append("RESET")
        return actions

    def render(
        self,
        keys: str = "0123456789abcdef",
        gap: str = " ",
        y_ticks: bool = False,
        x_ticks: bool = False,
        crop: tuple[int, int, int, int] | None = None,
    ) -> str:
        """
        Render the grid as a text string.

        Args:
            keys: 16-char string mapping each int 0-15 to a display character.
            gap: Separator between cells horizontally; default is " " (no gap might reduce legibility).
            y_ticks: Prefix each row with its y coordinate.
            x_ticks: Add column-number header lines.
            crop: (x1, y1, x2, y2) sub-region with exclusive end. None = full grid.
        """
        x1, y1, x2, y2 = crop or (0, 0, self.width, self.height)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.width, x2), min(self.height, y2)

        lines: list[str] = []
        pad = "       " if y_ticks else ""

        if x_ticks:
            lines.append(pad + gap.join(str(c // 10) for c in range(x1, x2)))
            lines.append(pad + gap.join(str(c % 10) for c in range(x1, x2)))
            data_width = len(gap.join("x" for _ in range(x1, x2)))
            lines.append(pad + "-" * data_width)

        for y in range(y1, y2):
            row = gap.join(keys[self.grid[y][x]] for x in range(x1, x2))
            if y_ticks:
                lines.append(f"y={y:>2d} | {row}")
            else:
                lines.append(row)

        return "\n".join(lines)

    def diff(self, other: Self, margin: int = 2) -> list[DiffRegion]:
        """
        Cells that changed between ``old_frame`` (``other``) and ``new_frame`` (``self``).

        Call as ``new_frame.diff(old_frame)``.  Returns a list of DiffRegions,
        each with a bounding box and individual cell changes
        ``(x, y, old_val, new_val)``.  Changes within *margin* pixels of each
        other are merged into the same region.

        Tip: use region bounds to zoom in:
        ``new_frame.render_diff(old_frame, crop=(r.x0, r.y0, r.x1, r.y1))``.
        """
        changes: list[tuple[int, int, int, int]] = []
        for y in range(min(self.height, other.height)):
            self_row, other_row = self.grid[y], other.grid[y]
            for x in range(min(len(self_row), len(other_row))):
                if self_row[x] != other_row[x]:
                    changes.append((x, y, other_row[x], self_row[x]))
        return _cluster_changes(changes, margin=margin)

    def render_diff(
        self,
        other: Self,
        keys: str = "0123456789abcdef",
        gap: str = " ",
        crop: tuple[int, int, int, int] | Literal["auto"] | None = None,
    ) -> str:
        """
        Render a visual diff.  Call as ``new_frame.render_diff(old_frame)``.

        Args:
            keys: Character map for color values 0-15.
            gap: Separator between cells.
            crop: Region to render.
                - None: show all changes (no crop, full grid bounds).
                - 'auto': auto-crop to the bounding box of all changes.
                - (x1, y1, x2, y2): explicit crop with exclusive end.

        Changed cells show their new value; unchanged cells show as ".".
        Includes a summary header and y/x tick marks.

        Tip: call ``diff()`` first to get DiffRegions, then pass a region's
        bounds as ``crop=(r.x0, r.y0, r.x1, r.y1)`` to zoom into it.
        """
        regions = self.diff(other)
        if not regions:
            return "No changes."

        all_changes: dict[tuple[int, int], int] = {}
        total = 0
        for region in regions:
            for x, y, _, new_val in region.changes:
                all_changes[(x, y)] = new_val
            total += region.count

        if crop == "auto":
            min_x = min(r.x0 for r in regions)
            min_y = min(r.y0 for r in regions)
            max_x = max(r.x1 for r in regions)
            max_y = max(r.y1 for r in regions)
        elif crop is not None:
            min_x, min_y, max_x, max_y = crop
        else:
            min_x, min_y = 0, 0
            max_x, max_y = self.width, self.height

        header = f"{total} changes in {len(regions)} region{'s' if len(regions) != 1 else ''}"
        if len(regions) > 1:
            region_strs = [
                f"  [{r.x0},{r.y0})-[{r.x1},{r.y1}): {r.count} changes" for r in regions
            ]
            header += "\n" + "\n".join(region_strs)

        lines: list[str] = [header]

        pad = "       "
        lines.append(pad + gap.join(str(c // 10) for c in range(min_x, max_x)))
        lines.append(pad + gap.join(str(c % 10) for c in range(min_x, max_x)))
        data_width = len(gap.join("x" for _ in range(min_x, max_x)))
        lines.append(pad + "-" * data_width)

        for y in range(min_y, max_y):
            cells: list[str] = []
            for x in range(min_x, max_x):
                if (x, y) in all_changes:
                    cells.append(keys[all_changes[(x, y)]])
                else:
                    cells.append(".")
            lines.append(f"y={y:>2d} | {gap.join(cells)}")

        return "\n".join(lines)

    def change_summary(self, other: Self, margin: int = 2) -> str:
        """
        One-line-per-region summary.  Call as ``new_frame.change_summary(old_frame)``.

        Returns region bounding boxes, cell counts, and color transitions.
        Example output::

            24 cells changed across 3 region(s):
              [10,5)-[15,10): 8 cells -- 0→5 ×6, 3→5 ×2
              [30,40)-[32,42): 4 cells -- 3→0 ×4
              [55,60)-[60,64): 12 cells -- 0→7 ×10, 0→2 ×2

        Returns "No changes." when the grids are identical.
        Returns a short notice when a level transition occurred (the grids
        belong to different levels so a full diff is not meaningful).
        """
        if self.levels_completed != other.levels_completed:
            return (
                f"Level changed ({other.levels_completed} → {self.levels_completed}). "
                f"Inspect the new grid directly."
            )
        regions = self.diff(other, margin=margin)
        if not regions:
            return "No changes."
        total = sum(r.count for r in regions)
        lines = [f"{total} cells changed across {len(regions)} region(s):"]
        for r in regions:
            counts: dict[tuple[int, int], int] = {}
            for _, _, old, new in r.changes:
                key = (old, new)
                counts[key] = counts.get(key, 0) + 1
            transitions = sorted(counts.items(), key=lambda kv: -kv[1])
            parts = ", ".join(f"{o}→{n} ×{c}" for (o, n), c in transitions)
            lines.append(
                f"  [{r.x0},{r.y0})-[{r.x1},{r.y1}): {r.count} cells -- {parts}"
            )
        return "\n".join(lines)

    def find(self, *colors: int) -> list[tuple[int, int, int]]:
        """
        All pixels matching any of the given color values.

        Returns:
            [(x, y, value), ...] sorted by (y, x).
        """
        g = self.grid_np
        mask = np.isin(g, colors)
        ys, xs = np.where(mask)
        vals = g[ys, xs]
        return [(int(x), int(y), int(v)) for y, x, v in zip(ys, xs, vals)]

    def color_counts(self) -> dict[int, int]:
        """
        Count of each color value present in the grid.
        """
        bins = np.bincount(self.grid_np.ravel(), minlength=16)
        return {int(c): int(n) for c, n in enumerate(bins) if n}

    def bounding_box(self, *colors: int) -> tuple[int, int, int, int] | None:
        """
        Tight bounding box of matching pixels: (x1, y1, x2, y2), exclusive end.

        At least one color must be specified.
        Returns None if no pixels match.
        """
        mask = np.isin(self.grid_np, colors)
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None
        return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    def __repr__(self) -> str:
        actions = ", ".join(self.available_actions)
        return (
            f"Frame(level={self.levels_completed}/{self.win_levels}, "
            f"state={self.state.name}, "
            f"grid={self.width}x{self.height}, "
            f"actions=[{actions}])"
        )
