"""
Arcgentica agent — ARC-AGI-3 harness built on the Agentica SDK.
"""

import asyncio
import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal

import numpy as np
from arcengine import FrameData, GameAction, GameState

from agentica import spawn
from agentica.logging import AgentListener
from agents.agent import Agent
from agents.tracing import trace_agent_session

from .logging.events import UsageSummaryEvent
from .logging.logger import EventServer, WsLogger
from .logging.tracker import UsageTracker
from .model import OPUS_4_6, ModelConfig
from .prompts import GAME_REFERENCE, premise
from .scope import FinishStatus, Frame, Memories, Memory

logger = logging.getLogger()

ActionName = Literal[
    "RESET",
    "ACTION1",
    "ACTION2",
    "ACTION3",
    "ACTION4",
    "ACTION5",
    "ACTION6",
]


class Arcgentica(Agent):
    """ARC-AGI-3 agent harness built on the Agentica SDK."""

    def __init__(
        self,
        *args: Any,
        model: ModelConfig = OPUS_4_6,
        visualize: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.visualize = visualize or os.environ.get("VISUALIZE", "") == "1"
        self._server: EventServer | None = None
        self._tracker: UsageTracker | None = None
        self._pending_reasoning: dict[str, object] | None = None
        self._action_log_dir: Path | None = None
        self._action_log_file: Any = None
        self._logged_level: int = -1

    def _log_action(self, action: GameAction, frame: Frame) -> None:
        """
        Append one JSONL line per action. Rotates file on level change.
        """
        if self._action_log_dir is None:
            self._action_log_dir = Path("actions_log") / self.game_id
            self._action_log_dir.mkdir(parents=True, exist_ok=True)

        level = frame.levels_completed
        if level != self._logged_level:
            if self._action_log_file is not None:
                self._action_log_file.close()
            path = self._action_log_dir / f"level_{level}.jsonl"
            self._action_log_file = open(path, "a")
            self._logged_level = level

        entry = {
            "action": action.name,
            "count": self.action_counter,
            "level": level,
            "state": frame.state.name,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if action.is_complex():
            entry["x"] = action.action_data.x
            entry["y"] = action.action_data.y

        self._action_log_file.write(json.dumps(entry) + "\n")
        self._action_log_file.flush()

    def _close_action_log(self) -> None:
        if self._action_log_file is not None:
            self._action_log_file.close()
            self._action_log_file = None

    def do_action_request(self, action: GameAction) -> FrameData:
        data = action.action_data.model_dump()
        reasoning = self._pending_reasoning or {}
        self._pending_reasoning = None
        raw = self.arc_env.step(action, data=data, reasoning=reasoning)
        return self._convert_raw_frame_data(raw)

    # Required abstract methods (not used since we override main)
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        raise NotImplementedError("Arcgentica agent overrides main()")

    def _make_submit_action(self, server: EventServer | None = None):
        """
        Create the submit_action tool function for the agentica agent.
        """

        _MAX_HISTORY = 50
        last_available: list[int] = []
        # True when a non-RESET action has been taken since the last reset.
        # When False, the engine's _action_count is 0 and a RESET would
        # trigger a destructive full_reset back to level 0.
        has_moves_since_reset: bool = False
        last_frame: Frame | None = None
        _action_history: deque[tuple[str, Frame]] = deque(maxlen=_MAX_HISTORY)
        _win_history: list[tuple[str, Frame]] = []

        def _push_frame(action: GameAction, raw: FrameData) -> Frame:
            nonlocal last_available, has_moves_since_reset, last_frame
            last_available = raw.available_actions
            self.append_frame(raw)
            if action is not GameAction.RESET:
                self.action_counter += 1
            if action is GameAction.RESET:
                has_moves_since_reset = False
            elif (
                last_frame is not None
                and raw.levels_completed != last_frame.levels_completed
            ):
                # Level transition
                has_moves_since_reset = False
            else:
                has_moves_since_reset = True
            logger.info(
                f"{self.game_id} - {action.name}: count {self.action_counter}, "
                f"level {raw.levels_completed}/{raw.win_levels}"
            )
            prev_lc = last_frame.levels_completed if last_frame is not None else None
            frame = Frame(raw, prev_levels_completed=prev_lc)
            last_frame = frame
            _action_history.append((action.name, frame))
            if frame.winning_frame is not None:
                _win_history.append((action.name, frame))
            self._log_action(action, frame)
            if server is not None:
                cx = action.action_data.x if action.is_complex() else None
                cy = action.action_data.y if action.is_complex() else None
                server.push_game_action(
                    action.name,
                    frame,
                    self.action_counter,
                    click_x=cx,
                    click_y=cy,
                )
                if self._tracker is not None:
                    t = self._tracker.total_usage()
                    server.push(
                        UsageSummaryEvent(
                            input_tokens=t.input_tokens,
                            output_tokens=t.output_tokens,
                            cached_tokens=t.cached_tokens,
                            reasoning_tokens=t.reasoning_tokens,
                            total_tokens=t.total_tokens,
                        )
                    )
            return frame

        def submit_action(
            action_name: ActionName | Literal["NOOP"], x: int = 0, y: int = 0
        ) -> Frame:
            """
            Submit a game action and receive the new frame (sync function).

            Args:
                action_name: One of RESET, ACTION1-ACTION6.
                    You can also pass "NOOP" to retrieve the current frame
                    without taking any action or incrementing the action count.
                x: X coordinate (0-63), only for ACTION6
                y: Y coordinate (0-63), only for ACTION6

            Returns:
                Frame with the new game state, grid, and available actions.
            """
            if action_name.upper() == "NOOP":
                assert last_frame is not None, "NOOP before any RESET"
                return last_frame

            action = GameAction.from_name(action_name)

            if (
                last_available
                and action is not GameAction.RESET
                and action.value not in last_available
            ):
                allowed = [GameAction.from_id(a).name for a in last_available]
                raise ValueError(
                    f"{action.name} is not available. Available actions: {allowed}"
                )

            # Block redundant resets when no moves have been made.
            if action is GameAction.RESET and not has_moves_since_reset:
                if last_frame is not None:
                    logger.info(f"{self.game_id} - RESET skipped (level already clean)")
                    return last_frame

            if action.is_complex():
                action.set_data({"x": x, "y": y})

            if self._tracker is not None:
                reasoning = self._tracker.drain_reasoning()
                if reasoning is not None:
                    reasoning["level"] = (
                        last_frame.levels_completed if last_frame else 0
                    )
                    reasoning["action_count"] = self.action_counter
                self._pending_reasoning = reasoning

            raw = self.take_action(action)
            if raw is None:
                raise RuntimeError(
                    f"{action.name} returned no frame data. "
                    "The server may have rejected the action."
                )
            return _push_frame(action, raw)

        def history(
            n: int = _MAX_HISTORY, wins_only: bool = False
        ) -> list[tuple[str, Frame]]:
            """
            Return the last n (action_name, Frame) pairs from the game, oldest first.
            This is a synchronous function, do NOT use await.

            Covers actions taken by ALL agents (not just the current one). Use this
            to review what happened after executing a sequence of actions, or to
            understand the game state before you started.

            Args:
                n: How many recent entries to return. Defaults to 50 (the max stored).
                wins_only: If True, return only entries where frame.winning_frame is
                    not None (i.e. actions that completed a level). Useful for
                    reviewing what the winning state looked like on past levels.
            """
            if wins_only:
                entries = _win_history
            else:
                entries = list(_action_history)
            return entries[-n:] if n < len(entries) else list(entries)

        return submit_action, history

    class bounded_submit_action:
        """Callable ``submit_action`` wrapper with an action budget.

        Attributes:
            remaining: how many game actions are left.
            used: how many game actions have been spent.
            limit: the total budget this was created with.
        """

        remaining: int
        used: int
        limit: int

        def __init__(self, inner, limit: int) -> None:
            self._inner = inner
            self._limit = limit
            self._used = 0
            self.__class__.__doc__ = dedent(inner.__doc__ or "")

        def __call__(
            self, action_name: ActionName | Literal["NOOP"], x: int = 0, y: int = 0
        ) -> Frame:
            upper = action_name.upper()
            if upper != "NOOP" and upper != "RESET":
                if self._used >= self._limit:
                    raise ValueError(
                        f"Action budget exhausted: all {self._limit} actions have been used."
                    )
                self._used += 1
            return self._inner(action_name, x, y)

        @property
        def remaining(self) -> int:
            """How many game actions are left in this budget."""
            return self._limit - self._used

        @property
        def used(self) -> int:
            """How many game actions have been spent so far."""
            return self._used

        @property
        def limit(self) -> int:
            """The total action budget this was created with."""
            return self._limit

        def __repr__(self) -> str:
            return f"bounded_submit_action({self.used}/{self.limit} used, {self.remaining} remaining)"

    @staticmethod
    def _make_bounded_submit_action(inner, limit: int | None):
        """
        Wrap an existing submit_action, optionally with a hard action budget.

        The returned callable delegates to `inner` for all calls. It shares
        inner's closure state (last_frame, has_moves_since_reset, etc.) so
        RESET guards and NOOP work correctly.

        Args:
            inner: The submit_action to wrap.
            limit: Maximum non-NOOP, non-RESET actions allowed, or None for unlimited.
        """
        if limit is None:
            return inner
        return Arcgentica.bounded_submit_action(inner, limit)

    def _make_listener(self):
        """Build a listener constructor, or None if no server is active."""
        server = self._server
        tracker = self._tracker
        if server is None and tracker is None:
            return None
        return lambda: AgentListener(WsLogger(server, tracker=tracker))

    async def spawn_agent(self, system_prompt: str | None = None):
        """
        Spawn a new subagent that can be called repeatedly.

        Returns an agent handle. Use ``await agent.call(return_type, task, **objects)``
        to invoke it. The same handle can be called multiple times; each call
        continues the conversation so the agent retains context from prior calls.
        Pass ``submit_action`` only to agents that need to take game actions.

        Be careful about when you want to reuse context vs. spawn a new agent,
        performance degrades as context grows, so there is a trade-off you have to consider.
        """
        return await spawn(
            model=self.model.subagent_model,
            premise=system_prompt,
            reasoning_effort=self.model.reasoning_effort,
            listener=self._make_listener(),
            scope={
                "spawn_agent": self.spawn_agent,
                "numpy": np,
                "np": np,
                "Memories": Memories,
                "Memory": Memory,
            },
        )

    async def _run(self) -> None:
        """Run an agent on a single ARC-AGI-3 game."""
        tracker = UsageTracker()
        self._tracker = tracker

        server: EventServer | None = None
        if self.visualize:
            server = EventServer(game_id=self.game_id)
            await server.start()
            self._server = server
        submit_action, history = self._make_submit_action(server=server)

        def make_bounded_submit_action(limit: int | None):
            """
            Create a new ``submit_action`` function, optionally with a hard action budget.

            Returns a ``submit_action`` that works identically to the normal one.
            If ``limit`` is an int, raises ValueError after that many game actions.
            If ``limit`` is None, the returned function is unbounded.
            NOOP and RESET are free and do not count toward the limit.

            Use this to enforce action budgets on subagents:
                bounded_sa = make_bounded_submit_action(10)
                await agent.call(..., submit_action=bounded_sa)

            Args:
                limit: Max game actions (ACTION1-ACTION6) allowed, or None for unlimited.
            """
            return Arcgentica._make_bounded_submit_action(submit_action, limit)

        # Double RESET guarantees a completely fresh game
        initial_raw = self.take_action(GameAction.RESET)
        if initial_raw:
            self.append_frame(initial_raw)

        initial_frame = submit_action("RESET")

        orchestrator = await spawn(
            model=self.model.main_agent_model,
            premise=premise(self.model),
            reasoning_effort=self.model.reasoning_effort,
            listener=self._make_listener(),
            scope={
                "spawn_agent": self.spawn_agent,
                "numpy": np,
                "np": np,
                "Memories": Memories,
                "Memory": Memory,
            },
        )

        actions = ", ".join(initial_frame.available_actions)

        memories = Memories(model=self.model.subagent_model)

        status = await orchestrator.call(
            FinishStatus,
            f"""You are playing the game `{self.game_id}`. \
Level {initial_frame.levels_completed}/{initial_frame.win_levels}. \
Available actions for this level: {actions}

You have a shared `memories` database — pass it to every subagent \
so they can read prior knowledge and write new discoveries as they go. \
This persists across agent lifetimes, so use it as the primary way to \
accumulate and transfer knowledge.

Take a moment to plan your approach before spawning any agents. \
When ready, spawn an explorer and give it a bounded submit_action, \
`initial_frame`, `memories`, and `GAME_REFERENCE`.""",
            initial_frame=initial_frame,
            make_bounded_submit_action=make_bounded_submit_action,
            history=history,
            memories=memories,
            GAME_REFERENCE=GAME_REFERENCE,
        )
        logger.info(f"Finish status: {status}")

    def _write_usage(self) -> None:
        if self._tracker is None:
            return
        print("\n" + self._tracker.summary() + "\n")
        if self._action_log_dir is None:
            self._action_log_dir = Path("actions_log") / self.game_id
            self._action_log_dir.mkdir(parents=True, exist_ok=True)
        path = self._action_log_dir / "usage.json"
        path.write_text(json.dumps(self._tracker.to_dict(), indent=2))
        logger.info(f"Token usage written to {path}")

    @trace_agent_session
    def main(self) -> None:
        """
        Override the base agent loop — agentica drives the game.
        """
        self.timer = time.time()
        try:
            asyncio.run(self._run())
        finally:
            self._write_usage()
            self._close_action_log()
            if self._server is not None:
                self._server.close_log()
            self.cleanup()
