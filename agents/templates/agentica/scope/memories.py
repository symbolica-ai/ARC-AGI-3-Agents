import asyncio
from dataclasses import dataclass, field
from datetime import datetime

from agentica import Agent, spawn
from agentica.logging.agent_listener import AgentListener
from agentica.logging.loggers.file_logger import FileLogger

__all__ = ["Memory", "Memories", "MemoryQueryError"]


@dataclass(slots=True, frozen=True)
class Memory:
    """
    A vital piece of information that should be remembered across all future agents.

    summary: a short, high-level description of the information.
    details: a detailed description of the information.
    """

    summary: str
    details: str
    timestamp: datetime = field(default_factory=datetime.now)


class MemoryQueryError(Exception):
    """Raised when a memory query is not possible to answer with the given memories or in the desired format."""

    message: str

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class Memories:
    """
    A database of shared memories/information.
    Use this to store and retrieve crucial insights, observations, and knowledge.

    Keep entries factual. In the details, clearly separate what is confirmed from
    what is hypothesised, e.g. "CONFIRMED: ... HYPOTHESIS: ...". Other agents
    trust this database -- unverified guesses written as facts will mislead them.
    """

    stack: list[Memory]
    _memory_agent: asyncio.Task[Agent]
    _last_seen: int

    def __init__(self, model: str) -> None:
        self.stack = []
        self._last_seen = 0
        self._memory_agent = asyncio.ensure_future(
            spawn(
                model=model,
                listener=lambda: AgentListener(FileLogger("logs/", "memory-agent-")),
                premise=(
                    "You retrieve information from a shared `memories` object. "
                    "You can call any of its methods: `memories.stack`, `memories.get(i)`, "
                    "`memories.add(summary, details)`, `memories.evict(i)`, `memories.summaries()`. "
                    "Do not make up information -- only return what the memories support. "
                    "Raise MemoryQueryError if: no memories address the question, the stored "
                    "information is too vague to give a confident answer, or the requested "
                    "return format is not appropriate for the query."
                ),
                scope={"memories": self, "MemoryQueryError": MemoryQueryError},
            )
        )

    def add(self, summary: str, details: str) -> None:
        """Append an insight."""
        self.stack.append(Memory(summary, details))

    def summaries(self) -> list[str]:
        """Short summary of every stored memory, for a quick glance at what's already known."""
        return [f"[{i}] {m.summary}" for i, m in enumerate(self.stack)]

    def get(self, i: int) -> Memory:
        """Retrieve an insight by index. Negative indices are supported."""
        return self.stack[i]

    def evict(self, i: int | None = None) -> None:
        """Remove an insight by index. If no index is provided, pop the last insight."""
        self.stack.pop(i)

    async def query[T](self, return_type: type[T], query: str) -> T:
        """
        Natural language query information from the memories.

        Use `return_type` and `query` to structure how and what information to retrieve.

        Example:
            memories.query(list[int], "What triggers X to happen? Give me a list of indices for the corresponding memory entries.")
            memories.query(str, "What is the premise of the level X?")
            memories.query(Memory, "What happens when I take action X?")
            memories.query(list[Memory], "Give me the last 3 memories pertaining to level X.")
        """
        agent = await self._memory_agent
        new_count = len(self.stack) - self._last_seen
        self._last_seen = len(self.stack)
        if new_count > 0:
            preamble = f"There are {new_count} new memories since your last call.\n\n"
        else:
            preamble = ""
        return await agent.call(
            return_type,
            f"{preamble}Answer the following query. Raise MemoryQueryError if: no "
            f"memories address the question, the stored information is too vague to "
            f"give a confident answer, or the requested return format is not appropriate "
            f"for the query.\n\nQuery: {query}",
            memories=self,
            stack=self.stack,
        )

    def __repr__(self) -> str:
        return f"Memories(<{len(self.stack)} memories>)"
