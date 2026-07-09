"""In-process message bus for brain-to-brain communication.

Brains never call each other directly; they publish/subscribe on topics.
That keeps every brain replaceable and independently testable (spec: "no
module should become a hardcoded monolith").
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class BusMessage:
    topic: str
    payload: Any
    sender: str = "unknown"
    timestamp: float = field(default_factory=time.time)


class FabricBus:
    def __init__(self, keep_history: int = 500):
        self._subscribers: Dict[str, List[Callable[[BusMessage], None]]] = \
            defaultdict(list)
        self._history: List[BusMessage] = []
        self._keep_history = keep_history
        self._lock = threading.RLock()

    def subscribe(self, topic: str,
                  handler: Callable[[BusMessage], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers[topic].append(handler)

        def unsubscribe() -> None:
            with self._lock:
                if handler in self._subscribers.get(topic, []):
                    self._subscribers[topic].remove(handler)

        return unsubscribe

    def publish(self, topic: str, payload: Any, sender: str = "unknown") -> BusMessage:
        msg = BusMessage(topic=topic, payload=payload, sender=sender)
        with self._lock:
            self._history.append(msg)
            if len(self._history) > self._keep_history:
                self._history = self._history[-self._keep_history:]
            handlers = list(self._subscribers.get(topic, [])) + \
                list(self._subscribers.get("*", []))
        for handler in handlers:
            handler(msg)
        return msg

    def history(self, topic: str = None) -> List[BusMessage]:
        with self._lock:
            if topic is None:
                return list(self._history)
            return [m for m in self._history if m.topic == topic]
