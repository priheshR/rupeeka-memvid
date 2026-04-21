from dataclasses import dataclass, field
from typing import Set, Dict, List, Tuple, Optional
import uuid
import time

@dataclass
class SessionMemory:
    """Tracks retrieval state across conversational turns."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    retrieved: Set[str] = field(default_factory=set)
    boosted: Dict[str, float] = field(default_factory=dict)
    demoted: Set[str] = field(default_factory=set)
    lang_pref: Optional[str] = None
    turn: int = 0
    created_at: float = field(default_factory=time.time)

    def record(self, results: List[Tuple[str, float]]):
        self.turn += 1
        for text, _ in results:
            self.retrieved.add(text)

    def mark_helpful(self, text: str, boost: float = 1.5):
        self.boosted[text] = boost

    def mark_not_helpful(self, text: str):
        self.demoted.add(text)

    def apply(
        self,
        results: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        adjusted = []
        for text, score in results:
            if text in self.demoted:
                continue
            score *= self.boosted.get(text, 1.0)
            adjusted.append((text, score))
        return sorted(adjusted, key=lambda x: x[1], reverse=True)
