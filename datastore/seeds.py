# aethermind-awareness/datastore/seeds.py
from __future__ import annotations
import os, json, dataclasses, datetime as dt
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Tuple

# Optional speedups
try:
    import orjson as _jsonlib
    def _loads(s: bytes) -> Any: return _jsonlib.loads(s)
except Exception:
    _jsonlib = None
    def _loads(s: bytes) -> Any: return json.loads(s.decode("utf-8"))

# Optional validation
try:
    from pydantic import BaseModel, Field, ValidationError
    _HAS_PYD = True
except Exception:
    BaseModel = object  # type: ignore
    Field = lambda *a, **k: None  # type: ignore
    class ValidationError(Exception): ...
    _HAS_PYD = False


# ------------------------- schema / model -------------------------

@dataclasses.dataclass
class Seed:
    event_uid: str
    session_id: str
    start: float
    end: float
    source: str
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    actions: Optional[List[Dict[str, Any]]] = None
    sync: Optional[Dict[str, Any]] = None
    video_dyn: Optional[Dict[str, Any]] = None
    audio_dyn: Optional[Dict[str, Any]] = None
    system: Optional[Dict[str, Any]] = None
    decision_trace: Optional[Dict[str, Any]] = None
    valence: Optional[Dict[str, Any]] = None
    schema_version: Optional[Dict[str, int]] = None
    created_at: Optional[str] = None
    raw: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def duration(self) -> float:
        return float(self.end) - float(self.start)

    @property
    def key(self) -> str:
        # stable dedup key
        return self.event_uid

    def resolve_media(self, media_root: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """Best-effort resolve media paths using AETHERMIND_MEDIA_ROOT if present."""
        def _res(p: Optional[str]) -> Optional[str]:
            if not p: return None
            if os.path.isabs(p) and os.path.exists(p): return p
            if media_root:
                cand = os.path.normpath(os.path.join(media_root, p))
                if os.path.exists(cand): return cand
            ap = os.path.abspath(p)
            return ap if os.path.exists(ap) else p
        root = media_root or os.environ.get("AETHERMIND_MEDIA_ROOT")
        return _res(self.video_path), _res(self.audio_path)


if _HAS_PYD:
    class PSeed(BaseModel):
        event_uid: str
        session_id: str
        start: float
        end: float
        source: str
        video_path: Optional[str] = None
        audio_path: Optional[str] = None
        actions: Optional[List[Dict[str, Any]]] = None
        sync: Optional[Dict[str, Any]] = None
        video_dyn: Optional[Dict[str, Any]] = None
        audio_dyn: Optional[Dict[str, Any]] = None
        system: Optional[Dict[str, Any]] = None
        decision_trace: Optional[Dict[str, Any]] = None
        valence: Optional[Dict[str, Any]] = None
        schema_version: Optional[Dict[str, int]] = None
        created_at: Optional[str] = None

        def to_seed(self) -> Seed:
            d = self.dict()
            return Seed(raw=d, **d)  # raw keeps exact original for downstream use


# ------------------------- IO / parsing -------------------------

def load_seeds_jsonl(path: str) -> Iterator[Seed]:
    """
    Stream Seeds from a JSONL file. Validates if pydantic is installed;
    otherwise performs minimal checks.
    """
    with open(path, "rb") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = _loads(line)
            except Exception as e:
                raise ValueError(f"{path}:{line_no} could not parse JSON: {e}") from e

            if _HAS_PYD:
                try:
                    ps = PSeed(**obj)
                except ValidationError as ve:
                    raise ValueError(f"{path}:{line_no} schema error: {ve}") from ve
                yield ps.to_seed()
            else:
                # minimal required fields
                for k in ("event_uid", "session_id", "start", "end", "source"):
                    if k not in obj:
                        raise ValueError(f"{path}:{line_no} missing required field '{k}'")
                yield Seed(
                    event_uid=obj["event_uid"],
                    session_id=obj["session_id"],
                    start=float(obj["start"]),
                    end=float(obj["end"]),
                    source=obj["source"],
                    video_path=obj.get("video_path"),
                    audio_path=obj.get("audio_path"),
                    actions=obj.get("actions"),
                    sync=obj.get("sync"),
                    video_dyn=obj.get("video_dyn"),
                    audio_dyn=obj.get("audio_dyn"),
                    system=obj.get("system"),
                    decision_trace=obj.get("decision_trace"),
                    valence=obj.get("valence"),
                    schema_version=obj.get("schema_version"),
                    created_at=obj.get("created_at"),
                    raw=obj,
                )


# ------------------------- utilities -------------------------

def count_seeds(path: str) -> int:
    c = 0
    for _ in load_seeds_jsonl(path): c += 1
    return c

def head(path: str, n: int = 5) -> List[Seed]:
    out: List[Seed] = []
    it = load_seeds_jsonl(path)
    for _ in range(n):
        try:
            out.append(next(it))
        except StopIteration:
            break
    return out
