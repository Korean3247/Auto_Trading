from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


class CheckpointManager:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_checkpoints(self) -> List[Path]:
        return sorted([p for p in self.base_dir.rglob("*.pt")])

    def load_metadata(self, ckpt_path: Path) -> Dict:
        meta_path = ckpt_path.with_suffix(".json")
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        return {}

    def save_metadata(self, ckpt_path: Path, metadata: Dict) -> None:
        meta_path = ckpt_path.with_suffix(".json")
        meta_path.write_text(json.dumps(metadata, indent=2))

    def latest(self, subdir: Optional[str] = None) -> Optional[Path]:
        base = self.base_dir / subdir if subdir else self.base_dir
        candidates = sorted(base.rglob("*.pt"), reverse=True)
        return candidates[0] if candidates else None

    def resolve(self, name: str) -> Optional[Path]:
        target = self.base_dir / name
        if target.exists():
            return target
        for p in self.base_dir.rglob("*.pt"):
            if p.stem == name:
                return p
        return None
