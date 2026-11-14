"""File-persisted mask revision store for review sessions.

The in-memory :class:`~restoration.scratch.MaskRevisionStore` defines the
versioning rules; this subclass adds session-directory persistence so a server
restart keeps the reviewed history.  Each revision is a numbered mask PNG plus a
JSON manifest, and ``mask.png`` always mirrors the head revision for download
compatibility.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from .scratch import MaskRevisionStore


class FileMaskRevisionStore(MaskRevisionStore):
    def __init__(self, session_dir, shape: tuple[int, int]):
        super().__init__(shape)
        self._dir = session_dir
        self._revisions_dir = self._dir / "revisions"
        self._manifest = self._dir / "revisions.json"
        self._revisions_dir.mkdir(parents=True, exist_ok=True)
        self._sync_head()

    @classmethod
    def open(cls, session_dir) -> "FileMaskRevisionStore":
        manifest = session_dir / "revisions.json"
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        store = cls.__new__(cls)
        store._dir = session_dir
        store._revisions_dir = session_dir / "revisions"
        store._manifest = manifest
        store._shape = (int(meta["height"]), int(meta["width"]))
        MaskRevisionStore.__init__(store, store._shape)
        store._revisions = list(store._revisions[:1])
        for entry in meta["revisions"][1:]:
            arr = cv2.imread(str(store._revisions_dir / entry["file"]),
                             cv2.IMREAD_GRAYSCALE) > 0
            rev = store.commit(
                arr, entry["origin"], based_on_proposal=entry.get("based_on_proposal")
            )
            assert rev.mask_id == entry["mask_id"]
        return store

    def _rev_file(self, revision: int) -> str:
        return f"r{revision:04d}.png"

    def _sync_head(self) -> None:
        head = self.head
        cv2.imwrite(str(self._revisions_dir / self._rev_file(head.revision)),
                    head.mask.astype(np.uint8) * 255)
        # The downloadable working mask always equals the head revision.
        cv2.imwrite(str(self._dir / "mask.png"), head.mask.astype(np.uint8) * 255)
        manifest = {
            "width": self._shape[1],
            "height": self._shape[0],
            "revisions": [
                {
                    "revision": r.revision,
                    "file": self._rev_file(r.revision),
                    "origin": r.origin,
                    "mask_id": r.mask_id,
                    "based_on_proposal": r.based_on_proposal,
                }
                for r in self._revisions
            ],
        }
        self._manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def commit(self, mask, origin: str, based_on_proposal: str | None = None):
        rev = super().commit(mask, origin, based_on_proposal)
        self._sync_head()
        return rev

    def undo(self):
        head = super().undo()
        self._sync_head()
        return head
