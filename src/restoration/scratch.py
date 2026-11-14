"""Versioned scratch workflow: candidates, reviewed revisions, frozen repair masks.

The mask flow is split into three explicit stages, and every stage links to the
previous one by an explicit version identity instead of sharing a mutable file:

1. **Candidate proposal** (:class:`CandidateProposal`) - the detector's output,
   identified by a content hash.  A proposal never changes a reviewed mask on
   its own; the user must commit it.
2. **Mask revision** (:class:`MaskRevisionStore`) - an append-only history of
   reviewed masks.  Brush strokes, candidate application, clears and inversion
   are committed as numbered revisions, and :meth:`MaskRevisionStore.undo`
   rewinds without deleting history.  The store is the only owner of the
   working selection.
3. **Frozen mask** (:class:`FrozenMask`) - an immutable snapshot taken at repair
   time.  The processor receives this snapshot, cannot reach back into the
   revision store, and every preview/export written for a run derives from the
   same snapshot, so what was reviewed is provably what was repaired.

No reconstruction algorithm lives here.  Pixel evaluation is done by
:func:`restoration.services.repair_with_frozen_mask`, which consumes a
:class:`FrozenMask` and returns the same outputs as the legacy pipeline entry.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .detect import Candidate, DetectParams, detect_candidates


def mask_identity(mask: np.ndarray) -> str:
    """Stable content identity for a boolean mask (shape + packed bytes)."""
    arr = np.ascontiguousarray(mask.astype(bool))
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("ascii"))
    h.update(b"\x00")
    h.update(arr.tobytes())
    return h.hexdigest()


def _readonly(mask: np.ndarray) -> np.ndarray:
    """Return an owned, contiguous, non-writable boolean copy of the mask."""
    arr = np.array(mask, dtype=bool, copy=True, order="C")
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class CandidateProposal:
    """An immutable detector output with an explicit version identity."""

    mask: np.ndarray
    candidates: tuple[Candidate, ...]
    diagnostics: dict
    params: DetectParams
    proposal_id: str

    @classmethod
    def create(
        cls,
        image: np.ndarray,
        params: DetectParams | None = None,
    ) -> "CandidateProposal":
        p = params or DetectParams()
        mask, candidates, diagnostics = detect_candidates(image, p)
        frozen = _readonly(mask)
        return cls(
            mask=frozen,
            candidates=tuple(candidates),
            diagnostics=dict(diagnostics),
            params=p,
            proposal_id=mask_identity(frozen),
        )

    @property
    def pixel_count(self) -> int:
        return int(self.mask.sum())


@dataclass(frozen=True)
class MaskRevision:
    """One reviewed mask version.  Revisions are never mutated in place."""

    revision: int
    mask: np.ndarray
    origin: str                 # "initial" | "stroke" | "candidates" | "clear" | "invert"
    mask_id: str
    based_on_proposal: str | None = None


class MaskRevisionStore:
    """Append-only reviewed-mask history with explicit undo.

    Only this store owns the working selection.  The repair processor never
    writes here; it asks for :meth:`freeze` instead.
    """

    def __init__(self, shape: tuple[int, int]):
        if len(shape) != 2:
            raise ValueError(f"mask shape must be 2D, got {shape}")
        self._shape = (int(shape[0]), int(shape[1]))
        initial = _readonly(np.zeros(self._shape, dtype=bool))
        self._revisions: list[MaskRevision] = [
            MaskRevision(0, initial, "initial", mask_identity(initial))
        ]

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def head_revision(self) -> int:
        return self._revisions[-1].revision

    @property
    def head(self) -> MaskRevision:
        return self._revisions[-1]

    @property
    def head_mask(self) -> np.ndarray:
        # A writable copy: callers painting strokes must not mutate history.
        return self._revisions[-1].mask.copy()

    def history(self) -> tuple[MaskRevision, ...]:
        return tuple(self._revisions)

    def commit(
        self,
        mask: np.ndarray,
        origin: str,
        based_on_proposal: str | None = None,
    ) -> MaskRevision:
        """Append a reviewed version; identical content is still a new revision."""
        arr = np.asarray(mask, dtype=bool)
        if arr.shape != self._shape:
            raise ValueError(
                f"mask shape {arr.shape} does not match store shape {self._shape}"
            )
        stored = _readonly(arr)
        rev = MaskRevision(
            revision=self.head_revision + 1,
            mask=stored,
            origin=origin,
            mask_id=mask_identity(stored),
            based_on_proposal=based_on_proposal,
        )
        self._revisions.append(rev)
        return rev

    def apply_proposal(self, proposal: CandidateProposal) -> MaskRevision:
        """Explicit user action: union accepted candidates into the head mask."""
        if proposal.mask.shape != self._shape:
            raise ValueError(
                f"proposal shape {proposal.mask.shape} does not match store shape {self._shape}"
            )
        merged = self.head_mask | proposal.mask
        return self.commit(merged, "candidates", based_on_proposal=proposal.proposal_id)

    def undo(self) -> MaskRevision:
        """Rewind one reviewed revision.  The initial empty revision is the floor."""
        if len(self._revisions) > 1:
            self._revisions.pop()
        return self.head

    def freeze(self) -> "FrozenMask":
        """Snapshot the head for a repair run; later revisions cannot affect it."""
        head = self.head
        return FrozenMask(
            mask=head.mask.copy(),
            mask_id=head.mask_id,
            source_revision=head.revision,
            based_on_proposal=head.based_on_proposal,
        )


@dataclass(frozen=True)
class FrozenMask:
    """Immutable mask snapshot consumed by the repair processor.

    The processor is handed the boolean array directly and never receives the
    revision store, so it cannot modify the UI's working selection.  Export and
    preview share this one snapshot.
    """

    mask: np.ndarray
    mask_id: str
    source_revision: int
    based_on_proposal: str | None = None

    def __post_init__(self) -> None:
        stored = _readonly(self.mask)
        if mask_identity(stored) != self.mask_id:
            raise ValueError("mask_id does not match mask content")
        object.__setattr__(self, "mask", stored)

    @property
    def pixel_count(self) -> int:
        return int(self.mask.sum())

    @classmethod
    def from_array(
        cls,
        mask: np.ndarray,
        *,
        source_revision: int = 0,
        based_on_proposal: str | None = None,
    ) -> "FrozenMask":
        arr = _readonly(mask)
        return cls(
            mask=arr,
            mask_id=mask_identity(arr),
            source_revision=source_revision,
            based_on_proposal=based_on_proposal,
        )
