"""Versioned scratch workflow: candidates, revisions, frozen masks.

Covers the three-stage decoupling and its invariants:
* proposals are immutable and content-identified;
* the revision store is the only owner of the working selection and supports undo;
* the frozen mask is an immutable snapshot the processor cannot reach back into.
"""
import numpy as np
import pytest

from restoration.scratch import (
    CandidateProposal,
    FrozenMask,
    MaskRevisionStore,
    mask_identity,
)


def _scratch_image(h=120, w=240, value=0.05, bg=0.5):
    img = np.full((h, w, 3), bg, np.float32)
    img = img + np.random.default_rng(0).normal(0, 0.003, img.shape).astype(np.float32)
    img[60, 40:200] = value
    return img


def test_proposal_has_content_identity_and_is_immutable():
    img = _scratch_image()
    proposal = CandidateProposal.create(img)
    assert proposal.proposal_id == mask_identity(proposal.mask)
    assert not proposal.mask.flags.writeable
    before = proposal.mask.copy()
    with pytest.raises(ValueError):
        proposal.mask[0, 0] = True  # the stored array is read-only
    assert np.array_equal(proposal.mask, before)


def test_revision_store_versions_every_edit_and_links_proposal():
    img = _scratch_image()
    proposal = CandidateProposal.create(img)
    store = MaskRevisionStore(img.shape[:2])
    assert store.head_revision == 0 and store.head_mask.sum() == 0

    rev1 = store.apply_proposal(proposal)
    assert rev1.revision == 1
    assert rev1.origin == "candidates"
    assert rev1.based_on_proposal == proposal.proposal_id
    assert store.head_revision == 1
    assert (store.head_mask & proposal.mask).sum() == proposal.mask.sum()

    extra = np.zeros_like(store.head_mask)
    extra[10:14, 10:14] = True
    rev2 = store.commit(store.head_mask | extra, "stroke")
    assert rev2.revision == 2
    assert rev2.mask_id == mask_identity(store.head_mask)

    # Handed-out masks are copies: mutating one cannot corrupt history.
    view = store.head_mask
    view[:] = True
    assert store.head_mask.sum() != view.size


def test_undo_rewinds_and_initial_revision_is_the_floor():
    shape = (40, 60)
    store = MaskRevisionStore(shape)
    m1 = np.zeros(shape, bool); m1[5, 5:10] = True
    m2 = np.zeros(shape, bool); m2[8, 8:12] = True
    r1 = store.commit(m1, "stroke")
    r2 = store.commit(m2, "stroke")
    assert store.head_revision == 2
    back = store.undo()
    assert back.revision == 1 and np.array_equal(store.head_mask, m1)
    floor = store.undo()
    assert floor.revision == 0 and store.head_mask.sum() == 0
    again = store.undo()  # cannot go below the initial revision
    assert again.revision == 0


def test_freeze_is_detached_from_later_revisions():
    shape = (30, 50)
    store = MaskRevisionStore(shape)
    m = np.zeros(shape, bool); m[10:12, 10:20] = True
    store.commit(m, "stroke")
    frozen = store.freeze()
    assert isinstance(frozen, FrozenMask)
    assert frozen.mask_id == mask_identity(m)
    assert frozen.source_revision == 1

    # More painting after the freeze must not change what the run repairs.
    later = np.zeros(shape, bool); later[0, 0] = True
    store.commit(later, "stroke")
    assert np.array_equal(frozen.mask, m)
    assert store.freeze().mask_id != frozen.mask_id


def test_frozen_mask_rejects_tampered_identity():
    m = np.zeros((20, 20), bool); m[3:6, 3:6] = True
    with pytest.raises(ValueError):
        FrozenMask(mask=m, mask_id="deadbeef", source_revision=1)


def test_frozen_mask_from_array_copies_input():
    m = np.zeros((15, 15), bool); m[2, 2] = True
    frozen = FrozenMask.from_array(m)
    m[2, 2] = False
    assert frozen.mask[2, 2] and frozen.pixel_count == 1


def test_store_rejects_shape_mismatch():
    store = MaskRevisionStore((10, 10))
    with pytest.raises(ValueError):
        store.commit(np.zeros((8, 8), bool), "stroke")
