"""Search/Delivery separation invariants (#56/#57, §93)."""

from auto_mobility.reconstruction.pipeline.controller import FrameSplit, PipelineState


def test_final_rebuild_uses_all_fuse_frames():
    split = FrameSplit(train_ids=tuple(range(0, 100)), val_ids=tuple(range(50, 120)))
    state = PipelineState(split=split, winner_id="rtab_v10mm")

    delivered = state.final_integration_set()
    assert delivered == tuple(range(0, 120))
    assert len(delivered) > len(split.search_frames())
    for holdout in (54, 55, 99, 119):
        assert holdout in delivered


def test_search_holdout_not_used_for_fusion():
    holdouts = (5, 15, 25, 85)
    train = tuple(i for i in range(0, 90) if i not in holdouts)
    split = FrameSplit(train_ids=train, val_ids=holdouts)
    search_set = set(split.search_frames())

    for holdout in split.val_ids:
        assert holdout not in search_set


def test_delivery_superset_of_search_and_holdout_included():
    train = (1, 2, 3, 4)
    val = (2, 6, 8)
    split = FrameSplit(train_ids=train, val_ids=val)

    assert set(split.delivery_frames()) == {1, 2, 3, 4, 6, 8}


def test_decision_trace_records_winner():
    state = PipelineState(winner_id="cuvslam_v8mm")
    state.record("fusion_refinement", "SKIP_FINE_VOXEL", "residual dominates", residual_mm=4.2)
    trace = state.to_decision_trace()

    assert trace["winner"] == "cuvslam_v8mm"
    assert trace["decisions"][0]["decision"] == "SKIP_FINE_VOXEL"
