from modiscolite import util


def test_profile_recorder_tracks_named_blocks():
    profiler = util.ProfileRecorder()

    with profiler.time("stage"):
        pass

    summary = profiler.summary()
    assert summary["stage"]["count"] == 1
    assert summary["stage"]["seconds"] >= 0.0


def test_disabled_profile_recorder_does_not_record_blocks():
    profiler = util.ProfileRecorder(enabled=False)

    with profiler.time("stage"):
        pass

    assert profiler.records == []


def test_ensure_profile_recorder_reuses_existing_recorder():
    profiler = util.ProfileRecorder()

    assert util.ensure_profile_recorder(profiler) is profiler
    assert util.ensure_profile_recorder(False).enabled is False
    assert util.ensure_profile_recorder(True).enabled is True
