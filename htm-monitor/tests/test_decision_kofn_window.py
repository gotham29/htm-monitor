from htm_monitor.orchestration.decision import Decision


def test_kofn_window_hot_and_alert_basic():
    d = Decision(threshold=0.5, method="kofn_window", k=2, window_size=3, per_model_hits=2)

    # t0: A hit, B miss, C miss => hot=0
    out0 = d.step({"A": {"likelihood": 0.6}, "B": {"likelihood": 0.1}, "C": {"likelihood": 0.2}})
    assert out0["alert"] is False
    assert out0["system_score"] == 0.0

    # t1: A hit again (2/3 hits), B hit (1/3), C miss => hot=1 (A)
    out1 = d.step({"A": {"likelihood": 0.8}, "B": {"likelihood": 0.6}, "C": {"likelihood": 0.2}})
    assert out1["alert"] is False
    assert out1["system_score"] == 1.0 / 3.0

    # t2: B hit again (2/3), A miss, C miss => A still hot? (A has 2 hits in window), B hot => hot=2 => alert
    out2 = d.step({"A": {"likelihood": 0.0}, "B": {"likelihood": 0.9}, "C": {"likelihood": 0.1}})
    assert out2["alert"] is True
    assert out2["system_score"] == 2.0 / 3.0


def test_kofn_window_requires_k():
    d = Decision(threshold=0.5, method="kofn_window", k=None, window_size=3, per_model_hits=1)
    try:
        d.step({"A": {"likelihood": 0.6}})
        assert False, "Expected ValueError for missing k"
    except ValueError:
        pass


def test_kofn_window_rolls_off_old_hits():
    # window_size=2, per_model_hits=2 means "both last 2 steps must be hits" to be hot
    d = Decision(threshold=0.5, method="kofn_window", k=1, window_size=2, per_model_hits=2)

    # t0 hit
    out0 = d.step({"A": {"likelihood": 0.6}})
    assert out0["system_score"] == 0.0
    assert out0["alert"] is False

    # t1 miss -> window now [hit, miss] => not hot
    out1 = d.step({"A": {"likelihood": 0.1}})
    assert out1["system_score"] == 0.0
    assert out1["alert"] is False

    # t2 hit -> window now [miss, hit] => still not hot (old hit rolled off)
    out2 = d.step({"A": {"likelihood": 0.6}})
    assert out2["system_score"] == 0.0
    assert out2["alert"] is False


def test_kofn_window_returns_hot_by_model_for_all_models():
    d = Decision(threshold=0.5, method="kofn_window", k=1, window_size=3, per_model_hits=1)

    out = d.step({"A": {"likelihood": 0.6}, "B": {"likelihood": 0.4}})
    assert "hot_by_model" in out
    assert set(out["hot_by_model"].keys()) == {"A", "B"}
    assert out["hot_by_model"]["A"] is True
    assert out["hot_by_model"]["B"] is False