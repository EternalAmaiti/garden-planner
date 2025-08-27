from garden_planner.visualizer import (
    GardenBed,
    _col_letters,
    build_random_clusters,
    cluster_radius_m,
    default_theme,
    export_plan_csv,
    import_plan_csv,
    make_non_overlapping_layout,
    odd_partition,
)


def test_col_letters():
    assert _col_letters(0) == "A"
    assert _col_letters(25) == "Z"
    assert _col_letters(26) == "AA"
    assert _col_letters(27) == "AB"


def test_odd_partition_properties():
    out = odd_partition(17, rng=__import__("random").Random(42), avg=5, min_odd=3)
    assert sum(out) == 17
    assert all(x % 2 == 1 for x in out)


def test_radius_monotonic():
    r3 = cluster_radius_m(10, 3)
    r5 = cluster_radius_m(10, 5)
    assert r5 > r3


def test_csv_roundtrip(tmp_path):
    bed = GardenBed(3.0, 2.5)
    theme = default_theme()
    rng = __import__("random").Random(0)
    clusters = build_random_clusters(theme, bed, rng)
    # minimal non-overlap pass
    clusters = make_non_overlapping_layout(clusters, theme, bed, gap=0.01, max_iters=10, step=0.5)
    p = tmp_path / "plan.csv"
    export_plan_csv(clusters, bed, theme, str(p))
    clusters2, w, d = import_plan_csv(str(p))
    assert isinstance(w, float) and isinstance(d, float)
    assert set(clusters2.keys()) == set(clusters.keys())
