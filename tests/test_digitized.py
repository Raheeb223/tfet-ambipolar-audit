import pytest
import yaml

from src.dataio import digitized as dz

ENTRY = dict(
    file="test_curve.csv", source="synthetic test", doi_or_url="n/a", figure="n/a",
    curve_label="t", x_quantity="voltage", x_unit="mV", y_quantity="current_density",
    y_unit="mA/cm2", y_scale="log", temperature_K=300, bias_direction="reverse",
    doping_info="n/a", area_um2=None, digitizer_repeats=3, est_error="n/a", role="core_calibration",
)


def _write(tmp_path, entries, csv="V,J\n-100,1.0\n-200,10.0\n"):
    (tmp_path / "test_curve.csv").write_text(csv)
    (tmp_path / "metadata.yaml").write_text(yaml.safe_dump({"datasets": entries}))
    return tmp_path


def test_repo_metadata_is_valid_and_roles_are_separated():
    """The project's digitized data load, and no validation curve can reach a calibration."""
    ents = dz.load_metadata()
    assert len(dz.datasets_by_role(ents, "core_calibration")) >= 2
    assert len(dz.datasets_by_role(ents, "validation")) >= 3
    cal = dz.calibration_datasets(ents)
    assert all(e["role"] in dz.CALIBRATION_ROLES for e in cal)
    assert not any(e["file"].startswith("wan2011") for e in cal)


def test_load_and_convert_units(tmp_path):
    d = _write(tmp_path, [ENTRY])
    (e,) = dz.load_metadata(d)
    c = dz.load_curve(e, d)
    assert list(c.x) == pytest.approx([-0.1, -0.2])
    assert list(c.y) == pytest.approx([1e-3, 1e-2])
    assert c.attrs["y_unit_canonical"] == "A/cm2"


@pytest.mark.parametrize("change", [
    {"role": "fit_me"}, {"x_unit": "furlongs"}, {"y_scale": "semilog"}, {"digitizer_repeats": 0},
])
def test_rejects_bad_metadata(tmp_path, change):
    with pytest.raises(dz.MetadataError):
        dz.load_metadata(_write(tmp_path, [dict(ENTRY, **change)]))


def test_rejects_missing_field(tmp_path):
    e = dict(ENTRY)
    del e["source"]
    with pytest.raises(dz.MetadataError, match="source"):
        dz.load_metadata(_write(tmp_path, [e]))


def test_rejects_nonpositive_log_data(tmp_path):
    with pytest.raises(dz.MetadataError, match="log-scale"):
        dz.load_metadata(_write(tmp_path, [ENTRY], csv="V,J\n-100,-1.0\n"))


def test_warns_on_fewer_than_three_repeats(tmp_path):
    with pytest.warns(UserWarning, match="3x"):
        dz.load_metadata(_write(tmp_path, [dict(ENTRY, digitizer_repeats=1)]))


def test_validation_data_never_used_for_calibration(tmp_path):
    d = _write(tmp_path, [dict(ENTRY, role="validation")])
    ents = dz.load_metadata(d)
    assert dz.calibration_datasets(ents) == []
    with pytest.raises(dz.MetadataError):
        dz.calibration_datasets(ents, roles={"validation"})
