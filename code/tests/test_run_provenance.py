from core.benchmark.expectations import RunProvenance


def test_run_provenance_roundtrip_preserves_environment_fields():
    provenance = RunProvenance(
        git_commit="abc123",
        hardware_key="b200",
        profile_name="minimal",
        timestamp="2026-03-11T00:00:00Z",
        iterations=20,
        warmup_iterations=5,
        execution_environment="virtualized",
        validity_profile="strict",
        dmi_product_name="Standard PC (Q35 + ICH9, 2009)",
    )

    restored = RunProvenance.from_dict(provenance.to_dict())

    assert restored.execution_environment == "virtualized"
    assert restored.validity_profile == "strict"
    assert restored.dmi_product_name == "Standard PC (Q35 + ICH9, 2009)"


def test_run_provenance_matches_rejects_environment_mismatch():
    stored = RunProvenance(
        git_commit="abc123",
        hardware_key="b200",
        profile_name="minimal",
        timestamp="2026-03-10T00:00:00Z",
        iterations=20,
        warmup_iterations=5,
        execution_environment="bare_metal",
        validity_profile="strict",
    )
    observed = RunProvenance(
        git_commit="abc123",
        hardware_key="b200",
        profile_name="minimal",
        timestamp="2026-03-11T00:00:00Z",
        iterations=20,
        warmup_iterations=5,
        execution_environment="virtualized",
        validity_profile="strict",
    )

    assert observed.matches(stored) is False
    assert observed.mismatch_fields(stored) == ["execution_environment"]
