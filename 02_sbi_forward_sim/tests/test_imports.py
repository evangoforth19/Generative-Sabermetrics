def test_import_schema():
    from sbi_forward_sim.src import schema

    assert len(schema.G_COLUMNS) > 0
    assert schema.Z_TARGET_COLUMNS == ["x", "psi_deg", "e_y_star", "theta_deg"]
    assert "theta_deg" in schema.Z_TARGET_COLUMNS


def test_import_stage_z_modules():
    from sbi_forward_sim.src import calibration_z, data_z, eval_z, models_z, train_z  # noqa: F401
