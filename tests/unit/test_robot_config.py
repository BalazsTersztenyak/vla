"""Unit tests for UR10 agent configuration — no SAPIEN simulation needed."""
from pathlib import Path

import numpy as np
import pytest

from vla.robot.ur10_agent import HOME_QPOS, JOINT_NAMES, URDF_PATH, UR10


class TestURDFAsset:
    def test_urdf_path_is_absolute(self):
        assert Path(URDF_PATH).is_absolute()

    def test_urdf_file_exists(self):
        assert URDF_PATH.exists(), f"URDF not found at {URDF_PATH}"

    def test_urdf_is_xml(self):
        # utf-8-sig strips the BOM that Windows tools sometimes prepend
        content = URDF_PATH.read_text(encoding="utf-8-sig")
        assert content.strip().startswith("<?xml")
        assert "<robot" in content

    def test_mesh_paths_are_relative(self):
        content = URDF_PATH.read_text(encoding="utf-8-sig")
        # Absolute Windows or Unix paths should not be present
        assert "D:/" not in content
        assert "C:/" not in content
        # Should use relative mesh paths
        assert "meshes/ur10/" in content

    def test_mesh_files_exist(self):
        assets_dir = URDF_PATH.parent
        for name in ["base", "shoulder", "upperarm", "forearm", "wrist1", "wrist2", "wrist3"]:
            visual = assets_dir / "meshes" / "ur10" / "visual" / f"{name}.dae"
            collision = assets_dir / "meshes" / "ur10" / "collision" / f"{name}.stl"
            assert visual.exists(), f"Missing visual mesh: {visual}"
            assert collision.exists(), f"Missing collision mesh: {collision}"


class TestUR10JointConfig:
    def test_has_six_joints(self):
        assert len(JOINT_NAMES) == 6

    def test_joint_names_are_standard_ur(self):
        expected = {
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        }
        assert set(JOINT_NAMES) == expected

    def test_home_qpos_shape(self):
        assert HOME_QPOS.shape == (6,)

    def test_home_qpos_dtype(self):
        assert HOME_QPOS.dtype == np.float32

    def test_home_qpos_within_joint_limits(self):
        # UR10 joint limits are roughly ±2π; home should be well within
        assert (np.abs(HOME_QPOS) < 2 * np.pi).all()


class TestUR10AgentClass:
    def test_uid(self):
        assert UR10.uid == "ur10"

    def test_ee_link_name(self):
        assert UR10.ee_link_name == "tool0"

    def test_urdf_path_attribute_matches_module_constant(self):
        assert Path(UR10.urdf_path) == URDF_PATH

    def test_keyframe_rest_qpos_shape(self):
        qpos = UR10.keyframes["rest"].qpos
        assert qpos.shape == (6,)
