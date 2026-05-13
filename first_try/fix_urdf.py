"""
Fix package:// paths in ur10.urdf so SAPIEN/ManiSkill can load it.

Usage:
    python fix_urdf.py --urdf ur10.urdf --meshes_dir meshes/ur10

This will produce ur10_fixed.urdf in the same folder as the input URDF.
The meshes folder should be structured as:
    meshes/ur10/visual/   <- .dae files
    meshes/ur10/collision/ <- .stl files
"""

import argparse
import os

def fix_urdf(urdf_path: str, meshes_dir: str):
    urdf_path = os.path.abspath(urdf_path)
    meshes_dir = os.path.abspath(meshes_dir)

    with open(urdf_path, "r") as f:
        content = f.read()

    # Replace package://ur_description/meshes/ur10 with the actual meshes path
    fixed = content.replace("package://ur_description/meshes/ur10", meshes_dir)

    out_path = os.path.join(os.path.dirname(urdf_path), "ur10_fixed.urdf")
    with open(out_path, "w") as f:
        f.write(fixed)

    print(f"Fixed URDF saved to: {out_path}")
    print(f"Mesh paths now point to: {meshes_dir}")

    # Verify all referenced mesh files exist
    print("\nChecking mesh files exist...")
    import re
    mesh_refs = re.findall(r'filename="([^"]+)"', fixed)
    missing = []
    for ref in mesh_refs:
        if not os.path.exists(ref):
            missing.append(ref)

    if missing:
        print(f"\n  WARNING: {len(missing)} mesh file(s) not found:")
        for m in missing:
            print(f"    - {m}")
        print("\n  Make sure you ran:")
        print("    git clone https://github.com/ros-industrial/universal_robot.git")
        print("    mkdir -p meshes/ur10")
        print("    cp -r universal_robot/ur_description/meshes/ur10 meshes/")
    else:
        print(f"  All {len(mesh_refs)} mesh files found. You're good to go!")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", required=True, help="Path to ur10.urdf")
    p.add_argument("--meshes_dir", required=True,
                   help="Absolute path to the ur10 meshes folder (e.g. /home/user/meshes/ur10)")
    args = p.parse_args()
    fix_urdf(args.urdf, args.meshes_dir)
