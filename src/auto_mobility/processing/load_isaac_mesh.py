#!/usr/bin/env python3
"""
Isaac Sim Standalone Mesh Loader Script
Load 3D Mesh (.obj, .ply, .usd, .usda) generated from Open3D into Isaac Sim stage.
Applies lighting, ground plane, and physical collision boundaries (Triangle Mesh Collider).
"""

import sys
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Load 3D Mesh into NVIDIA Isaac Sim")
    parser.add_argument("mesh", help="Path to 3D mesh file (.obj, .ply, .usd, .usda)")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim in headless mode (no GUI)")
    parser.add_argument("--no-physics", action="store_true", help="Disable physical collision API on the mesh")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for the imported mesh (default: 1.0)")
    parser.add_argument("--prim-path", default="/World/DigitalTwinMesh", help="USD Prim Path for the imported mesh")
    return parser.parse_args()

def main():
    args = parse_args()

    mesh_file = os.path.abspath(args.mesh)
    if not os.path.exists(mesh_file):
        print(f"❌ Error: Mesh file does not exist: {mesh_file}")
        sys.exit(1)

    print(f"🚀 Initializing Isaac Sim SimulationApp...")
    print(f" 📁 Input Mesh: {mesh_file}")
    print(f" 🖥️ Headless Mode: {args.headless}")

    # Launch Omniverse Isaac Sim App
    try:
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({"headless": args.headless})
    except ImportError:
        print("❌ Error: Isaac Sim python environment not detected!")
        print("💡 Tip: Run this script using Isaac Sim's bundled python runner (e.g., ./python.sh load_isaac_mesh.py)")
        sys.exit(1)

    import omni.usd
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf

    print("🛠️ Setting up World & Stage...")
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # 1. Add Default Ground Plane
    world.scene.add_default_ground_plane()

    # 2. Add Dome Light for scene illumination
    dome_light_path = "/World/DomeLight"
    dome_light = stage.DefinePrim(dome_light_path, "DomeLight")
    dome_light.GetAttribute("inputs:intensity").Set(1000.0)

    # 3. Add Reference Mesh to Stage
    prim_path = args.prim_path
    print(f"📥 Loading Mesh into USD Stage at: {prim_path}")
    add_reference_to_stage(usd_path=mesh_file, prim_path=prim_path)

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"❌ Error: Failed to load mesh to prim path {prim_path}")
        simulation_app.close()
        sys.exit(1)

    # Apply Scale if needed
    if args.scale != 1.0:
        xform = UsdGeom.Xformable(prim)
        xform.AddScaleOp().Set(Gf.Vec3f(args.scale, args.scale, args.scale))

    # 4. Apply Collision Physics if requested
    if not args.no_physics:
        print("⚡ Applying Collision Physics (PhysxTriangleMeshCollision)...")
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_collision = PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(prim)
        mesh_collision.CreateApproximationAttribute().Set("none") # exact triangle mesh collision

    # 5. Reset and Run Simulation Loop
    world.reset()
    print("==========================================================")
    print(" ✅ Digital Twin Mesh successfully loaded into Isaac Sim!")
    print(" 🕹️ Press Ctrl+C in terminal or close window to exit.")
    print("==========================================================")

    try:
        while simulation_app.is_running():
            world.step(render=True)
    except KeyboardInterrupt:
        print("\nStopping Isaac Sim...")
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
