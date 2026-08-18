#!/usr/bin/env python3
"""
load_isaac_mesh.py — NVIDIA Isaac Sim 디지털 트윈 Mesh 로더

파이프라인 내 역할: Downstream Consumer (최종 출력 소비)
  - Input:  OBJ/PLY/USD/USDA mesh (reconstruct_tsdf.py 또는 mesh_open3d.py 출력)
  - Action: Isaac Sim Stage에 Mesh 로드 + 조명 + 지면 + 물리 충돌 경계 적용
  - Note:   Mesh 로드 실패(파일 없음·크기 0)는 Isaac Sim 진입 전에 barrier가 차단한다.

Isaac Sim 요구사항 때문에 센서 캡처 또는 SLAM 아키텍처를 변경하지 않는다.
검증된 derived artifact(mesh + coordinate metadata)만 여기서 소비한다.
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

    # File Barrier 1: Check existence and non-zero size
    if not os.path.exists(mesh_file):
        print(f"❌ Barrier Failure: Mesh file does not exist: {mesh_file}")
        sys.exit(1)

    file_size_kb = os.path.getsize(mesh_file) / 1024
    if file_size_kb < 1.0:
        print(f"❌ Barrier Failure: Mesh file is empty or corrupted ({file_size_kb:.2f} KB)")
        sys.exit(1)

    print(f"🚀 Initializing Isaac Sim SimulationApp...")
    print(f" 📁 Input Mesh     : {mesh_file} ({file_size_kb:.1f} KB)")
    print(f" 🖥️ Headless Mode  : {args.headless}")

    # Launch Omniverse Isaac Sim App
    try:
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({"headless": args.headless})
    except ImportError:
        print("❌ Barrier Failure: Isaac Sim python environment not detected!")
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
        print(f"❌ Barrier Failure: Failed to load mesh to prim path {prim_path}")
        simulation_app.close()
        sys.exit(1)

    # Apply Scale if needed
    if args.scale != 1.0:
        xform = UsdGeom.Xformable(prim)
        xform.AddScaleOp().Set(Gf.Vec3f(args.scale, args.scale, args.scale))

    # 4. Apply Collision Physics with Fallback
    if not args.no_physics:
        print("⚡ Applying Collision Physics...")
        try:
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_collision = PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(prim)
            mesh_collision.CreateApproximationAttribute().Set("none") # Exact Triangle Mesh
            print("✅ PhysxTriangleMeshCollision applied successfully!")
        except Exception as e:
            print(f"⚠️ TriangleMeshCollision failed ({e}). Falling back to ConvexHull approximation...")
            try:
                mesh_collision.CreateApproximationAttribute().Set("convexHull")
                print("✅ ConvexHull Physics fallback applied!")
            except Exception as ex:
                print(f"⚠️ Physics approximation warning: {ex}")

    # 5. Reset and Run Simulation Loop
    world.reset()
    print("==========================================================")
    print(" 🎉 [Success] Digital Twin Mesh loaded into Isaac Sim!")
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
