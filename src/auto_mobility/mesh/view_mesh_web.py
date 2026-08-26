"""
WSL CPU 환경 근본 해결용 웹 뷰어 - Windows 브라우저 GPU로 렌더링 위임

기존 view_mesh.py:10-15 가 llvmpipe 강제 + matplotlib mplot3d(소프트웨어 래스터)라
175만 정점(267MB) 모델에서 60fps 불가.

이 스크립트는 WSL에서 three.js HTML + glb 를 생성하고 http.server 로 서빙,
Windows 브라우저가 WebGL로 렌더링하므로 WSL X11 오버헤드 0.

Usage:
  python3 src/auto_mobility/mesh/view_mesh_web.py ros2_data/meshes/base3_rtab_reconstructed.obj
  python3 src/auto_mobility/mesh/view_mesh_web.py ros2_data/meshes/base3_rtab_reconstructed.obj --decimate 300000 --port 8000
"""
import argparse
import http.server
import os
import socket
import sys
import threading
import webbrowser

try:
    import open3d as o3d
except ImportError:
    print("open3d==0.19.0 필요: pip install open3d==0.19.0")
    sys.exit(1)
import numpy as np


def find_free_port(start=8000):
    for p in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", p))
                return p
            except OSError:
                continue
    return start


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<style>
  html,body{{margin:0;height:100%;overflow:hidden;background:#0a0a0a}}
  canvas{{display:block}}
  #info{{position:fixed;top:10px;left:10px;color:#ccc;font-family:monospace;font-size:13px;background:rgba(0,0,0,0.6);padding:8px 10px;border-radius:6px}}
  #help{{position:fixed;bottom:10px;left:10px;color:#888;font-family:monospace;font-size:12px}}
</style>
<script type="importmap">
{{
  "imports": {{
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }}
}}
</script>
</head>
<body>
<div id="info">{title} — {stats}<br><span style="color:#888">{hint}</span></div>
<div id="help">마우스 드래그: 회전 | 휠: 줌 | 우클릭 드래그: 이동 | 더블클릭: 리셋</div>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0a);
const camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.1, 1000);
camera.position.set(8, -8, 5);
const renderer = new THREE.WebGLRenderer({{antialias:true}});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;

scene.add(new THREE.HemisphereLight(0xffffff, 0x222222, 1.2));
const dir = new THREE.DirectionalLight(0xffffff, 1.0);
dir.position.set(5,5,10);
scene.add(dir);
scene.add(new THREE.GridHelper(10, 10, 0x333333, 0x222222));

const loader = new GLTFLoader();
loader.load('./model.glb', (gltf) => {{
  const obj = gltf.scene;
  // auto-center & scale
  const box = new THREE.Box3().setFromObject(obj);
  const center = box.getCenter(new THREE.Vector3());
  obj.position.sub(center);
  const size = box.getSize(new THREE.Vector3()).length();
  if (size > 10) {{ const s = 10/size; obj.scale.setScalar(s); }}
  scene.add(obj);
  document.getElementById('info').innerHTML += '<br><span style="color:#4af">✓ GPU(WebGL) 렌더링</span>';
}}, undefined, (e) => {{
  document.getElementById('info').innerHTML = 'GLB 로드 실패: '+e.message;
}});

addEventListener('resize', ()=>{{
  camera.aspect = innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}});
renderer.domElement.addEventListener('dblclick', ()=>{{ controls.reset(); }});
(function animate(){{ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }})();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="WSL 근본 해결 - 브라우저 GPU 웹 뷰어")
    ap.add_argument("input", help=".obj/.ply/.stl 경로")
    ap.add_argument("--port", type=int, default=0, help="포트 (0=자동)")
    ap.add_argument("--decimate", type=int, default=0, help="면 수 줄이기 목표 (예: 300000, 0=원본 유지)")
    ap.add_argument("--out_dir", default="/tmp/mesh_web_viewer", help="HTML/GLB 출력 폴더")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"파일 없음: {args.input}")
        sys.exit(1)

    print(f"📂 로드 중: {args.input}")
    mesh = o3d.io.read_triangle_mesh(args.input)
    if mesh.is_empty() or len(mesh.vertices) == 0:
        print("mesh 로드 실패, pointcloud 시도")
        pcd = o3d.io.read_point_cloud(args.input)
        if pcd.is_empty():
            print("로드 실패")
            sys.exit(1)
        # pcd -> mesh 없음, 그냥 pcd를 glb로 쓰기 위해 더미 mesh 생성 불가 -> pcd 전용 처리
        # pcd는 glb pointcloud로 저장 안되므로 임시 ply로
        mesh = None
        vertices = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        print(f"  points: {len(vertices):,}")
    else:
        print(f"  vertices: {len(mesh.vertices):,}, triangles: {len(mesh.triangles):,}")
        if args.decimate and len(mesh.triangles) > args.decimate:
            print(f"  🔧 Decimate {len(mesh.triangles):,} -> {args.decimate:,} ...")
            mesh = mesh.simplify_quadric_decimation(args.decimate)
            mesh.remove_duplicated_vertices()
            mesh.remove_degenerate_triangles()
            print(f"  -> {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")
        # vertex color 없으면 xray 대신 단색
        if not mesh.has_vertex_colors():
            mesh.paint_uniform_color([0.7, 0.7, 0.7])
        mesh.compute_vertex_normals()

    os.makedirs(args.out_dir, exist_ok=True)
    glb_path = os.path.join(args.out_dir, "model.glb")
    html_path = os.path.join(args.out_dir, "index.html")

    if mesh is not None:
        # Open3D 0.19는 glb 저장 지원
        ok = o3d.io.write_triangle_mesh(glb_path, mesh, write_ascii=False)
        if not ok:
            # fallback: ply
            glb_path = os.path.join(args.out_dir, "model.ply")
            o3d.io.write_triangle_mesh(glb_path, mesh)
            print("glb 저장 실패, ply로 대체 (three.js PLYLoader 필요 - 일단 glb 재시도 요망)")
            sys.exit(1)
        stats = f"{len(mesh.vertices):,} verts / {len(mesh.triangles):,} tris"
    else:
        # pointcloud path - pcd를 ply로 저장하고 html에서 Points로 렌더링은 별도 로더 필요
        # 간단히 open3d로 ply 저장
        import open3d as o3d2
        pcd2 = o3d2.io.read_point_cloud(args.input)
        o3d2.io.write_point_cloud(glb_path.replace('.glb','.ply'), pcd2)
        stats = f"{len(vertices):,} points"
        print("PointCloud는 현재 mesh 경로만 완전 지원 - ply 확인:", glb_path)

    html = HTML_TEMPLATE.format(
        title=os.path.basename(args.input),
        stats=stats,
        hint="Windows 브라우저 GPU 렌더링 (WSL X11 미사용)",
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    port = args.port if args.port else find_free_port(8000)
    os.chdir(args.out_dir)

    # WSL에서 Windows 브라우저 열기 시도
    url = f"http://localhost:{port}/"
    print("=" * 60)
    print(f"✅ 서빙 준비: {html_path}")
    print(f"   GLB: {glb_path} ({os.path.getsize(glb_path)/1024/1024:.1f} MB)")
    print(f"🌐 브라우저에서 열기: {url}")
    print(f"   느리면 --decimate 300000 추가")
    print("=" * 60)

    # 백그라운드로 브라우저 열기 시도 (wslview / explorer.exe / xdg-open)
    def try_open():
        for cmd in [f'wslview {url}', f'explorer.exe {url}', f'xdg-open {url}']:
            try:
                os.system(cmd + " >/dev/null 2>&1 &")
                break
            except Exception:
                pass
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Timer(0.8, try_open).start()

    handler = http.server.SimpleHTTPRequestHandler
    with http.server.ThreadingHTTPServer(("", port), handler) as httpd:
        print(f"Serving on port {port} ... Ctrl+C 종료")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n종료")

if __name__ == "__main__":
    main()
