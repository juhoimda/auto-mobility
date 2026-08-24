#!/usr/bin/env python3
"""WSL IP 및 Windows 호스트 IP를 감지하여 CycloneDDS XML 설정 파일을 자동으로 동기화한다.
"""
import os
import re
import subprocess
import sys

def get_network_ips():
    # 1. WSL eth0 IP
    wsl_ip = None
    try:
        res = subprocess.run(["ip", "-4", "addr", "show", "eth0"], capture_output=True, text=True, timeout=2)
        m = re.search(r'inet\s+(\d+\.\d+\.\d+\.\d+)', res.stdout)
        if m:
            wsl_ip = m.group(1)
    except Exception:
        pass

    # 2. Windows Gateway IP
    win_ip = None
    try:
        res = subprocess.run(["ip", "route"], capture_output=True, text=True, timeout=2)
        m = re.search(r'default via\s+(\d+\.\d+\.\d+\.\d+)', res.stdout)
        if m:
            win_ip = m.group(1)
    except Exception:
        pass

    return wsl_ip, win_ip

def update_xml(path, local_ip, peer_ip, buf_mb=4):
    if not local_ip or not peer_ip:
        return False

    content = f'''<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config">
    <Domain id="any">
        <General>
            <Interfaces>
                <NetworkInterface address="{local_ip}" />
            </Interfaces>
            <AllowMulticast>true</AllowMulticast>
        </General>
        <Internal>
            <SocketReceiveBufferSize min="{buf_mb * 1048576}B" />
            <SocketSendBufferSize min="{buf_mb * 1048576}B" />
            <Watermarks>
                <WhcHigh>{buf_mb}MB</WhcHigh>
            </Watermarks>
        </Internal>
        <Discovery>
            <Peers>
                <Peer Address="{peer_ip}" />
            </Peers>
        </Discovery>
    </Domain>
</CycloneDDS>
'''
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # 내용이 다를 때만 쓰기
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                if f.read().strip() == content.strip():
                    return True
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"[sync_dds] Error writing {path}: {e}", file=sys.stderr)
        return False

def main():
    wsl_ip, win_ip = get_network_ips()
    if not wsl_ip or not win_ip:
        return 0

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    
    # 1. WSL DDS 설정 갱신
    wsl_dds = os.path.join(project_dir, "config", "dds", "cyclonedds_camera.xml")
    update_xml(wsl_dds, local_ip=wsl_ip, peer_ip=win_ip, buf_mb=4)

    # 2. Windows DDS 설정 갱신 (/mnt/c/ros2_humble/cyclonedds.xml)
    win_dds = "/mnt/c/ros2_humble/cyclonedds.xml"
    if os.path.exists("/mnt/c/ros2_humble"):
        update_xml(win_dds, local_ip=win_ip, peer_ip=wsl_ip, buf_mb=16)

    return 0

if __name__ == '__main__':
    sys.exit(main())
