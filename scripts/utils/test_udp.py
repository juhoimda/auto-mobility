import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(2.0)
try:
    sock.sendto(b"PING", ("172.19.144.1", 7400))
    print("Sent UDP packet to Windows on 7400")
except Exception as e:
    print(f"Error: {e}")
