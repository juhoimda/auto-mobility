import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage

def main():
    rclpy.init()
    node = Node('test_sub')
    qos = QoSProfile(
        depth=5,
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        durability=DurabilityPolicy.VOLATILE,
    )
    def cb(msg):
        print(f"RECEIVED MSG! size={len(msg.data)}")

    node.create_subscription(CompressedImage, '/camera/camera/color/image_raw/compressed', cb, qos)
    print("Spinning 5 sec...")
    import time
    t0 = time.time()
    while time.time() - t0 < 5.0:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
