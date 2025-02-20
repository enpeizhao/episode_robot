# add directory to sys.path
import sys
from episode_controller.episodeServer import MotorControlServer
import signal
import argparse
def signal_handler(sig, frame):
    """信号处理函数，捕获 CTRL+C (SIGINT) 信号并安全关闭服务器"""
    print("\n捕获到信号，正在退出...")
    server.stop_server()
    sys.exit(0)

# 注册信号处理函数，处理 CTRL+C
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    # initialize argument parser
    parser = argparse.ArgumentParser(description="Motor Control Server")
    parser.add_argument("--resume", type=bool, default=False, help="Resume previous session")
    parser.add_argument("--usb_id", type=int, default=1, help="USB handler ID, use pcaninfo to check")
    parser.add_argument("--ip", type=str, default="localhost", help="Server IP address")
    parser.add_argument("--port", type=int, default=12345, help="Server port")

    args = parser.parse_args()
    # start the server
    # [100, 45, 100, 100, 100, 5]
    print(f"using usb_id: {args.usb_id}, resume: {args.resume}, ip: {args.ip}, port: {args.port}")
    server = MotorControlServer(resume=args.resume, usb_id=args.usb_id)
    server.start_server(host=args.ip, port=args.port)
    