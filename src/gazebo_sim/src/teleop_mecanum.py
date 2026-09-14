#! /usr/bin/env python3

import sys
import termios
import tty
import select
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# --- 按鍵功能說明 ---
msg = """
========================================
   Mecanum Robot Keyboard Controller
========================================
移動控制 (Moving):
   u    i    o      (左平移   前進   右平移)
   j    k    l      (原地左轉 停止   原地右轉)
   m    ,    .      (左後     後退   右後)

i : 前進 (Forward)
, : 後退 (Backward)
j : 原地左轉 (Rotate Left)
l : 原地右轉 (Rotate Right)
u : 向左橫移 (Strafe Left)
o : 向右橫移 (Strafe Right)
k : 停止 (Stop)

速度調整:
q : 增加速度 (+10%)
z : 減少速度 (-10%)

按 Ctrl-C 離開程式
========================================
"""

# 定義按鍵對應的動作 (x, y, theta)
# x: 前後, y: 左右橫移, theta: 旋轉
mecanumBindings = {
    'i': (0.5, 0.0, 0.0),   # 前
    ',': (-0.5, 0.0, 0.0),  # 後
    'j': (0.0, 0.0, 0.5),   # 左轉
    'l': (0.0, 0.0, -0.5),  # 右轉
    'u': (0.0, 0.5, 0.0),   # 左橫移
    'o': (0.0, -0.5, 0.0),  # 右橫移
    'k': (0.0, 0.0, 0.0),   # 停
}

# 速度調整按鍵
speedBindings = {
    'q': 1.1,
    'z': 0.9,
}

def getKey(settings):
    """讀取鍵盤輸入"""
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def main():
    settings = termios.tcgetattr(sys.stdin)

    rclpy.init()
    # 建立節點
    node = rclpy.create_node('teleop_mecanum')
    # 建立發布者，發布到 /cmd_vel
    pub = node.create_publisher(Twist, 'cmd_vel', 10)

    speed = 0.5 # 初始速度 (m/s)
    turn = 1.0  # 初始旋轉速度係數
    
    x = 0.0
    y = 0.0
    th = 0.0
    status = 0.0

    try:
        print(msg)
        while True:
            key = getKey(settings)
            
            # --- 顯示按下的鍵 (讓你知道程式有收到) ---
            if key and key != '\x03':
                print(f"\r按下按鍵: {key}   ", end="", flush=True)
            
            # 判斷移動指令
            if key in mecanumBindings.keys():
                x = mecanumBindings[key][0]
                y = mecanumBindings[key][1]
                th = mecanumBindings[key][2]
                
                # 如果是 'k' (停止)，強制歸零
                if key == 'k':
                    x = 0.0
                    y = 0.0
                    th = 0.0
                    print("\n[!] 停止指令發送")

            # 判斷速度調整
            elif key in speedBindings.keys():
                speed = speed * speedBindings[key]
                print(f"\n當前速度係數: {speed:.2f}")
                status = (status + 1) % 15
                
            # 離開程式 (Ctrl + C)
            elif key == '\x03': 
                print("\n結束程式")
                break
            
            # 發布 Twist 訊息
            twist = Twist()
            twist.linear.x = x * speed
            twist.linear.y = y * speed  # Mecanum 的橫移關鍵
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = th * turn
            pub.publish(twist)

    except Exception as e:
        print(e)

    finally:
        # 結束前發送停止指令
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.angular.z = 0.0
        pub.publish(twist)

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()