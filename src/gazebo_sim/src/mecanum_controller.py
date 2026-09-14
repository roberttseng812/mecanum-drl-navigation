#! /usr/bin/env python3

import traceback
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Twist, TransformStamped, Quaternion
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import numpy as np
import math

def quaternion_from_euler(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return Quaternion(x=qx, y=qy, z=qz, w=qw)

def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def yaw_from_quaternion(q: Quaternion) -> float:
    # ROS uses x,y,z,w
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

class MecanumController(Node):
    def __init__(self):
        super().__init__('mecanum_controller')

        self.declare_parameter('amr_wheel_width', 0.0)
        self.declare_parameter('amr_wheel_length', 0.0)
        self.declare_parameter('wheel_diameter', 0.0)
        # --- Odom tuning ---
        self.declare_parameter('use_jointstate_stamp', True)
        self.declare_parameter('use_imu_yaw', True)
        self.declare_parameter('vel_filter_alpha', 0.25)  # 0..1 (higher = less smoothing)
        self.declare_parameter('vy_deadband', 0.003)       # m/s
        self.declare_parameter('wz_deadband', 0.003)       # rad/s
        self.declare_parameter('wheel_feedback_signs', [1.0, 1.0, 1.0, 1.0])


        self.target_vel = np.matrix([0.0, 0.0, 0.0])
        self.current_vel = np.matrix([0.0, 0.0, 0.0])

        # accel limits are per-second (m/s^2, rad/s^2)
        self.declare_parameter('lin_accel_limit', 0.3)
        self.declare_parameter('ang_accel_limit', 0.3)
        self.control_period = 0.05
        self.create_timer(self.control_period, self.control_loop)

        try:
            self.setParam()
        except Exception as e:
            self.get_logger().error(f"Failed to initialize parameters: {e}")
            raise e

        # JointGroupVelocityController 訂閱 commands
        self.fr_pub = self.create_publisher(Float64MultiArray, 'front_right_controller/commands', 10)
        self.fl_pub = self.create_publisher(Float64MultiArray, 'front_left_controller/commands', 10)
        self.rr_pub = self.create_publisher(Float64MultiArray, 'rear_right_controller/commands', 10)
        self.rl_pub = self.create_publisher(Float64MultiArray, 'rear_left_controller/commands', 10)

        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.vel_sub = self.create_subscription(Twist, 'cmd_vel', self.cmdVelCB, 10)
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.jointStateCB, 10)
        
        # 里程計變數
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        
        # === [修改] 初始化時間控制變數 ===
        # last_time 設為 None，確保第一筆資料進來時只記錄時間不計算位移
        self.last_time = None
        
        # 記錄節點啟動時間，用於忽略剛啟動時物理引擎不穩定的狀態 (落地效應)
        self.node_start_time = self.get_clock().now()

        # velocity smoothing state
        self.vx_f = 0.0
        self.vy_f = 0.0
        self.wz_f = 0.0

        # wheel feedback sign (calibration knob)
        self.wheel_feedback_signs = np.array(self.get_parameter('wheel_feedback_signs').value, dtype=float)
        if self.wheel_feedback_signs.shape[0] != 4:
            self.get_logger().warn('wheel_feedback_signs must have 4 elements; fallback to [1,1,1,1]')
            self.wheel_feedback_signs = np.ones(4, dtype=float)

        # IMU (optional): improves heading -> improves lateral odom
        self.imu_yaw = None
        self.imu_sub = None
        if bool(self.get_parameter('use_imu_yaw').value):
            self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imuCB, 50)
        
        self.wheel_joint_names = [
            'front_right_wheel_joint', 
            'front_left_wheel_joint', 
            'rear_right_wheel_joint', 
            'rear_left_wheel_joint'
        ]
        
        self.get_logger().info('Mecanum Controller (Fixed Type) Started.')

    def imuCB(self, msg: Imu):
        # Use IMU orientation yaw as heading reference
        try:
            self.imu_yaw = yaw_from_quaternion(Quaternion(
                x=msg.orientation.x,
                y=msg.orientation.y,
                z=msg.orientation.z,
                w=msg.orientation.w,
            ))
        except Exception:
            # keep last value
            return

    def setParam(self):
        amr_wheel_width = self.get_parameter('amr_wheel_width').get_parameter_value().double_value
        amr_wheel_length = self.get_parameter('amr_wheel_length').get_parameter_value().double_value
        amr_wheel_diameter = self.get_parameter('wheel_diameter').get_parameter_value().double_value

        if amr_wheel_width == 0.0:
            raise Exception('Parameters not set correctly')

        self.radius = amr_wheel_diameter / 2.0
        lx = amr_wheel_length / 2.0 
        ly = amr_wheel_width / 2.0
        sum_l = lx + ly

        self.mat = np.matrix([
            [1,  1,  sum_l], # FR
            [1, -1, -sum_l], # FL
            [1, -1,  sum_l], # RR
            [1,  1, -sum_l]  # RL
        ])

        self.forward_mat = np.linalg.pinv(self.mat)

    def cmdVelCB(self, data):
        self.target_vel = np.matrix([data.linear.x, data.linear.y, data.angular.z])

    def publish_wheel_cmd(self, pub, linear_vel):
        msg = Float64MultiArray()
        msg.data = [linear_vel / self.radius] 
        pub.publish(msg)

    def jointStateCB(self, msg):
        # Prefer joint_states timestamp to reduce dt jitter; fallback to node clock
        if bool(self.get_parameter('use_jointstate_stamp').value) and hasattr(msg, 'header') and msg.header.stamp.sec != 0:
            current_time = Time.from_msg(msg.header.stamp)
        else:
            current_time = self.get_clock().now()

        # === [新增] 1. 等待物理穩定 ===
        # 忽略前 2.0 秒的數據，讓機器人落地穩定，避免落地滑動被計入里程計
        time_since_start = (current_time - self.node_start_time).nanoseconds / 1e9
        if time_since_start < 2.0:
            return

        # === [修改] 2. 處理第一次回呼 ===
        # 如果是第一次執行 (last_time 為 None)，只更新時間，不計算 dt
        if self.last_time is None:
            self.last_time = current_time
            return

        # 嚴格檢查時間是否推進 (避免同一時間戳重複計算)
        if current_time.nanoseconds <= self.last_time.nanoseconds:
            return

        current_wheel_vels = np.zeros(4)
        try:
            for i, name in enumerate(self.wheel_joint_names):
                if name in msg.name:
                    idx = msg.name.index(name)
                    current_wheel_vels[i] = msg.velocity[idx] * self.radius * self.wheel_feedback_signs[i]
                else:
                    return
        except ValueError:
            return

        robot_vel = np.dot(self.forward_mat, current_wheel_vels)
        vx, vy, wz = robot_vel[0, 0], robot_vel[0, 1], robot_vel[0, 2]
        # convert to float (avoid numpy.matrix scalars leaking into math)
        vx, vy, wz = float(vx), float(vy), float(wz)

        # deadband + low-pass filter (vy is usually the noisiest for mecanum)
        vy_db = float(self.get_parameter('vy_deadband').value)
        wz_db = float(self.get_parameter('wz_deadband').value)
        if abs(vy) < vy_db:
            vy = 0.0
        if abs(wz) < wz_db:
            wz = 0.0

        alpha = float(self.get_parameter('vel_filter_alpha').value)
        alpha = max(0.0, min(1.0, alpha))
        self.vx_f = (1.0 - alpha) * self.vx_f + alpha * vx
        self.vy_f = (1.0 - alpha) * self.vy_f + alpha * vy
        self.wz_f = (1.0 - alpha) * self.wz_f + alpha * wz
        vx, vy, wz = self.vx_f, self.vy_f, self.wz_f

        # 計算 dt
        dt = (current_time - self.last_time).nanoseconds / 1e9
        
        # 更新 last_time
        self.last_time = current_time

        # 計算位移 (mid-point integration reduces x/y error when rotating)
        th0 = float(self.th)

        use_imu = bool(self.get_parameter('use_imu_yaw').value) and (self.imu_yaw is not None)
        if use_imu:
            # unwrap IMU yaw to be continuous w.r.t current th
            th1 = th0 + wrap_to_pi(float(self.imu_yaw) - th0)
            delta_th = wrap_to_pi(th1 - th0)
        else:
            delta_th = wz * dt

        th_mid = th0 + 0.5 * delta_th
        delta_x = (vx * math.cos(th_mid) - vy * math.sin(th_mid)) * dt
        delta_y = (vx * math.sin(th_mid) + vy * math.cos(th_mid)) * dt

        self.x += delta_x
        self.y += delta_y
        self.th = wrap_to_pi(th0 + delta_th)

        q = quaternion_from_euler(0, 0, self.th)
        
        # 發布 TF
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation = q
        self.tf_broadcaster.sendTransform(t)

        # 發布 Odom
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation = q
        odom.twist.twist.linear.x = float(vx)
        odom.twist.twist.linear.y = float(vy)
        odom.twist.twist.angular.z = float(wz)
        self.odom_pub.publish(odom)

    def control_loop(self):
        # 簡單的加速度限制算法 (limits are per-second; scaled by control_period)
        diff = self.target_vel - self.current_vel
        lin_acc = float(self.get_parameter('lin_accel_limit').value)
        ang_acc = float(self.get_parameter('ang_accel_limit').value)
        max_step = np.matrix([
            lin_acc * self.control_period,
            lin_acc * self.control_period,
            ang_acc * self.control_period,
        ]).T
        # clip element-wise
        diff = np.maximum(np.minimum(diff.T, max_step), -max_step).T
        self.current_vel += diff
        
        wheel_linear_vel = (np.dot(self.mat, self.current_vel.T).A1).tolist()

        self.publish_wheel_cmd(self.fr_pub, wheel_linear_vel[0])
        self.publish_wheel_cmd(self.fl_pub, wheel_linear_vel[1])
        self.publish_wheel_cmd(self.rr_pub, wheel_linear_vel[2])
        self.publish_wheel_cmd(self.rl_pub, wheel_linear_vel[3])

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MecanumController()
        rclpy.spin(node)
    except Exception as e:
        print(traceback.format_exc())
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
