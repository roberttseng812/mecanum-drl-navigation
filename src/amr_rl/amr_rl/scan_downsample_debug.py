#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy


def downsample_min_pool(ranges: np.ndarray, n: int) -> np.ndarray:
    """Match AmrRosInterface.get_laser_downsample(): bucket min-pooling."""
    N = int(len(ranges))
    if N <= 0:
        return np.ones((n,), dtype=np.float32)

    edges = np.linspace(0, N, n + 1).astype(np.int32)
    out = np.empty((n,), dtype=np.float32)
    for i in range(n):
        s = int(edges[i])
        e = int(edges[i + 1])
        if e <= s:
            out[i] = float(ranges[min(max(s, 0), N - 1)])
        else:
            out[i] = float(np.min(ranges[s:e]))
    return out


class ScanDownsampleDebug(Node):
    def __init__(self, args):
        super().__init__('scan_downsample_debug')
        self.scan_topic = args.scan_topic
        self.downsample_n = int(args.downsample_n)
        self.print_hz = float(args.print_hz)
        self.use_clamp_rmin = bool(args.clamp_range_min)
        self.collision_threshold = float(args.collision_threshold)

        self._last_scan: Optional[LaserScan] = None
        self._cb_count = 0
        self._last_print_wall = time.time()
        qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT
)

        self.create_subscription(LaserScan,
    self.scan_topic,
    self._cb,
    qos)
        period = 1.0 / max(0.1, self.print_hz)
        self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"Listening {self.scan_topic}, downsample_n={self.downsample_n}, print_hz={self.print_hz},"
            f"clamp_range_min={self.use_clamp_rmin}, collision_threshold={self.collision_threshold}"
        )

    def _cb(self, msg: LaserScan):
        self._last_scan = msg
        self._cb_count += 1

    def _sanitize(self, msg: LaserScan) -> Tuple[np.ndarray, float, float]:
        rmax = float(msg.range_max)
        rmin = float(msg.range_min)
        ranges = np.array(msg.ranges, dtype=np.float32)

        # Standard sanitize
        ranges[np.isinf(ranges)] = rmax
        ranges[np.isnan(ranges)] = rmax

        # Many drivers/sims output 0.0 (or < range_min) for invalid.
        # If clamp_range_min=True, treat those as invalid and map to rmax to avoid false collision.
        if self.use_clamp_rmin:
            invalid = (ranges <= 0.0) | (ranges < rmin)
            ranges[invalid] = rmax

        return ranges, rmin, rmax

    def _min_with_angle(self, msg: LaserScan, ranges: np.ndarray) -> Tuple[float, int, float]:
        idx = int(np.argmin(ranges))
        r = float(ranges[idx])
        ang = float(msg.angle_min + idx * msg.angle_increment)
        return r, idx, ang

    def _on_timer(self):
        msg = self._last_scan
        if msg is None:
            return

        ranges_raw = np.array(msg.ranges, dtype=np.float32)
        rmin = float(msg.range_min)
        rmax = float(msg.range_max)

        # counts for diagnosing invalid patterns
        n_total = int(len(ranges_raw))
        n_zero_or_neg = int(np.sum(ranges_raw <= 0.0))
        n_lt_rmin = int(np.sum((ranges_raw > 0.0) & (ranges_raw < rmin)))
        n_inf = int(np.sum(np.isinf(ranges_raw)))
        n_nan = int(np.sum(np.isnan(ranges_raw)))

        ranges, rmin2, rmax2 = self._sanitize(msg)
        assert abs(rmin2 - rmin) < 1e-6 and abs(rmax2 - rmax) < 1e-6

        # downsample to match env/ros interface
        down = downsample_min_pool(ranges, self.downsample_n)

        raw_min, raw_idx, raw_ang = self._min_with_angle(msg, ranges)
        down_min = float(np.min(down))

        # percentiles help detect single-ray spikes
        p1 = float(np.percentile(ranges, 1))
        p5 = float(np.percentile(ranges, 5))

        # collision decisions
        would_collide_raw = int(raw_min < self.collision_threshold)
        would_collide_down = int(down_min < self.collision_threshold)

        now = time.time()
        age = now - self._last_print_wall
        self._last_print_wall = now

        print(
            f"[SCAN] cb={self._cb_count} n={n_total} rmin={rmin:.3f} rmax={rmax:.3f}"
            f"invalid(zero/neg={n_zero_or_neg}, <rmin={n_lt_rmin}, inf={n_inf}, nan={n_nan})"
            f"min={raw_min:.3f} (idx={raw_idx}, ang={raw_ang:+.2f}rad)"
            f"p1={p1:.3f} p5={p5:.3f} down_min={down_min:.3f}"
            f"collide(min)={would_collide_raw} collide(down)={would_collide_down}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan-topic', default='/scan', help='LaserScan topic')
    ap.add_argument('--downsample-n', type=int, default=120, help='Downsample bins (match AmrRosInterface)')
    ap.add_argument('--print-hz', type=float, default=5.0, help='Print frequency (Hz)')
    ap.add_argument('--clamp-range-min', type=int, default=1, help='1: treat <=0 or <range_min as invalid (map to range_max)')
    ap.add_argument('--collision-threshold', type=float, default=0.57, help='Collision threshold to test')
    args = ap.parse_args()

    rclpy.init()
    node = ScanDownsampleDebug(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
