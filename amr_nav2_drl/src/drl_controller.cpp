#include "amr_nav2_drl/drl_controller.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "tf2/utils.h"
#include "tf2/exceptions.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/LinearMath/Quaternion.h"

namespace
{

bool computePathTangentYaw(
  const nav_msgs::msg::Path & path,
  size_t idx,
  double & yaw_out)
{
  if (path.poses.empty()) {
    return false;
  }

  idx = std::min(idx, path.poses.size() - 1);
  const auto & p = path.poses[idx].pose.position;

  for (size_t j = idx + 1; j < path.poses.size(); ++j) {
    const auto & q = path.poses[j].pose.position;
    const double dx = q.x - p.x;
    const double dy = q.y - p.y;
    if (std::hypot(dx, dy) > 1e-6) {
      yaw_out = std::atan2(dy, dx);
      return true;
    }
  }

  for (size_t j = idx; j > 0; --j) {
    const auto & q = path.poses[j - 1].pose.position;
    const double dx = p.x - q.x;
    const double dy = p.y - q.y;
    if (std::hypot(dx, dy) > 1e-6) {
      yaw_out = std::atan2(dy, dx);
      return true;
    }
  }

  return false;
}

void setPoseYaw(geometry_msgs::msg::Pose & pose, double yaw)
{
  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, yaw);
  q.normalize();
  pose.orientation = tf2::toMsg(q);
}

}  // namespace

namespace amr_nav2_drl
{

void DrlController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent.lock();
  if (!node_) {
    throw std::runtime_error("DrlController: parent node is null");
  }

  name_ = std::move(name);
  logger_ = node_->get_logger();
  tf_ = tf;
  costmap_ros_ = costmap_ros;

  node_->declare_parameter(name_ + ".goal_topic", goal_topic_);
  node_->declare_parameter(name_ + ".cmd_topic", cmd_topic_);
  node_->declare_parameter(name_ + ".lookahead_dist", lookahead_dist_);
  node_->declare_parameter(name_ + ".cmd_timeout", cmd_timeout_);
  node_->declare_parameter(name_ + ".nominal_speed", nominal_speed_);
  node_->declare_parameter(name_ + ".transform_tolerance", transform_tolerance_);

  node_->get_parameter(name_ + ".goal_topic", goal_topic_);
  node_->get_parameter(name_ + ".cmd_topic", cmd_topic_);
  node_->get_parameter(name_ + ".lookahead_dist", lookahead_dist_);
  node_->get_parameter(name_ + ".cmd_timeout", cmd_timeout_);
  node_->get_parameter(name_ + ".nominal_speed", nominal_speed_);
  node_->get_parameter(name_ + ".transform_tolerance", transform_tolerance_);

  goal_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
    goal_topic_, rclcpp::SystemDefaultsQoS());

  cmd_sub_ = node_->create_subscription<geometry_msgs::msg::Twist>(
    cmd_topic_, rclcpp::SystemDefaultsQoS(),
    [this](geometry_msgs::msg::Twist::SharedPtr msg)
    {
      std::lock_guard<std::mutex> lk(cmd_mtx_);
      last_cmd_ = *msg;
      last_cmd_time_ = node_->now();
    });

  RCLCPP_INFO(
    logger_,
    "Configured DRL controller '%s' (lookahead=%.2f, timeout=%.2f, tf_tol=%.2f)",
    name_.c_str(), lookahead_dist_, cmd_timeout_, transform_tolerance_);
}

void DrlController::cleanup()
{
  goal_pub_.reset();
  cmd_sub_.reset();
  global_plan_ = nav_msgs::msg::Path();
}

void DrlController::activate()
{
  if (goal_pub_) goal_pub_->on_activate();
}

void DrlController::deactivate()
{
  if (goal_pub_) goal_pub_->on_deactivate();
}

void DrlController::setPlan(const nav_msgs::msg::Path & path)
{
  global_plan_ = path;
}

void DrlController::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  speed_limit_ = speed_limit;
  speed_limit_is_percentage_ = percentage;
}

size_t DrlController::findClosestPoseIndex_(
  const geometry_msgs::msg::PoseStamped & pose) const
{
  if (global_plan_.poses.empty()) return 0;

  const auto & p = pose.pose.position;
  size_t best_i = 0;
  double best_d2 = std::numeric_limits<double>::infinity();

  for (size_t i = 0; i < global_plan_.poses.size(); ++i) {
    const auto & q = global_plan_.poses[i].pose.position;
    const double dx = p.x - q.x;
    const double dy = p.y - q.y;
    const double d2 = dx * dx + dy * dy;
    if (d2 < best_d2) {
      best_d2 = d2;
      best_i = i;
    }
  }
  return best_i;
}

size_t DrlController::findLookaheadPoseIndex_(
  size_t start_index, double lookahead_dist) const
{
  if (global_plan_.poses.empty()) return 0;

  start_index = std::min(start_index, global_plan_.poses.size() - 1);
  double accum = 0.0;

  for (size_t i = start_index; i + 1 < global_plan_.poses.size(); ++i) {
    const auto & a = global_plan_.poses[i].pose.position;
    const auto & b = global_plan_.poses[i + 1].pose.position;
    accum += std::hypot(b.x - a.x, b.y - a.y);
    if (accum >= lookahead_dist) return i + 1;
  }
  return global_plan_.poses.size() - 1;
}

geometry_msgs::msg::TwistStamped DrlController::getLatestCmd_(const rclcpp::Time & stamp)
{
  geometry_msgs::msg::TwistStamped out;
  out.header.stamp = stamp;
  out.header.frame_id = costmap_ros_ ? costmap_ros_->getBaseFrameID() : "base_link";

  geometry_msgs::msg::Twist cmd;
  rclcpp::Time cmd_time{0, 0, RCL_ROS_TIME};

  {
    std::lock_guard<std::mutex> lk(cmd_mtx_);
    cmd = last_cmd_;
    cmd_time = last_cmd_time_;
  }

  if (cmd_time.nanoseconds() == 0) {
    return out;
  }

  if ((stamp - cmd_time).seconds() > cmd_timeout_) {
    RCLCPP_WARN_THROTTLE(
      logger_, *node_->get_clock(), 2000,
      "Latest DRL cmd is stale by %.3f s (timeout %.3f s); commanding stop.",
      (stamp - cmd_time).seconds(), cmd_timeout_);
    return out;
  }

  if (speed_limit_ > 0.0) {
    double scale = 1.0;

    if (speed_limit_is_percentage_) {
      scale = std::clamp(speed_limit_ / 100.0, 0.0, 1.0);
    } else {
      const double linear_speed = std::hypot(cmd.linear.x, cmd.linear.y);
      if (linear_speed > 1e-6 && linear_speed > speed_limit_) {
        scale = speed_limit_ / linear_speed;
      }
    }

    cmd.linear.x *= scale;
    cmd.linear.y *= scale;
    cmd.angular.z *= scale;
  }

  out.twist = cmd;
  return out;
}

geometry_msgs::msg::TwistStamped DrlController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist &,
  nav2_core::GoalChecker *)
{
  const rclcpp::Time stamp =
    (pose.header.stamp.sec == 0 && pose.header.stamp.nanosec == 0)
      ? node_->now()
      : rclcpp::Time(pose.header.stamp);

  const std::string base_frame = costmap_ros_->getBaseFrameID();

  geometry_msgs::msg::TwistStamped stop;
  stop.header.stamp = stamp;
  stop.header.frame_id = base_frame;

  if (global_plan_.poses.empty()) {
    return stop;
  }

  const std::string plan_frame =
    !global_plan_.header.frame_id.empty() ? global_plan_.header.frame_id : pose.header.frame_id;

  geometry_msgs::msg::PoseStamped pose_in_plan = pose;
  pose_in_plan.header.stamp = stamp;

  try {
    if (pose.header.frame_id != plan_frame) {
      tf_->transform(
        pose,
        pose_in_plan,
        plan_frame,
        tf2::durationFromSec(transform_tolerance_));
      pose_in_plan.header.stamp = stamp;
      pose_in_plan.header.frame_id = plan_frame;
    }
  } catch (const tf2::TransformException & ex) {
    try {
      auto tf_latest = tf_->lookupTransform(
        plan_frame,
        pose.header.frame_id,
        tf2::TimePointZero);

      tf2::doTransform(pose, pose_in_plan, tf_latest);
      pose_in_plan.header.frame_id = plan_frame;
      pose_in_plan.header.stamp = stamp;

      RCLCPP_WARN_THROTTLE(
        logger_, *node_->get_clock(), 2000,
        "Pose transform to plan frame failed at stamped time, fallback to latest TF: %s",
        ex.what());

    } catch (const tf2::TransformException & ex2) {
      RCLCPP_WARN_THROTTLE(
        logger_, *node_->get_clock(), 2000,
        "Pose transform to plan frame completely failed: %s | fallback failed: %s",
        ex.what(), ex2.what());
      return stop;
    }
  }

  size_t closest_i = findClosestPoseIndex_(pose_in_plan);
  size_t goal_i = findLookaheadPoseIndex_(closest_i, lookahead_dist_);

  RCLCPP_DEBUG_THROTTLE(
    logger_, *node_->get_clock(), 1000,
    "pose frame=%s, plan frame=%s, closest_i=%zu, goal_i=%zu, plan size=%zu",
    pose.header.frame_id.c_str(),
    plan_frame.c_str(),
    closest_i, goal_i, global_plan_.poses.size());

  geometry_msgs::msg::PoseStamped goal_global = global_plan_.poses[goal_i];
  goal_global.header.stamp = stamp;
  goal_global.header.frame_id = plan_frame;

  const bool is_final = (goal_i >= global_plan_.poses.size() - 1);
  if (!is_final) {
    double yaw = 0.0;
    if (computePathTangentYaw(global_plan_, goal_i, yaw)) {
      setPoseYaw(goal_global.pose, yaw);
    }
  }

  geometry_msgs::msg::PoseStamped goal_base;

  try {
    tf_->transform(
      goal_global,
      goal_base,
      base_frame,
      tf2::durationFromSec(transform_tolerance_));
  } catch (const tf2::TransformException & ex) {
    try {
      auto tf_latest = tf_->lookupTransform(
        base_frame,
        goal_global.header.frame_id,
        tf2::TimePointZero);

      tf2::doTransform(goal_global, goal_base, tf_latest);
      goal_base.header.frame_id = base_frame;
      goal_base.header.stamp = stamp;

      RCLCPP_WARN_THROTTLE(
        logger_, *node_->get_clock(), 2000,
        "Goal transform failed at stamped time, fallback to latest TF: %s",
        ex.what());

    } catch (const tf2::TransformException & ex2) {
      RCLCPP_WARN_THROTTLE(
        logger_, *node_->get_clock(), 2000,
        "Goal transform completely failed: %s | fallback failed: %s",
        ex.what(), ex2.what());
      return stop;
    }
  }

  if (goal_pub_ && goal_pub_->is_activated()) {
    goal_pub_->publish(goal_base);
  }

  return getLatestCmd_(stamp);
}

}  // namespace amr_nav2_drl

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(amr_nav2_drl::DrlController, nav2_core::Controller)
