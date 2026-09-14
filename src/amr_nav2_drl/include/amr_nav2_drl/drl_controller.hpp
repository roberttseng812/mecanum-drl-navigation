#pragma once

#include <memory>
#include <mutex>
#include <string>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav2_core/controller.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "tf2_ros/buffer.h"

namespace nav2_costmap_2d
{
class Costmap2DROS;
}

namespace amr_nav2_drl
{

class DrlController : public nav2_core::Controller
{
public:
  DrlController() = default;
  ~DrlController() override = default;

  // IMPORTANT: match nav2_core::Controller interface EXACTLY (ROS 2 Humble)
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  void setPlan(const nav_msgs::msg::Path & path) override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;

  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

private:
  // Helpers
  size_t findClosestPoseIndex_(const geometry_msgs::msg::PoseStamped & pose) const;
  size_t findLookaheadPoseIndex_(size_t start_index, double lookahead_dist) const;
  geometry_msgs::msg::TwistStamped getLatestCmd_(const rclcpp::Time & stamp);

  // ROS / Nav2 handles
  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  std::string name_;
  rclcpp::Logger logger_{rclcpp::get_logger("amr_nav2_drl")};

  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;

  nav_msgs::msg::Path global_plan_;

  // Topics
  std::string goal_topic_{"/drl/local_goal"};
  std::string cmd_topic_{"/drl/cmd_vel"};

  rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;

  // Latest cmd cache
  std::mutex cmd_mtx_;
  geometry_msgs::msg::Twist last_cmd_;
  rclcpp::Time last_cmd_time_{0, 0, RCL_ROS_TIME};

  // Params
  double lookahead_dist_{2.5};
  double cmd_timeout_{0.25};
  double nominal_speed_{0.5};
  double transform_tolerance_{0.10};

  // Speed limit hooks (Nav2 may call setSpeedLimit)
  double speed_limit_{0.0};
  bool speed_limit_is_percentage_{false};
};

}  // namespace amr_nav2_drl

