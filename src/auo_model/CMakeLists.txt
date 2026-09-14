cmake_minimum_required(VERSION 3.8)
project(auo_model)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(xacro REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()
# *** 新增：安裝 Launch 檔案 ***
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# *** 新增：安裝頂層 Robot (XACRO) 檔案 ***
install(DIRECTORY
  robot
  DESTINATION share/${PROJECT_NAME}
)

# *** 新增：安裝 URDF (XACRO) 檔案 ***
install(DIRECTORY
  urdf
  DESTINATION share/${PROJECT_NAME}
)

# *** 新增：安裝 Meshes (DAE, STL 檔案) ***
install(DIRECTORY
  meshes
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY
  materials
  DESTINATION share/${PROJECT_NAME}
)
ament_package()
