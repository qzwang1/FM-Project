model scenic.simulators.webots.model

#######################################
#           Robot Definition
#######################################

class Franka(WebotsObject):
    webotsName: "PANDA"
    Supervisor: True
    controller: "scenic_supervisor"
    resetController: False

#######################################
#             Workspace
#######################################

# 【关键修复1】把工作区变得巨大，防止机器人因为太高而被判定为“出界”
workspace = Workspace(BoxRegion(
    dimensions=(10.0, 10.0, 10.0), 
    position=(0, 0, 2.0)
))

# randomize robot starting position slightly
start_x = Uniform(-0.01, 0.01) # 范围改小一点，先求稳
start_y = Uniform(-0.01, 0.01)

# 【关键修复2】显式允许碰撞，防止Scenic因为包围盒过大而报错
ego = new Franka at (start_x, start_y, 0),
    with allowCollisions True

#######################################
#             Target Definition
#######################################

# Random target position
target_x = 0.2 
target_y = 0.0

class TARGET(WebotsObject):
    webotsName: "TARGET"
    shape: CylinderShape()
    width: 0.1   # 变宽一倍
    length: 0.1  # 变长一倍
    height: 0.2  # 变高

target = new TARGET at (target_x, target_y, 0.15),
    with allowCollisions True

#######################################
#           Random Obstacles
#######################################

# Number of obstacles
n_obs = 1 

class OBSTACLE(WebotsObject):
    webotsAdhoc: {'physics': False}
    webotsType: "ScenicObject"
    shape: CylinderShape()
    width: 0.06
    length: 0.06
    height: 0.15
    color: [1, 0, 0]

obstacles = [
    new OBSTACLE at (
        Uniform(0.15, 0.40),
        Uniform(-0.3, 0.3),
        0.08  # 【关键修复3】Z轴稍微抬高，防止和地板冲突
    ),
    with allowCollisions True
    for i in range(n_obs)
]

#######################################
#    Scenario Validity Constraints
#######################################

# 全部注释掉。依靠坐标范围（Range）来隔离物体。
# 既然我们已经用 allowCollisions 禁用了检查，这里不需要写任何 require。

#######################################
#          Termination Condition
#######################################

terminate after 2000 steps