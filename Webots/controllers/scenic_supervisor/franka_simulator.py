from collections import defaultdict
import ctypes
import math
import random
from os import path
import tempfile
from textwrap import dedent

import numpy as np
import trimesh

from scenic.core.regions import MeshVolumeRegion
from scenic.core.simulators import Simulation, Simulator
from scenic.core.type_support import toOrientation
from scenic.core.vectors import Vector
from scenic.simulators.webots.utils import ENU, WebotsCoordinateSystem

class WebotsFrankaSimulator(Simulator):
    """`Simulator` object for Webots."""
    def __init__(self, supervisor):
        super().__init__()
        self.supervisor = supervisor
        topLevelNodes = supervisor.getRoot().getField("children")
        worldInfo = None
        for i in range(topLevelNodes.getCount()):
            child = topLevelNodes.getMFNode(i)
            if child.getTypeName() == "WorldInfo":
                worldInfo = child
                break
        if not worldInfo:
            raise RuntimeError("Webots world does not contain a WorldInfo node")
        system = worldInfo.getField("coordinateSystem").getSFString()
        self.coordinateSystem = WebotsCoordinateSystem(system)

    def createSimulation(self, scene, **kwargs):
        return WebotsFrankaSimulation(
            scene, self.supervisor, coordinateSystem=self.coordinateSystem, **kwargs
        )

class WebotsFrankaSimulation(Simulation):
    """`Simulation` object for Webots."""

    def __init__(self, scene, supervisor, coordinateSystem=ENU, *, timestep, **kwargs):
        print("\n\n" + "="*40)
        print("   >>> 终极修复版代码已加载 (Final Fix Loaded) <<<   ")
        print("="*40 + "\n\n")
        
        self.supervisor = supervisor
        self.coordinateSystem = coordinateSystem
        self.mode2D = scene.compileOptions.mode2D
        self.nextAdHocObjectId = 1
        self.usedObjectNames = defaultdict(lambda: 0)

        self.timestep = supervisor.getBasicTimeStep() / 1000 if timestep is None else timestep
        self.tmpMeshDir = tempfile.mkdtemp()
        self.supervisor_node = self.supervisor.getSelf()

        # Sensors and Motors
        sensor_names = [f"panda_joint{i}_sensor" for i in range(1,8)]
        self.position_sensors = [self.supervisor.getDevice(name) for name in sensor_names]
        
        motor_names = [f"panda_joint{i}" for i in range(1,8)]
        self.motors = [self.supervisor.getDevice(name) for name in motor_names]

        self.total_reward = 0
        self.enable_sensors = False
        self.ms = round(1000 * self.timestep)

        # 存储 Webots 节点句柄
        self.target_node = None
        self.hand_node = None     
        self.obstacle_nodes = []  
        
        self.actions = np.zeros(7)
        # 初始化观测：7关节 + 3目标位置 + 3手部位置 = 13维
        self.observation = np.zeros(13, dtype=np.float32) 

        self.prev_distance = 0.0

        super().__init__(scene, timestep=timestep, **kwargs)

    def setup(self):
        super().setup()
        self.supervisor.simulationResetPhysics()

    def createObjectInSimulator(self, obj):
        if not hasattr(obj, "webotsName"):
            return  # not a Webots object

        name = None
        
        # --- Case 1: Dynamic Object (Obstacles) ---
        if obj.webotsAdhoc is not None:
            objectRawMesh = obj.shape.mesh
            objectScaledMesh = MeshVolumeRegion(
                mesh=objectRawMesh,
                dimensions=(obj.width, obj.length, obj.height),
            ).mesh
            objFilePath = path.join(self.tmpMeshDir, f"{self.nextAdHocObjectId}.obj")
            trimesh.exchange.export.export_mesh(objectScaledMesh, objFilePath)

            name = self._getAdhocObjectName(self.nextAdHocObjectId)
            protoName = ("ScenicObjectWithPhysics" if isPhysicsEnabled(obj) else "ScenicObject")
            
            # Windows/Path fix
            objFilePath = str(objFilePath).replace("\\", "\\\\")

            protoDef = dedent(
                f"""
                DEF {name} {protoName} {{
                    url "{objFilePath}"
                }}
                """
            )
            rootNode = self.supervisor.getRoot()
            rootChildrenField = rootNode.getField("children")
            rootChildrenField.importMFNodeFromString(-1, protoDef)
            self.nextAdHocObjectId += 1
            
        # --- Case 2: Static/Managed Object (Target/Panda) ---
        else:
            if obj.webotsName:
                name = obj.webotsName
            else:
                ty = obj.webotsType
                if not ty:
                    raise RuntimeError(f"object {obj} has no webotsName or webotsType")
                nextID = self.usedObjectNames[ty]
                self.usedObjectNames[ty] += 1
                if nextID == 0 and self.supervisor.getFromDef(ty):
                    name = ty
                else:
                    name = f"{ty}_{nextID}"

        # Get handle to Webots node
        webotsObj = self.supervisor.getFromDef(name)
        if webotsObj is None:
            return # Skip if not found
        
        obj.webotsObject = webotsObj
        obj.webotsName = name

        # 存储句柄
        if name == "TARGET":
            self.target_node = webotsObj
        elif name.startswith("SCENIC_ADHOC"):
            self.obstacle_nodes.append(webotsObj)

        # Set fields (Position, Orientation, etc.)
        if self.mode2D:
             if obj.elevation is None:
                pos = webotsObj.getField("translation").getSFVec3f()
                spos = self.coordinateSystem.positionToScenic(pos)
                obj.elevation = spos[2]
             pos = self.coordinateSystem.positionFromScenic(
                Vector(obj.position.x, obj.position.y, obj.elevation) + obj.positionOffset
            )
             webotsObj.getField("translation").setSFVec3f(pos)
        else:
            pos = self.coordinateSystem.positionFromScenic(obj.position + obj.positionOffset)
            webotsObj.getField("translation").setSFVec3f(pos)

        offsetOrientation = toOrientation(obj.rotationOffset)
        webotsObj.getField("rotation").setSFRotation(
            self.coordinateSystem.orientationFromScenic(obj.orientation, offsetOrientation)
        )

        # customData
        customData = getattr(obj, "customData", None)
        if customData:
            webotsObj.getField("customData").setSFString(customData)

    def step(self): 
        if not self.enable_sensors: 
            self.init_step()

        # 1. 获取关节状态 (7维)
        joint_vals = [0.0] * 7
        for i, sensor in enumerate(self.position_sensors):
            if sensor:
                joint_vals[i] = sensor.getValue()
        
        # 2. 获取目标位置 (3维) - 使用 getPosition 确保实时性
        target_pos = [0.0, 0.0, 0.0]
        if self.target_node:
            target_pos = self.target_node.getPosition()
        
        # 3. 获取手部位置 (3维) - 使用 getPosition
        hand_pos = self._get_hand_position()

        # 4. 拼接观测向量
        self.observation = np.concatenate([joint_vals, target_pos, hand_pos]).astype(np.float32)
        
        # 执行动作 (限速 40% 防止物理爆炸)
        for action, motor in zip(self.actions, self.motors):
            if motor:
                val = action * 0.4
                motor.setVelocity(val)

        self.supervisor.step(self.ms)
        
        # 我把 step 里的 print 删掉了，防止刷屏影响速度
        # 调试信息在 get_reward 里看就够了

    def init_step(self):
        """Initialize sensors and find static nodes if not found yet."""
        # 初始化电机
        for motor in self.motors:
            if motor:
                motor.setPosition(float('inf'))
                motor.setVelocity(0)

        # 初始化传感器
        for sensor in self.position_sensors:
            if sensor:
                sensor.enable(self.ms)

        self.supervisor_node.enablePoseTracking(self.ms)
        self.supervisor.step(self.ms)
        
        # 再次尝试寻找目标节点
        if self.target_node is None:
            self.target_node = self.supervisor.getFromDef("TARGET") 
            
        # 强制手动随机移动 Target (解决 Scenic 随机失效的问题)
        if self.target_node:
            # 手动生成随机坐标 (范围 0.15 ~ 0.25)
            new_x = 0.2 + random.uniform(-0.05, 0.05) 
            new_y = 0.0 + random.uniform(-0.1, 0.1)
            # 抬高一点，防止嵌在地板里
            new_z = 0.08 
            
            # 强制设置 Webots 物体位置
            trans_field = self.target_node.getField("translation")
            trans_field.setSFVec3f([new_x, new_y, new_z])
            self.target_node.resetPhysics() # 重置物理状态
            print(f"DEBUG: Manual Reset Target to ({new_x:.3f}, {new_y:.3f}, {new_z:.3f})")
        
        # 寻找手部节点
        if self.hand_node is None:
            possible_hand_names = ["GRIPPER", "panda_link8", "panda_link7", "panda_hand"]
            for name in possible_hand_names:
                node = self.supervisor.getFromDef(name)
                if node:
                    self.hand_node = node
                    print(f"SUCCESS: Found robot hand node with name: '{name}'")
                    break
            
            if self.hand_node is None:
                print("WARNING: Could not find ANY hand node.")

        # 初始化上一帧距离
        if self.target_node and self.hand_node:
            t_pos = np.array(self.target_node.getPosition())
            h_pos = np.array(self._get_hand_position())
            self.prev_distance = np.linalg.norm(t_pos - h_pos)

        self.enable_sensors = True

    # --- 关键修改：使用 getPosition 获取绝对坐标 ---
    def _get_hand_position(self):
        if self.hand_node:
            return self.hand_node.getPosition() 
        else:
            return [0, 0, 0] # Fallback

    def getProperties(self, obj, properties):
        webotsObj = getattr(obj, "webotsObject", None)
        if not webotsObj:
            return {prop: getattr(obj, prop) for prop in properties}

        pos = webotsObj.getField("translation").getSFVec3f()
        x, y, z = self.coordinateSystem.positionToScenic(pos)
        
        velocity = Vector(0,0,0)
        angularVelocity = Vector(0,0,0)
        angularSpeed = 0
        speed = 0

        offsetOrientation = toOrientation(obj.rotationOffset)
        globalOrientation = self.coordinateSystem.orientationToScenic(
            webotsObj.getField("rotation").getSFRotation(), offsetOrientation
        )
        yaw, pitch, roll = obj.parentOrientation.localAnglesFor(globalOrientation)

        values = dict(
            position=Vector(x, y, z),
            velocity=velocity,
            speed=speed,
            angularSpeed=angularSpeed,
            angularVelocity=angularVelocity,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            elevation=z,
        )
        return values

    def destroy(self):
        for i in range(1, self.nextAdHocObjectId):
            name = self._getAdhocObjectName(i)
            node = self.supervisor.getFromDef(name)
            if node:
                node.remove()
        self.obstacle_nodes = [] 
        self.supervisor.step(self.ms)

    def _getAdhocObjectName(self, i: int) -> str:
        return f"SCENIC_ADHOC_{i}"

    # --- 【重点】奖励函数 ---
    def get_reward(self):
        # 获取绝对坐标
        target_pos = np.array(self.target_node.getPosition()) if self.target_node else np.array([0,0,0])
        hand_pos = np.array(self._get_hand_position())
        
        current_distance = np.linalg.norm(target_pos - hand_pos)
        
        # 1. 距离惩罚
        reward = -current_distance 
        
        # 2. 进步奖励 (放大信号)
        improvement = (self.prev_distance - current_distance)
        reward += improvement * 200.0 
        
        # 3. 成功奖励
        if current_distance < 0.05:
            reward += 100.0 
            print(f">>> TARGET CAUGHT! Dist={current_distance:.3f} <<<")

        self.prev_distance = current_distance
        
        # 调试打印 (1% 概率)
        if random.random() < 0.01:
            print(f"DEBUG: Dist={current_distance:.4f} | TargetX={target_pos[0]:.4f} | HandZ={hand_pos[2]:.3f}")
            
        return reward

    # --- 【重点】结束条件 ---
    def get_truncation(self): 
        target_pos = np.array(self.target_node.getPosition()) if self.target_node else np.array([0,0,0])
        hand_pos = np.array(self._get_hand_position())
        distance = np.linalg.norm(target_pos - hand_pos)
        
        if distance < 0.05:
            return True
        return False
    
    def get_info(self):
        return {}
     
    def get_obs(self):
        return self.observation
    
    def sampler_feedback(self): 
        return 0

# Helper functions... (Keeping existing helpers)
def getFieldSafe(webotsObject, fieldName):
    field = webotsObject.getField(fieldName)
    if field is None: return None
    if isinstance(field._ref, ctypes.c_void_p) and field._ref.value is not None:
        return field
    return None

def isPhysicsEnabled(webotsObject):
    if webotsObject.webotsAdhoc is None:
        return webotsObject
    if isinstance(webotsObject.webotsAdhoc, dict):
        return webotsObject.webotsAdhoc.get("physics", True)
    raise TypeError(f"webotsAdhoc must be None or a dictionary")