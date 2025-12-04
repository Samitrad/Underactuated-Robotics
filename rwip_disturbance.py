import sys
import select
#!/usr/bin/env python3
import numpy as np
from pydrake.all import (
    DiagramBuilder,
    Simulator,
    LeafSystem,
    BasicVector,
    ZeroOrderHold,
    LinearQuadraticRegulator,
)
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf


# ================================================================
# NICE BACKGROUND (WHITE BOX + GREY FLOOR)
# ================================================================
def setup_background(vis):
    # --- Parameters for walls ---
    wall_distance = 1.5  
    wall_length = 3.0    
    wall_height = 2.0    
    wall_thickness = 0.05
    wall_color = 0xFFFFFF  # White

    # ---- SKY ----
    vis["/Background"].set_object(
        g.Box([20, 20, 0.1]),
        g.MeshLambertMaterial(color=0x87CEEB, opacity=1.0)
    )
    vis["/Background"].set_transform(tf.translation_matrix([0, 0, -2]))

    # ---- FLOOR ----
    vis["/Floor"].set_object(
        g.Box([20, 20, 0.1]),
        g.MeshLambertMaterial(color=0x999999, opacity=1.0) # Grey
    )
    vis["/Floor"].set_transform(tf.translation_matrix([0, 0, -1]))

    # --- WALLS ---
    vis["/Wall_Back"].set_object(g.Box([wall_length, wall_thickness, wall_height]), g.MeshLambertMaterial(color=wall_color, opacity=1.0))
    vis["/Wall_Back"].set_transform(tf.translation_matrix([0, -wall_distance, wall_height / 2]))

    vis["/Wall_Front"].set_object(g.Box([wall_length, wall_thickness, wall_height]), g.MeshLambertMaterial(color=wall_color, opacity=1.0))
    vis["/Wall_Front"].set_transform(tf.translation_matrix([0, wall_distance, wall_height / 2]))

    vis["/Wall_Left"].set_object(g.Box([wall_thickness, wall_length, wall_height]), g.MeshLambertMaterial(color=wall_color, opacity=1.0))
    vis["/Wall_Left"].set_transform(tf.translation_matrix([-wall_distance, 0, wall_height / 2]))

    vis["/Wall_Right"].set_object(g.Box([wall_thickness, wall_length, wall_height]), g.MeshLambertMaterial(color=wall_color, opacity=1.0))
    vis["/Wall_Right"].set_transform(tf.translation_matrix([wall_distance, 0, wall_height / 2]))


# ================================================================
# 1. RWIP PLANT (FULL PHYSICS VERSION)
# ================================================================
class RWIPPlant(LeafSystem):
    def __init__(self, m=0.5, l=0.3, I_eff=0.05, I_w=0.03, b_p=0.001, b_w=0.005, g=9.81):
        super().__init__()
        self.m = m
        self.l = l
        self.Ieff = I_eff
        self.Iw = I_w
        self.b_p = b_p
        self.b_w = b_w
        self.g = g
        self.inv_Ieff = 1.0 / I_eff
        self.inv_Iw = 1.0 / I_w

        self.DeclareContinuousState(3)              # [θ, θ̇, ωw]
        self.DeclareVectorInputPort("torque", BasicVector(1))
        self.DeclareVectorOutputPort("state", BasicVector(3), self.CopyStateOut)

    def DoCalcTimeDerivatives(self, context, derivatives):
        theta, theta_dot, omega_w = context.get_continuous_state_vector().CopyToVector()
        u = self.get_input_port(0).Eval(context)[0]

        # 1. Net torque on Rotor
        tau_rotor = u - self.b_w * omega_w

        # 2. Pendulum Dynamics (Full Lagrangian)
        # Gravity is RESTORING (-) towards 0
        grav_torque = -self.m * self.g * self.l * np.sin(theta)
        fric_torque = -self.b_p * theta_dot
        
        # Torque on rod includes reaction from rotor (-tau_rotor)
        torque_on_rod = grav_torque + fric_torque - tau_rotor
        
        theta_ddot = self.inv_Ieff * torque_on_rod

        # 3. Rotor Dynamics (Coupled)
        # Accounts for the fact that the wheel frame is accelerating (theta_ddot)
        omega_ddot = (self.inv_Iw * tau_rotor) - theta_ddot

        derivatives.get_mutable_vector().SetFromVector([
            theta_dot, theta_ddot, omega_ddot
        ])

    def CopyStateOut(self, context, output):
        output.SetFromVector(context.get_continuous_state_vector().CopyToVector())


# ================================================================
# 2. CONTROLLER (FULL COUPLED LQR)
# ================================================================
class StabilizingController(LeafSystem):
    def __init__(self, p):
        super().__init__()
        self.m, self.l, self.g = p["m"], p["l"], p["g"]
        self.Ieff, self.Iw = p["I_eff"], p["I_w"]
        self.b_p, self.b_w = p["b_p"], p["b_w"]

        # Linearization around Upright (theta = pi)
        C1 = 1.0 / self.Ieff
        C2 = 1.0 / self.Iw
        g_term = self.m * self.g * self.l

        # State Matrix A (Full Dynamics Linearization)
        # Captures the coupling between rod and wheel acceleration
        A = np.array([
            [0, 1, 0],
            [C1 * g_term,  -C1 * self.b_p,   C1 * self.b_w], 
            [-C1 * g_term,  C1 * self.b_p,  -(C1 + C2) * self.b_w]
        ])

        # Input Matrix B (Full Dynamics)
        B = np.array([
            [0],
            [-C1],
            [C1 + C2]
        ])


        Q = np.diag([1000, 100, 0.1]) 
        R = np.array([[0.1]]) 
        
        self.K, _ = LinearQuadraticRegulator(A, B, Q, R)

        self.theta_upright = np.pi
        self.stab_thresh = 0.5 

        self.DeclareVectorInputPort("state", BasicVector(3))
        self.DeclareVectorOutputPort("u", BasicVector(1), self.Calc)

    def wrap_to_pi(self, a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def Calc(self, context, output):
        theta, theta_dot, omega = self.get_input_port(0).Eval(context)
        theta_err = self.wrap_to_pi(theta - self.theta_upright)

        u = 0.0

        # LQR STABILIZATION 
        if abs(theta_err) < self.stab_thresh:
            x_err = np.array([theta_err, theta_dot, omega])
            u = -self.K.dot(x_err).item()
            # High torque limit for capture
            u = np.clip(u, -0.5, 0.5)

        # ENERGY SWING-UP
        else:
            E = 0.5 * self.Ieff * theta_dot**2 + self.m * self.g * self.l * (1 - np.cos(theta))
            E_des = 2 * self.m * self.g * self.l
            E_err = E_des - E

            k_swing = 3.0
            u = -k_swing * E_err * theta_dot
            u = np.clip(u, -5, 5)

        output.SetFromVector([u])


# ================================================================
# 3. VISUALIZERS
# ================================================================
class RWIPVisualizer(LeafSystem):
    def __init__(self, vis, L):
        super().__init__()
        self.vis = vis
        self.L = L
        self.DeclareVectorInputPort("state", BasicVector(3))
        self.DeclarePeriodicPublishEvent(0.03, 0, self.DoPublish)
        self.rod_length = L * 2.3
        self.rod_radius = L * 0.05
        self.vis["rwip"]["pivot"].set_object(g.Sphere(L*0.12), g.MeshLambertMaterial(color=0x000000))
        self.vis["rwip"]["rod"].set_object(g.Cylinder(height=self.rod_length, radius=self.rod_radius), g.MeshLambertMaterial(color=0xAA0000))

    def DoPublish(self, context):
        theta = self.get_input_port(0).Eval(context)[0]
        R = tf.rotation_matrix(theta, [0, 0, 1])
        T = tf.translation_matrix([0, -self.rod_length/2, 0])
        self.vis["rwip"]["rod"].set_transform(R @ T)


class RotorVisualizer(LeafSystem):
    def __init__(self, vis, L):
        super().__init__()
        self.vis = vis
        self.L = L
        self.DeclareVectorInputPort("state", BasicVector(3))
        self.DeclareDiscreteState(1)
        self.DeclarePeriodicDiscreteUpdateEvent(0.01, 0, self.UpdateAngle)
        self.DeclarePeriodicPublishEvent(0.03, 0, self.DoPublish)
        
        self.r = L * 0.9
        self.spoke_len = self.r * 1.8
        self.spoke_w = L * 0.12
        self.spoke_h = L * 0.12
        
        geom_spoke = g.Box([self.spoke_len, self.spoke_w, self.spoke_h])
        mat_spoke = g.MeshLambertMaterial(color=0x1E90FF)
        self.spokes = []
        for i in range(3):
            node = self.vis["rwip"]["wheel"][f"spoke{i}"]
            node.set_object(geom_spoke, mat_spoke)
            self.spokes.append(node)
            
        self.num_seg = 24
        ring_radius = self.spoke_len / 2.0 
        tube_thickness = self.spoke_h * 0.25 
        seg_len = (2 * np.pi * ring_radius) / self.num_seg
        geom_seg = g.Cylinder(height=seg_len * 1.05, radius=tube_thickness)
        mat_ring = g.MeshLambertMaterial(color=0x888888)
        self.ring_segments = []
        for i in range(self.num_seg):
            node = self.vis["rwip"]["wheel"][f"ring_seg_{i}"]
            node.set_object(geom_seg, mat_ring)
            self.ring_segments.append(node)

    def UpdateAngle(self, context, state):
        omega_w = self.get_input_port(0).Eval(context)[2]
        current_ang = state.get_vector().GetAtIndex(0)
        state.get_mutable_vector().SetAtIndex(0, current_ang + omega_w * 0.01)

    def DoPublish(self, context):
        theta = self.get_input_port(0).Eval(context)[0]
        wheel_ang = context.get_discrete_state(0)[0]
        
        tip_len = self.L * 2.3
        x = tip_len * np.sin(theta)
        y = -tip_len * np.cos(theta)
        
        T_pos = tf.translation_matrix([x, y, 0])
        R_pend = tf.rotation_matrix(theta, [0, 0, 1])
        R_wheel = tf.rotation_matrix(wheel_ang, [0, 0, 1])
        
        T_final = T_pos @ R_pend @ R_wheel
        
        for i, s in enumerate(self.spokes):
            R_s = tf.rotation_matrix(i * 2*np.pi/3, [0, 0, 1])
            s.set_transform(T_final @ R_s)
            
        for i, seg in enumerate(self.ring_segments):
            angle = 2 * np.pi * i / self.num_seg
            T_static = tf.rotation_matrix(angle, [0, 0, 1]) @ tf.translation_matrix([self.spoke_len/2.0, 0, 0])
            seg.set_transform(T_final @ T_static)


# ================================================================
# 4. MAIN
# ================================================================

def main():
    builder = DiagramBuilder()
    vis = meshcat.Visualizer().open()
    vis.delete()

    setup_background(vis)

    # --- Camera Setup ---
    camera_pos = np.array([-0.6, -0.2, -0.2])
    target_pos = np.array([0, 0, 0])
    up_vector = np.array([0, 0, 1]) 
    z_axis = target_pos - camera_pos
    z_axis /= np.linalg.norm(z_axis) 
    x_axis = np.cross(up_vector, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    R_world_camera = np.column_stack([x_axis, y_axis, z_axis])
    T_world_camera = np.eye(4)
    T_world_camera[:3, :3] = R_world_camera
    T_world_camera[:3, 3] = camera_pos
    
    camera_transform = np.linalg.inv(T_world_camera)
    vis["/Cameras/default/rotated"].set_transform(camera_transform)

    # --- System Parameters ---
    params = {
        "m": 0.5, "l": 0.3,
        "I_eff": 0.05, 
        "I_w": 0.03,   
        "b_p": 0.001,  
        "b_w": 0.005,
        "g": 9.81,
    }

    plant = builder.AddSystem(RWIPPlant(**params))
    ctrl = builder.AddSystem(StabilizingController(params))
    zoh = builder.AddSystem(ZeroOrderHold(0.005, 1)) 

    viz_p = builder.AddSystem(RWIPVisualizer(vis, params["l"]))
    viz_w = builder.AddSystem(RotorVisualizer(vis, params["l"]))

    builder.Connect(plant.get_output_port(0), ctrl.get_input_port(0))
    builder.Connect(ctrl.get_output_port(0), zoh.get_input_port(0))
    builder.Connect(zoh.get_output_port(0), plant.get_input_port(0))
    builder.Connect(plant.get_output_port(0), viz_p.get_input_port(0))
    builder.Connect(plant.get_output_port(0), viz_w.get_input_port(0))

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    # Initial State: Slightly off-center to test start-up stabilization
    diagram.GetMutableSubsystemContext(plant, context).SetContinuousState([0.1, 0.0, 0.0])

    sim = Simulator(diagram, context)
    sim.set_target_realtime_rate(1.0)
    sim.Initialize()

    print("==========================================")
    print("SIMULATION RUNNING")
    print(">> Press [ENTER] in this terminal to PUSH the robot.")
    print(">> Press [Ctrl+C] to exit.")
    print("==========================================")

    # --- INTERACTIVE SIMULATION LOOP ---
    simulation_time = 30.0
    current_time = 0.0
    step_size = 0.05  # Check for input 20 times per second

    try:
        while current_time < simulation_time:
            # 1. Run physics for a tiny step
            sim.AdvanceTo(current_time + step_size)
            current_time += step_size

            # 2. Check if ENTER is pressed (Non-blocking)
            # select.select([sys.stdin], [], [], 0) checks if there is input waiting
            if select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline() # Read the key press to clear buffer
                
                # 3. APPLY THE PUSH (Disturbance)
                print("\n>>> PUSH APPLIED! <<<")
                
                # Get the mutable context for the plant
                plant_context = diagram.GetMutableSubsystemContext(plant, context)
                state_vector = plant_context.get_mutable_continuous_state_vector()
                
                # state indices: [0]=theta, [1]=theta_dot, [2]=omega_wheel
                current_vel = state_vector.GetAtIndex(1)
                
                # Add a sudden velocity kick (impulse)
                kick_strength = 1.5  # Adjust this value for a harder/softer push
                state_vector.SetAtIndex(1, current_vel + kick_strength)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

if __name__ == "__main__":
    main()
