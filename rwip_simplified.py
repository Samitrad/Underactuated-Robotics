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
# 1. RWIP PLANT (PHYSICALLY CORRECTED)
# ================================================================
class RWIPPlant(LeafSystem):
    def __init__(self, m=0.5, l=0.3, I_eff=0.05, I_w=0.005, b_p=0.01, b_w=0.005, g=9.81):
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

        # --- 1. Pendulum Dynamics ---
        # I_eff * theta_ddot = -u - friction - gravity
        torque_p = -u - self.b_p * theta_dot - self.m * self.g * self.l * np.sin(theta)
        theta_ddot = self.inv_Ieff * torque_p
        
        # --- 2. Rotor Dynamics (CORRECTED) ---
        # The rotor is attached to the accelerating frame of the rod.
        # Equation: Iw * (omega_ddot + theta_ddot) = u - b_w * omega_w
        # Rearranged: omega_ddot = (u - b_w * omega_w)/Iw - theta_ddot
        omega_ddot = self.inv_Iw * (u - self.b_w * omega_w) - theta_ddot

        derivatives.get_mutable_vector().SetFromVector([
            theta_dot, theta_ddot, omega_ddot
        ])

    def CopyStateOut(self, context, output):
        output.SetFromVector(context.get_continuous_state_vector().CopyToVector())


# ================================================================
# 2. CONTROLLER
# ================================================================
class StabilizingController(LeafSystem):
    def __init__(self, p):
        super().__init__()

        self.m, self.l, self.g = p["m"], p["l"], p["g"]
        self.Ieff, self.Iw = p["I_eff"], p["I_w"]
        self.b_p, self.b_w = p["b_p"], p["b_w"]

        # LQR linearization around theta = pi (Upright)
        # Note: We use the simplified linearization for control design 
        # (ignoring the theta_ddot coupling in B matrix) as it is standard practice
        # and sufficiently robust for this system.
        A = np.array([
            [0, 1, 0],
            [self.m*self.g*self.l/self.Ieff, -self.b_p/self.Ieff, 0],
            [0, 0, -self.b_w/self.Iw]
        ])
        B = np.array([[0], [-1/self.Ieff], [1/self.Iw]])

        Q = np.diag([200, 10, 1])
        R = np.array([[0.3]])
        self.K, _ = LinearQuadraticRegulator(A, B, Q, R)

        self.theta_upright = np.pi
        self.stab_thresh = 0.25

        self.DeclareVectorInputPort("state", BasicVector(3))
        self.DeclareVectorOutputPort("u", BasicVector(1), self.Calc)

    def wrap_to_pi(self, a):
        return np.arctan2(np.sin(a), np.cos(a))

    def Calc(self, context, output):
        theta, theta_dot, omega = self.get_input_port(0).Eval(context)
        theta_err = self.wrap_to_pi(theta - self.theta_upright)

        # --- LQR Stabilization ---
        if abs(theta_err) < self.stab_thresh:
            x_err = np.array([theta_err, theta_dot, omega])
            u = -self.K.dot(x_err).item()
            u = np.clip(u, -5, 5)

        # --- Energy Swing-Up ---
        else:
            # Potential Energy is defined 0 at bottom (theta=0), max at top (theta=pi)
            # PE = mgl(1 - cos(theta))
            E = 0.5*self.Ieff*theta_dot**2 + self.m*self.g*self.l*(1 - np.cos(theta))
            E_des = 2*self.m*self.g*self.l
            E_err = E_des - E

            u = -6 * E_err * theta_dot
            
            # Kick to escape perfect bottom equilibrium
            if abs(theta_dot) < 0.05 and abs(np.cos(theta)) > 0.98:
                u += 0.5 * np.sign(np.sin(theta) + 1e-6)

            u = np.clip(u, -0.5, 0.5)

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
        self.period = 0.01

        self.DeclarePeriodicDiscreteUpdateEvent(self.period, 0, self.UpdateAngle)
        self.DeclarePeriodicPublishEvent(0.03, 0, self.DoPublish)

        # Large wheel geometry
        self.r = L * 0.9
        self.spoke_len = self.r * 1.8
        self.spoke_w = L * 0.12
        self.spoke_h = L * 0.12

        # Spokes
        geom_spoke = g.Box([self.spoke_len, self.spoke_w, self.spoke_h])
        mat_spoke = g.MeshLambertMaterial(color=0x1E90FF)
        self.spokes = []
        for i in range(3):
            node = self.vis["rwip"]["wheel"][f"spoke{i}"]
            node.set_object(geom_spoke, mat_spoke)
            self.spokes.append(node)
            
        # --- THIN RING ---
        self.num_seg = 24
        ring_radius = self.spoke_len / 2.0 
        tube_thickness = self.spoke_h * 0.25 
        
        seg_len = (2 * np.pi * ring_radius) / self.num_seg
        geom_seg = g.Cylinder(height=seg_len * 1.05, radius=tube_thickness)
        mat_ring = g.MeshLambertMaterial(color=0x888888) # Grey
        
        self.ring_segments = []
        for i in range(self.num_seg):
            node = self.vis["rwip"]["wheel"][f"ring_seg_{i}"]
            node.set_object(geom_seg, mat_ring)
            self.ring_segments.append(node)

    def UpdateAngle(self, context, state):
        x = self.get_input_port(0).Eval(context)
        omega_w = x[2]
        angle = state.get_mutable_vector()[0] + omega_w * self.period
        state.get_mutable_vector().SetFromVector([angle])

    def DoPublish(self, context):
        theta, _, _ = self.get_input_port(0).Eval(context)
        a = context.get_discrete_state(0)[0]

        rod_tip_dist = self.L * 2.3
        x = rod_tip_dist * np.sin(theta)
        y = -rod_tip_dist * np.cos(theta)

        T_pos = tf.translation_matrix([x, y, 0])
        R_pend = tf.rotation_matrix(theta, [0, 0, 1])
        R_spin = tf.rotation_matrix(a, [0, 0, 1])
        T_final = T_pos @ R_pend @ R_spin

        for i, s in enumerate(self.spokes):
            R_s = tf.rotation_matrix(i * 2*np.pi/3, [0, 0, 1])
            s.set_transform(T_final @ R_s)
            
        for i, seg in enumerate(self.ring_segments):
            angle = 2 * np.pi * i / self.num_seg
            T_static = tf.rotation_matrix(angle, [0, 0, 1]) @ \
                       tf.translation_matrix([self.spoke_len/2.0, 0, 0])
            seg.set_transform(T_final @ T_static)


# ================================================================
# 4. MAIN
# ================================================================
def main():
    builder = DiagramBuilder()
    vis = meshcat.Visualizer().open()
    vis.delete()

    setup_background(vis)

    # --- UPDATED CAMERA POSITION: ZOOMED IN ---
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
    # ----------------------------

    params = {
        "m": 0.5, "l": 0.3,
        "I_eff": 0.05, "I_w": 0.005,
        "b_p": 0.01, "b_w": 0.005,
        "g": 9.81,
    }

    plant = builder.AddSystem(RWIPPlant(**params))
    ctrl = builder.AddSystem(StabilizingController(params))
    zoh = builder.AddSystem(ZeroOrderHold(0.01, 1))

    viz_p = builder.AddSystem(RWIPVisualizer(vis, params["l"]))
    viz_w = builder.AddSystem(RotorVisualizer(vis, params["l"]))

    builder.Connect(plant.get_output_port(0), ctrl.get_input_port(0))
    builder.Connect(ctrl.get_output_port(0), zoh.get_input_port(0))
    builder.Connect(zoh.get_output_port(0), plant.get_input_port(0))

    builder.Connect(plant.get_output_port(0), viz_p.get_input_port(0))
    builder.Connect(plant.get_output_port(0), viz_w.get_input_port(0))

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    # Initial condition: slightly offset from bottom
    diagram.GetMutableSubsystemContext(plant, context).SetContinuousState(
        [0.15, 0.0, 0.0]
    )

    sim = Simulator(diagram, context)
    sim.set_target_realtime_rate(1.0)
    sim.Initialize()

    print("Simulating RWIP (Physically Corrected)...")
    sim.AdvanceTo(25.0)


if __name__ == "__main__":
    main()
