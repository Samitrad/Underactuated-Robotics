# Reaction Wheel Inverted Pendulum (RWIP) Simulation
### 🤖 Underactuated Robotics Project | LAU

## 📌 Project Overview

This repository contains our implementation of the **Reaction Wheel Inverted Pendulum (RWIP)** system using **Drake**, **Meshcat**, and **Python**.

The project focuses on the modeling, simulation, visualization, and control of an underactuated system. It features a custom 3D environment setup and **Linear Quadratic Regulator (LQR)** stabilization to keep the pendulum upright using the reaction wheel's inertia.

**Course:** Underactuated Robotics at Lebanese American University (LAU).

## 🧠 System Architecture

The simulation is built using `pydrake`'s `DiagramBuilder` to connect the following components:

| Component | Description |
| :--- | :--- |
| **RWIP Dynamics** | A custom `LeafSystem` defining the equations of motion for the pendulum and reaction wheel. |
| **LQR Controller** | Linearizes the dynamics around the upright fixed point and applies $u = -Kx$ to stabilize the system. |
| **Meshcat Visualizer** | Renders the robot in a browser-based 3D scene with custom lighting and geometry. |
| **Zero-Order Hold** | Discretizes the control inputs to simulate digital control. |

### The Physics
The system is modeled as a double integrator system with coupling between the pendulum arm and the wheel.
* **State Vector:** $x = [\theta_{pendulum}, \theta_{wheel}, \dot{\theta}_{pendulum}, \dot{\theta}_{wheel}]^T$
* **Control Input:** Torque applied to the reaction wheel.

## 🔧 Installation & Setup (Ubuntu)

Follow these steps to set up the simulation environment on Ubuntu (22.04 / 24.04).

### 1. System Dependencies
First, ensure you have Python, Pip, and the virtual environment package installed:

```bash
sudo apt update
sudo apt install -y python3 python3-pip git
pip install drake meshcat numpy
```
🚀 How to Run

```bash
cd Underactuated-Robotics
python3 rwip.py
```
View the Visualization:

The terminal will output a URL (e.g., http://localhost:7000/static/).

Ctrl+Click the link or paste it into your web browser.

You should see the custom 3D environment with the RWIP system.

## 📊 Features

[x] Custom Dynamics: Full implementation of RWIP equations of motion in a Drake LeafSystem.

[x] Swing-up control: Get the system to an upright position

[x] LQR Stabilization: Controller tuned to maintain the upright equilibrium.

[x] Meshcat Visualization: Integrated browser-based 3D rendering.

[x] Modular Code: Clean separation between plant, controller, and visualization logic.

## 👥 Team
This project was created by:

Sami Trad

Christina Kamel
