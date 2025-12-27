import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

# System parameters (cart-pole physics)
M = 1.0      # cart mass (kg)
m = 0.1      # pole mass (kg)
L = 1.0      # pole length (m)
g = 9.81     # gravity (m/s^2)
b = 0.1      # friction coefficient

# Linearized state-space model around upright position
# State: [x, x_dot, theta, theta_dot]
# x = cart position, theta = pole angle (0 = upright)

# Compute system matrices
A = np.array([
    [0, 1, 0, 0],
    [0, -b/M, -m*g/M, 0],
    [0, 0, 0, 1],
    [0, b/(M*L), (M+m)*g/(M*L), 0]
])

B = np.array([
    [0],
    [1/M],
    [0],
    [-1/(M*L)]
])

# LQR cost matrices (what do we care about?)
# Q: penalize state deviations
# R: penalize control effort

# Aggressive control (care about angle, less about energy)
Q_aggressive = np.diag([1, 1, 100, 10])  # Really care about pole angle!
R_aggressive = np.array([[0.1]])         # Don't care much about energy

# Balanced control
Q_balanced = np.diag([10, 1, 50, 10])
R_balanced = np.array([[1]])

# Conservative control (care about energy)
Q_conservative = np.diag([10, 1, 50, 10])
R_conservative = np.array([[10]])        # Penalize large control inputs

def compute_lqr_gain(A, B, Q, R):
    """Solve the continuous-time LQR problem"""
    # Solve Algebraic Riccati Equation
    P = solve_continuous_are(A, B, Q, R)
    
    # Compute optimal gain matrix
    K = np.linalg.inv(R) @ B.T @ P
    
    return K

def simulate_cartpole(K, duration=10, dt=0.01, initial_angle=0.2):
    """Simulate cart-pole with LQR controller"""
    time = np.arange(0, duration, dt)
    
    # Initial state: small disturbance from upright
    x = np.array([0.0, 0.0, initial_angle, 0.0])  # [position, velocity, angle, angular_vel]
    
    # Storage
    states = []
    controls = []
    
    for t in time:
        # LQR control law: u = -K * x
        u = -K @ x
        u = np.clip(u, -20, 20)  # Limit maximum force
        
        # State derivative (linearized dynamics)
        x_dot = A @ x + B @ u.flatten()
        
        # Euler integration
        x = x + x_dot * dt
        
        states.append(x.copy())
        controls.append(u[0])
    
    return time, np.array(states), np.array(controls)

# Compute LQR gains for different strategies
K_aggressive = compute_lqr_gain(A, B, Q_aggressive, R_aggressive)
K_balanced = compute_lqr_gain(A, B, Q_balanced, R_balanced)
K_conservative = compute_lqr_gain(A, B, Q_conservative, R_conservative)

print("LQR Gains Computed!")
print(f"Aggressive:   K = {K_aggressive.flatten()}")
print(f"Balanced:     K = {K_balanced.flatten()}")
print(f"Conservative: K = {K_conservative.flatten()}")

# Simulate all three controllers
initial_disturbance = 0.3  # 0.3 radians (~17 degrees)

time_agg, states_agg, controls_agg = simulate_cartpole(K_aggressive, initial_angle=initial_disturbance)
time_bal, states_bal, controls_bal = simulate_cartpole(K_balanced, initial_angle=initial_disturbance)
time_con, states_con, controls_con = simulate_cartpole(K_conservative, initial_angle=initial_disturbance)

# Plot results
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

controllers = [
    ("Aggressive (Fast Response)", states_agg, controls_agg),
    ("Balanced (Medium)", states_bal, controls_bal),
    ("Conservative (Low Energy)", states_con, controls_con)
]

for i, (name, states, controls) in enumerate(controllers):
    # Pole angle
    axes[i, 0].plot(time_agg, states[:, 2] * 180/np.pi)
    axes[i, 0].set_ylabel('Pole Angle (deg)')
    axes[i, 0].set_title(f'{name}\nPole Angle')
    axes[i, 0].grid(True)
    axes[i, 0].axhline(0, color='r', linestyle='--', alpha=0.3)
    
    # Cart position
    axes[i, 1].plot(time_agg, states[:, 0])
    axes[i, 1].set_ylabel('Cart Position (m)')
    axes[i, 1].set_title('Cart Position')
    axes[i, 1].grid(True)
    
    # Control input
    axes[i, 2].plot(time_agg, controls)
    axes[i, 2].set_ylabel('Force (N)')
    axes[i, 2].set_title('Control Force')
    axes[i, 2].grid(True)

axes[2, 0].set_xlabel('Time (s)')
axes[2, 1].set_xlabel('Time (s)')
axes[2, 2].set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig('lqr_cartpole.png', dpi=150)

print("\nSimulation complete!")
print(f"Initial disturbance: {initial_disturbance*180/np.pi:.1f} degrees")
print("\nFinal pole angles:")
print(f"  Aggressive:   {states_agg[-1, 2]*180/np.pi:.3f} deg")
print(f"  Balanced:     {states_bal[-1, 2]*180/np.pi:.3f} deg")
print(f"  Conservative: {states_con[-1, 2]*180/np.pi:.3f} deg")
print("\nMax control forces:")
print(f"  Aggressive:   {np.max(np.abs(controls_agg)):.2f} N")
print(f"  Balanced:     {np.max(np.abs(controls_bal)):.2f} N")
print(f"  Conservative: {np.max(np.abs(controls_con)):.2f} N")
