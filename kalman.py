import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
dt = 0.1  # time step
time = np.arange(0, 50, dt)

# True robot motion (unknown to the filter)
true_position = []
true_velocity = 1.0  # moving at 1 m/s
pos = 0

# Kalman filter state
x = np.array([[0.0],    # position estimate
              [0.0]])   # velocity estimate

# State covariance (uncertainty)
P = np.array([[1.0, 0.0],
              [0.0, 1.0]])

# Process model (how position changes)
F = np.array([[1.0, dt],   # position = position + velocity*dt
              [0.0, 1.0]]) # velocity stays constant

# Measurement model (we only measure position, not velocity)
H = np.array([[1.0, 0.0]])

# Process noise (uncertainty in motion model)
Q = np.array([[0.01, 0.0],
              [0.0, 0.01]])

# Measurement noise (sensor uncertainty)
R = np.array([[4.0]])  # GPS has ±2m error (variance = 4)

# Storage for plotting
measurements = []
estimates = []
uncertainties = []

# Simulation loop
for t in time:
    # TRUE MOTION (what's really happening)
    pos += true_velocity * dt
    true_position.append(pos)
    
    # NOISY MEASUREMENT (what sensor reports)
    measurement = pos + np.random.normal(0, 2.0)  # add noise
    measurements.append(measurement)
    
    # KALMAN FILTER
    # 1. PREDICT
    x = F @ x  # predict next state
    P = F @ P @ F.T + Q  # predict uncertainty
    
    # 2. UPDATE
    y = measurement - (H @ x)[0, 0]  # innovation (measurement - prediction)
    S = H @ P @ H.T + R  # innovation covariance
    K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
    
    x = x + K * y  # update estimate
    P = (np.eye(2) - K @ H) @ P  # update uncertainty
    
    estimates.append(x[0, 0])
    uncertainties.append(np.sqrt(P[0, 0]))

# Plot results
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(time, true_position, 'g-', label='True Position', linewidth=2)
plt.plot(time, measurements, 'r.', label='Noisy Measurements', alpha=0.5, markersize=3)
plt.plot(time, estimates, 'b-', label='Kalman Estimate', linewidth=2)
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)
plt.title('Kalman Filter: Position Tracking')

plt.subplot(3, 1, 2)
errors = np.array(estimates) - np.array(true_position)
plt.plot(time, errors, 'b-')
plt.ylabel('Estimation Error (m)')
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.grid(True)
plt.title('Kalman Filter Error')

plt.subplot(3, 1, 3)
plt.plot(time, uncertainties, 'b-')
plt.ylabel('Uncertainty (σ)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.title('Filter Confidence (lower = more confident)')

plt.tight_layout()
plt.savefig('kalman_filter.png')
print("Simulation complete! Graph saved.")
print(f"Final true position: {true_position[-1]:.2f} m")
print(f"Final estimate: {estimates[-1]:.2f} m")
print(f"Final error: {abs(estimates[-1] - true_position[-1]):.2f} m")
print(f"Measurement noise: ±2.0 m, but Kalman reduced error significantly!")
