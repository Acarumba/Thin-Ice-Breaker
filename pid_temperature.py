import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
dt = 0.1  # time step (seconds)
time = np.arange(0, 100, dt)  # simulate for 100 seconds
target_temp = 75  # target temperature in Celsius

# System parameters (how the heater behaves)
temp = 20  # starting temperature
heat_loss_rate = 0.05  # how fast heat escapes
heat_gain_rate = 2.0  # how much each unit of power heats

# PID parameters (we'll tune these)
Kp = 1.0  # Proportional gain
Ki = 0.05  # Integral gain (start with 0)
Kd = 0.0  # Derivative gain (start with 0)

# Storage for plotting
temps = []
errors = []
control_signals = []

# PID state
integral = 0
last_error = 0

# Simulation loop
for t in time:
    # Calculate error
    error = target_temp - temp
    
    # PID calculations
    integral += error * dt
    derivative = (error - last_error) / dt
    
    # Control signal (heater power)
    control = Kp * error + Ki * integral + Kd * derivative
    control = max(0, min(control, 100))  # limit between 0-100%
    
    # Update temperature (physics simulation)
    temp += (heat_gain_rate * control - heat_loss_rate * temp) * dt
    
    # Store for plotting
    temps.append(temp)
    errors.append(error)
    control_signals.append(control)
    last_error = error

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time, temps, label='Temperature')
plt.axhline(target_temp, color='r', linestyle='--', label='Target')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time, errors)
plt.ylabel('Error (°C)')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time, control_signals)
plt.ylabel('Heater Power (%)')
plt.xlabel('Time (s)')
plt.grid(True)

plt.tight_layout()
plt.savefig('pid_simulation.png')
print("Simulation complete! Graph saved as pid_simulation.png")
print(f"Final temperature: {temps[-1]:.2f}°C")
print(f"Final error: {errors[-1]:.2f}°C")


