import matplotlib.pyplot as plt

# Data

x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
y1 = [50, 50, 50, 43, 35, 30, 26, 23, 21, 19]
y2 = [50, 50, 50, 44, 36, 31, 27, 24, 22, 20]
y3 = [50, 50, 50, 46, 38, 32, 28, 25, 22, 20]
y4 = [50, 50, 50, 47, 39, 33, 29, 25, 23, 21]

# Plotting
plt.plot(x, y1, color='blue', label='Power 1W')
plt.plot(x, y2, color='red', label='Power 2W')
plt.plot(x, y3, color='purple', label='Power 5W')
plt.plot(x, y4, color='black', label='Power 10W')
plt.grid()
# Labels and title
plt.xlabel('Threshold Value')
plt.ylabel('Served User Number')
plt.title('TD3 Total User Number:50 Radius:300m Attitude:200m')
plt.legend()

# Show plot
plt.show()
