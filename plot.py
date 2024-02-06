import matplotlib.pyplot as plt

# Example arrays for more points in each line
# Line 1: Points (1, 2), (2, 3), (3, 4)
x1_line = [30, 35, 40, 45, 50]
y1_line = [2.76, 4.35, 4.75, 5.34, 5.60]

# Line 2: Points (2, 2), (3, 3), (4, 4)
x2_line = [30, 35, 40, 45, 50]
y2_line = [2.9, 3.35, 3.75, 4.26, 4.30]

# Plotting the lines with more points
plt.figure()
# 'o' marker to indicate the points
plt.plot(x1_line, y1_line, marker='o', label='DDPG')
# 'x' marker to indicate the points
plt.plot(x2_line, y2_line, marker='x', label='TD3')

# Adding labels and title
plt.xlabel('Power budget (Watt)')
plt.ylabel('Optimal number of user')
plt.title('Plot of Two Lines with Multiple Points')
plt.legend()

# Display the plot
plt.show()
