import matplotlib.pyplot as plt

# Data
labels = ["1", "2"  , "3", "4", "5", "6"]
values = [0.30, 0.07, 0.02, 0.35, 0.01, 0.11]

# Figure and axes
fig, ax = plt.subplots()
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

# Bar plot (cyan bars)
ax.bar(labels, values, color="cyan")

# Styling for black background
ax.tick_params(colors="white")
ax.spines["bottom"].set_color("white")
ax.spines["left"].set_color("white")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel("Value", color="white")

plt.show()
