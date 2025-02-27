import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Load the CSV file
file_path = "/mnt/storage/nacc_sub/mm_dit_con_vae_500/tmp/debug_scales.csv"  # Update with your actual file path
df = pd.read_csv(file_path, header=None, names=["Step", "Latent Mean", "Latent Std", "Real Mean", "Real Std"])

# Compute the differences
df["Mean Difference"] = df["Real Mean"] - df["Latent Mean"]
df["Std Difference"] = df["Real Std"] - df["Latent Std"]

# Function for high-resolution smoothing
def smooth_curve(x, y, num_points=1000):  # Increased to 1000 points
    x_new = np.linspace(x.min(), x.max(), num_points)  # More points for interpolation
    spline = make_interp_spline(x, y, k=3)  # Cubic spline interpolation
    y_smooth = spline(x_new)
    return x_new, y_smooth

# Generate smooth curves
steps_smooth, latent_mean_smooth = smooth_curve(df["Step"], df["Latent Mean"])
_, latent_std_smooth = smooth_curve(df["Step"], df["Latent Std"])
_, real_mean_smooth = smooth_curve(df["Step"], df["Real Mean"])
_, real_std_smooth = smooth_curve(df["Step"], df["Real Std"])
_, mean_diff_smooth = smooth_curve(df["Step"], df["Mean Difference"])
_, std_diff_smooth = smooth_curve(df["Step"], df["Std Difference"])

# Create the figure
plt.figure(figsize=(12, 10))

# First plot: original values (smoothed)
plt.subplot(2, 1, 1)
plt.plot(steps_smooth, latent_mean_smooth, label="Latent Mean", linestyle='-', color='blue')
plt.plot(steps_smooth, latent_std_smooth, label="Latent Std", linestyle='-', color='cyan')
plt.plot(steps_smooth, real_mean_smooth, label="Real Mean", linestyle='--', color='red')
plt.plot(steps_smooth, real_std_smooth, label="Real Std", linestyle='--', color='orange')
plt.xlabel("Training Step")
plt.ylabel("Values")
plt.title("Latent and Real Distributions Over Training Steps (Ultra-Smooth)")
plt.legend()
plt.grid(True)

# Second plot: differences (smoothed)
plt.subplot(2, 1, 2)
plt.plot(steps_smooth, mean_diff_smooth, label="Mean Difference (Real - Latent)", linestyle='-', color='purple')
plt.plot(steps_smooth, std_diff_smooth, label="Std Difference (Real - Latent)", linestyle='-', color='green')
plt.xlabel("Training Step")
plt.ylabel("Difference")
plt.title("Difference Between Real and Latent Distributions (Ultra-Smooth)")
plt.legend()
plt.grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Load the CSV file
file_path = "your_log_file.csv"  # Update with your actual file path
df = pd.read_csv(file_path, header=None, names=["Step", "Latent Mean", "Latent Std", "Real Mean", "Real Std"])

# Compute the differences
df["Mean Difference"] = df["Real Mean"] - df["Latent Mean"]
df["Std Difference"] = df["Real Std"] - df["Latent Std"]

# Function for high-resolution smoothing
def smooth_curve(x, y, num_points=1000):
    x_new = np.linspace(x.min(), x.max(), num_points)  # More points for interpolation
    spline = make_interp_spline(x, y, k=3)  # Cubic spline interpolation
    y_smooth = spline(x_new)
    return x_new, y_smooth

# Generate smooth curves
steps_smooth, latent_mean_smooth = smooth_curve(df["Step"], df["Latent Mean"])
_, latent_std_smooth = smooth_curve(df["Step"], df["Latent Std"])
_, real_mean_smooth = smooth_curve(df["Step"], df["Real Mean"])
_, real_std_smooth = smooth_curve(df["Step"], df["Real Std"])
_, mean_diff_smooth = smooth_curve(df["Step"], df["Mean Difference"])
_, std_diff_smooth = smooth_curve(df["Step"], df["Std Difference"])

# Compute trend lines (moving averages)
window_size = 50  # Adjust for smoother or rougher trends
df["Mean Difference Trend"] = df["Mean Difference"].rolling(window=window_size).mean()
df["Std Difference Trend"] = df["Std Difference"].rolling(window=window_size).mean()

# Smooth the trend lines for better visualization
steps_trend_smooth, mean_diff_trend_smooth = smooth_curve(df["Step"], df["Mean Difference Trend"].dropna())
_, std_diff_trend_smooth = smooth_curve(df["Step"], df["Std Difference Trend"].dropna())

# Create the figure
plt.figure(figsize=(12, 10))

# First plot: original values (smoothed)
plt.subplot(2, 1, 1)
plt.plot(steps_smooth, latent_mean_smooth, label="Latent Mean", linestyle='-', color='blue')
plt.plot(steps_smooth, latent_std_smooth, label="Latent Std", linestyle='-', color='cyan')
plt.plot(steps_smooth, real_mean_smooth, label="Real Mean", linestyle='--', color='red')
plt.plot(steps_smooth, real_std_smooth, label="Real Std", linestyle='--', color='orange')
plt.xlabel("Training Step")
plt.ylabel("Values")
plt.title("Latent and Real Distributions Over Training Steps (Ultra-Smooth)")
plt.legend()
plt.grid(True)

# Second plot: differences (smoothed) with trend lines
plt.subplot(2, 1, 2)
plt.plot(steps_smooth, mean_diff_smooth, label="Mean Difference (Real - Latent)", linestyle='-', color='purple')
plt.plot(steps_smooth, std_diff_smooth, label="Std Difference (Real - Latent)", linestyle='-', color='green')
plt.plot(steps_trend_smooth, mean_diff_trend_smooth, label="Mean Difference Trend", linestyle='--', color='darkviolet', linewidth=2)
plt.plot(steps_trend_smooth, std_diff_trend_smooth, label="Std Difference Trend", linestyle='--', color='darkgreen', linewidth=2)
plt.xlabel("Training Step")
plt.ylabel("Difference")
plt.title("Difference Between Real and Latent Distributions (Ultra-Smooth) with Trends")
plt.legend()
plt.grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()