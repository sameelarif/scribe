import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Import the spline functions for smoothing
from scipy.interpolate import splprep, splev

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define screen dimensions for normalization
SCREEN_WIDTH = 1920   # Replace with your screen width if different
SCREEN_HEIGHT = 1080  # Replace with your screen height if different

# ====================================
# 1. Model Definition
# ====================================

class CVAE(nn.Module):
    def __init__(self, input_size=2, condition_size=4, hidden_size=128, latent_size=32):
        super(CVAE, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size  # To access input_size in methods

        # Encoder
        self.encoder_lstm = nn.LSTM(input_size + 1, hidden_size, batch_first=True)  # +1 for time intervals
        self.fc_mu = nn.Linear(hidden_size + condition_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size + condition_size, latent_size)

        # Decoder
        # Input size: input_size + latent_size + condition_size + 1
        self.decoder_lstm = nn.LSTM(input_size + latent_size + condition_size + 1, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def encode(self, x, time_intervals, condition):
        batch_size = x.size(0)
        x = torch.cat([x, time_intervals.unsqueeze(-1)], dim=-1)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, batch_first=True, lengths=[x.size(1)] * batch_size, enforce_sorted=False)
        _, (h_n, _) = self.encoder_lstm(packed_input)
        h_n = h_n.squeeze(0)
        h_n = torch.cat([h_n, condition], dim=-1)
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition, seq_len, start_point):
        batch_size = z.size(0)
        outputs = []
        hidden = None

        # Initialize x_t with the start point
        x_t = start_point.to(z.device)  # Shape: (batch_size, input_size)

        # Use average time intervals or a constant value
        average_delta_time = 0.05  # Adjust based on your data (in seconds)
        time_intervals = torch.full((batch_size, seq_len), average_delta_time).to(z.device)

        for t in range(seq_len):
            delta_t = time_intervals[:, t].unsqueeze(1)  # Shape: (batch_size, 1)
            z_t = z  # Shape: (batch_size, latent_size)
            cond_t = condition  # Shape: (batch_size, condition_size)

            # Include x_t in the decoder input
            decoder_input = torch.cat([x_t, z_t, cond_t, delta_t], dim=-1).unsqueeze(1)
            # Shape: (batch_size, 1, input_size + latent_size + condition_size + 1)

            output, hidden = self.decoder_lstm(decoder_input, hidden)
            x_t = self.output_layer(output.squeeze(1))  # Shape: (batch_size, input_size)
            outputs.append(x_t.unsqueeze(1))

            # x_t is used in the next iteration

        outputs = torch.cat(outputs, dim=1)  # Shape: (batch_size, seq_len, input_size)
        return outputs

    def forward(self, x, time_intervals, condition, seq_len):
        # Not used during testing
        pass

# ====================================
# 2. Inference Function
# ====================================

def generate_path(model, start_point, end_point, seq_len=50):
    model.eval()
    with torch.no_grad():
        condition = torch.cat([start_point, end_point], dim=-1).unsqueeze(0).to(device)  # Shape: (1, 4)
        z = torch.randn(1, model.latent_size).to(device)  # Sample from standard normal

        generated_seq = model.decode(z, condition, seq_len, start_point.unsqueeze(0))
        generated_seq = generated_seq.squeeze(0).cpu().numpy()  # Shape: (seq_len, input_size)
        return generated_seq

# ====================================
# 3. Transformation Function
# ====================================

def transform_path_to_endpoints(path, start_point, end_point):
    """
    Transform the generated path so that it starts at start_point and ends at end_point.
    This is done by applying an affine transformation (rotation, scaling, and translation).
    """
    original_start = path[0]
    original_end = path[-1]

    # Vectors from start to end
    v_o = original_end - original_start
    v_d = end_point - start_point

    # Lengths of the vectors
    len_o = np.linalg.norm(v_o)
    len_d = np.linalg.norm(v_d)

    # Avoid division by zero
    if len_o == 0 or len_d == 0:
        # Return a straight line from start_point to end_point
        transformed_path = np.linspace(start_point, end_point, num=len(path))
        return transformed_path

    # Scale factor
    scale = len_d / len_o

    # Compute the angle between v_o and v_d
    # First, normalize the vectors
    v_o_unit = v_o / len_o
    v_d_unit = v_d / len_d

    # Compute rotation angle
    angle = np.arctan2(v_d_unit[1], v_d_unit[0]) - np.arctan2(v_o_unit[1], v_o_unit[0])

    # Build rotation matrix
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    R = np.array([[cos_theta, -sin_theta],
                  [sin_theta,  cos_theta]])

    # Apply transformation to the path
    # Shift the path to origin based on original_start
    shifted_path = path - original_start

    # Rotate
    rotated_path = shifted_path @ R.T  # Using matrix multiplication

    # Scale
    scaled_path = rotated_path * scale

    # Translate to desired start point
    transformed_path = scaled_path + start_point

    return transformed_path

# ====================================
# 4. Main Testing Function
# ====================================

def main():
    # Path to the saved model file
    MODEL_FILE = 'models/Model Final 20241031.pt'  # Replace with your model file path

    # Initialize the model architecture
    model = CVAE(input_size=2, condition_size=4, hidden_size=128, latent_size=32).to(device)

    # Load the saved model parameters
    if not os.path.exists(MODEL_FILE):
        print(f"Model file {MODEL_FILE} not found. Please check the path.")
        return

    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    print(f"Model loaded from {MODEL_FILE}")

    # Input start and end points (normalized between 0 and 1)
    # You can modify these values as needed
    start_point = np.array([0.5, 0.2], dtype=np.float32)  # Example start point
    end_point = np.array([0.5, 0.8], dtype=np.float32)    # Example end point

    start_point_tensor = torch.tensor(start_point).to(torch.float32).to(device)
    end_point_tensor = torch.tensor(end_point).to(torch.float32).to(device)

    # Generate the mouse movement path
    generated_path = generate_path(model, start_point_tensor, end_point_tensor, seq_len=50)[2:]

    # Denormalize the coordinates for visualization or further processing
    generated_path_denormalized = generated_path * np.array([SCREEN_WIDTH, SCREEN_HEIGHT])

    # Denormalize the start and end points for printing
    start_point_denormalized = start_point * np.array([SCREEN_WIDTH, SCREEN_HEIGHT])
    end_point_denormalized = end_point * np.array([SCREEN_WIDTH, SCREEN_HEIGHT])

    # Transform the generated path to match the exact input start and end points
    transformed_path = transform_path_to_endpoints(generated_path, start_point, end_point)
    transformed_path_denormalized = transformed_path * np.array([SCREEN_WIDTH, SCREEN_HEIGHT])

    # Apply smoothing to the transformed path
    # Extract x and y coordinates
    x = transformed_path_denormalized[:, 0]
    y = transformed_path_denormalized[:, 1]

    # Smooth parameter (adjust s_value to control smoothing)
    s_value = 3.0  # Increase s_value for more smoothing

    # Generate the spline representation
    tck, u = splprep([x, y], s=s_value)

    # Evaluate the spline at a new set of points
    unew = np.linspace(0, 1.0, num=100)
    x_smooth, y_smooth = splev(unew, tck)

    # Combine x and y into smoothed path
    smoothed_path = np.vstack((x_smooth, y_smooth)).T

    # Print the denormalized start and end points
    print("Start Point (Denormalized):")
    print(start_point_denormalized)
    print("End Point (Denormalized):")
    print(end_point_denormalized)

    # Print or visualize the generated paths
    print("Original Generated Path (Denormalized):")
    print(generated_path_denormalized)
    print("Transformed Generated Path (Denormalized):")
    print(transformed_path_denormalized)
    print("Smoothed Transformed Path (Denormalized):")
    print(smoothed_path)

    # Visualize the paths using matplotlib
    plt.figure(figsize=(18, 12))

    # First row: plots with markers
    # Plot 1: Original Generated Path with markers
    plt.subplot(2, 3, 1)
    plt.plot(generated_path_denormalized[:, 0], generated_path_denormalized[:, 1],
             marker='o', label='Original Generated Path with Markers')
    # Plot start and end points after the path
    plt.scatter(generated_path_denormalized[0, 0], generated_path_denormalized[0, 1],
                color='green', label='Start Point (Generated)', s=100, zorder=5)
    plt.scatter(generated_path_denormalized[-1, 0], generated_path_denormalized[-1, 1],
                color='red', label='End Point (Generated)', s=100, zorder=5)
    # Display coordinates
    plt.text(generated_path_denormalized[0, 0], generated_path_denormalized[0, 1] + 20,
             f"({generated_path_denormalized[0, 0]:.1f}, {generated_path_denormalized[0, 1]:.1f})",
             color='green')
    plt.text(generated_path_denormalized[-1, 0], generated_path_denormalized[-1, 1] + 20,
             f"({generated_path_denormalized[-1, 0]:.1f}, {generated_path_denormalized[-1, 1]:.1f})",
             color='red')
    plt.legend()
    plt.title('Original Generated Path (With Markers)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.xlim(0, SCREEN_WIDTH)
    plt.ylim(0, SCREEN_HEIGHT)
    plt.gca().invert_yaxis()
    plt.grid(True)

    # Plot 2: Transformed Generated Path with markers
    plt.subplot(2, 3, 2)
    plt.plot(transformed_path_denormalized[:, 0], transformed_path_denormalized[:, 1],
             marker='o', label='Transformed Generated Path with Markers')
    # Plot start and end points after the path
    plt.scatter(start_point_denormalized[0], start_point_denormalized[1],
                color='green', label='Start Point (Input)', s=100, zorder=5)
    plt.scatter(end_point_denormalized[0], end_point_denormalized[1],
                color='red', label='End Point (Input)', s=100, zorder=5)
    # Display coordinates
    plt.text(start_point_denormalized[0], start_point_denormalized[1] + 20,
             f"({start_point_denormalized[0]:.1f}, {start_point_denormalized[1]:.1f})",
             color='green')
    plt.text(end_point_denormalized[0], end_point_denormalized[1] + 20,
             f"({end_point_denormalized[0]:.1f}, {end_point_denormalized[1]:.1f})",
             color='red')
    plt.legend()
    plt.title('Transformed Generated Path (With Markers)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.xlim(0, SCREEN_WIDTH)
    plt.ylim(0, SCREEN_HEIGHT)
    plt.gca().invert_yaxis()
    plt.grid(True)

    # Plot 3: Smoothed Transformed Path with markers
    plt.subplot(2, 3, 3)
    plt.plot(smoothed_path[:, 0], smoothed_path[:, 1],
             marker='o', label='Smoothed Transformed Path with Markers')
    # Plot start and end points after the path
    plt.scatter(start_point_denormalized[0], start_point_denormalized[1],
                color='green', label='Start Point (Input)', s=100, zorder=5)
    plt.scatter(end_point_denormalized[0], end_point_denormalized[1],
                color='red', label='End Point (Input)', s=100, zorder=5)
    # Display coordinates
    plt.text(start_point_denormalized[0], start_point_denormalized[1] + 20,
             f"({start_point_denormalized[0]:.1f}, {start_point_denormalized[1]:.1f})",
             color='green')
    plt.text(end_point_denormalized[0], end_point_denormalized[1] + 20,
             f"({end_point_denormalized[0]:.1f}, {end_point_denormalized[1]:.1f})",
             color='red')
    plt.legend()
    plt.title('Smoothed Transformed Path (With Markers)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.xlim(0, SCREEN_WIDTH)
    plt.ylim(0, SCREEN_HEIGHT)
    plt.gca().invert_yaxis()
    plt.grid(True)

    # Second row: plots without markers
    # Plot 4: Original Generated Path without markers
    plt.subplot(2, 3, 4)
    plt.plot(generated_path_denormalized[:, 0], generated_path_denormalized[:, 1],
             label='Original Generated Path')
    # Plot start and end points after the path
    plt.scatter(generated_path_denormalized[0, 0], generated_path_denormalized[0, 1],
                color='green', label='Start Point (Generated)', s=100, zorder=5)
    plt.scatter(generated_path_denormalized[-1, 0], generated_path_denormalized[-1, 1],
                color='red', label='End Point (Generated)', s=100, zorder=5)
    # Display coordinates
    plt.text(generated_path_denormalized[0, 0], generated_path_denormalized[0, 1] + 20,
             f"({generated_path_denormalized[0, 0]:.1f}, {generated_path_denormalized[0, 1]:.1f})",
             color='green')
    plt.text(generated_path_denormalized[-1, 0], generated_path_denormalized[-1, 1] + 20,
             f"({generated_path_denormalized[-1, 0]:.1f}, {generated_path_denormalized[-1, 1]:.1f})",
             color='red')
    plt.legend()
    plt.title('Original Generated Path (Line Only)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.xlim(0, SCREEN_WIDTH)
    plt.ylim(0, SCREEN_HEIGHT)
    plt.gca().invert_yaxis()
    plt.grid(True)

    # Plot 5: Transformed Generated Path without markers
    plt.subplot(2, 3, 5)
    plt.plot(transformed_path_denormalized[:, 0], transformed_path_denormalized[:, 1],
             label='Transformed Generated Path')
    # Plot start and end points after the path
    plt.scatter(start_point_denormalized[0], start_point_denormalized[1],
                color='green', label='Start Point (Input)', s=100, zorder=5)
    plt.scatter(end_point_denormalized[0], end_point_denormalized[1],
                color='red', label='End Point (Input)', s=100, zorder=5)
    # Display coordinates
    plt.text(start_point_denormalized[0], start_point_denormalized[1] + 20,
             f"({start_point_denormalized[0]:.1f}, {start_point_denormalized[1]:.1f})",
             color='green')
    plt.text(end_point_denormalized[0], end_point_denormalized[1] + 20,
             f"({end_point_denormalized[0]:.1f}, {end_point_denormalized[1]:.1f})",
             color='red')
    plt.legend()
    plt.title('Transformed Generated Path (Line Only)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.xlim(0, SCREEN_WIDTH)
    plt.ylim(0, SCREEN_HEIGHT)
    plt.gca().invert_yaxis()
    plt.grid(True)

    # Plot 6: Smoothed Transformed Path without markers
    plt.subplot(2, 3, 6)
    plt.plot(smoothed_path[:, 0], smoothed_path[:, 1],
             label='Smoothed Transformed Path')
    # Plot start and end points after the path
    plt.scatter(start_point_denormalized[0], start_point_denormalized[1],
                color='green', label='Start Point (Input)', s=100, zorder=5)
    plt.scatter(end_point_denormalized[0], end_point_denormalized[1],
                color='red', label='End Point (Input)', s=100, zorder=5)
    # Display coordinates
    plt.text(start_point_denormalized[0], start_point_denormalized[1] + 20,
             f"({start_point_denormalized[0]:.1f}, {start_point_denormalized[1]:.1f})",
             color='green')
    plt.text(end_point_denormalized[0], end_point_denormalized[1] + 20,
             f"({end_point_denormalized[0]:.1f}, {end_point_denormalized[1]:.1f})",
             color='red')
    plt.legend()
    plt.title('Smoothed Transformed Path (Line Only)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.xlim(0, SCREEN_WIDTH)
    plt.ylim(0, SCREEN_HEIGHT)
    plt.gca().invert_yaxis()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
