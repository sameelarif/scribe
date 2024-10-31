import os
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend suitable for scripts and web servers
import matplotlib.pyplot as plt
import io
import base64

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define screen dimensions for normalization
SCREEN_WIDTH = 1920   # Adjust to your screen width if different
SCREEN_HEIGHT = 1080  # Adjust to your screen height if different

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

# ====================================
# 2. Inference and Transformation Functions
# ====================================

def generate_path(model, start_point, end_point, seq_len=50):
    model.eval()
    with torch.no_grad():
        condition = torch.cat([start_point, end_point], dim=-1).unsqueeze(0).to(device)  # Shape: (1, 4)
        z = torch.randn(1, model.latent_size).to(device)  # Sample from standard normal

        generated_seq = model.decode(z, condition, seq_len, start_point.unsqueeze(0))
        generated_seq = generated_seq.squeeze(0).cpu().numpy()  # Shape: (seq_len, input_size)
        return generated_seq

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
# 3. Flask App Setup
# ====================================

app = Flask(__name__)

# Load the trained model
MODEL_FILE = '/Users/sameel/Documents/GitHub/scribe/puppeteer-scribe/model/model-20241031.pt'  # Replace with your model file path

# Initialize the model architecture
model = CVAE(input_size=2, condition_size=4, hidden_size=128, latent_size=32).to(device)

# Load the saved model parameters
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file {MODEL_FILE} not found. Please check the path.")

model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
print(f"Model loaded from {MODEL_FILE}")

# ====================================
# 4. API Endpoint
# ====================================

@app.route('/generate_path', methods=['POST'])
def api_generate_path():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided.'}), 400

    try:
        # Extract start and end points
        start_point = data['start_point']  # Expected format: [x, y]
        end_point = data['end_point']      # Expected format: [x, y]
        visualize = data.get('visualize', False)  # Optional key to generate plot

        # Validate input
        if not (isinstance(start_point, list) and isinstance(end_point, list)):
            return jsonify({'error': 'Start and end points must be lists of [x, y].'}), 400
        if not (len(start_point) == 2 and len(end_point) == 2):
            return jsonify({'error': 'Start and end points must have two coordinates [x, y].'}), 400

        # Normalize the coordinates
        start_point_normalized = np.array([
            start_point[0] / SCREEN_WIDTH,
            start_point[1] / SCREEN_HEIGHT
        ], dtype=np.float32)
        end_point_normalized = np.array([
            end_point[0] / SCREEN_WIDTH,
            end_point[1] / SCREEN_HEIGHT
        ], dtype=np.float32)

        start_point_tensor = torch.tensor(start_point_normalized).to(torch.float32).to(device)
        end_point_tensor = torch.tensor(end_point_normalized).to(torch.float32).to(device)

        # Generate the mouse movement path
        generated_path = generate_path(model, start_point_tensor, end_point_tensor, seq_len=50)

        # Transform the path to match the exact input start and end points
        transformed_path = transform_path_to_endpoints(generated_path, start_point_normalized, end_point_normalized)

        # Denormalize the transformed path back to screen coordinates
        transformed_path_denormalized = transformed_path * np.array([SCREEN_WIDTH, SCREEN_HEIGHT])

        # Convert the path to a list of [x, y] coordinates
        path_list = transformed_path_denormalized.tolist()

        response_data = {'path': path_list}

        # If visualization is requested
        if visualize:
            # Generate the plot
            plt.figure(figsize=(6, 6))

            # Plot the transformed path
            plt.plot(transformed_path_denormalized[:, 0], transformed_path_denormalized[:, 1],
                     marker='o', label='Transformed Generated Path')

            # Plot start and end points
            plt.scatter(start_point[0], start_point[1],
                        color='green', label='Start Point (Input)', s=100, zorder=5)
            plt.scatter(end_point[0], end_point[1],
                        color='red', label='End Point (Input)', s=100, zorder=5)

            # Display coordinates
            plt.text(start_point[0], start_point[1] + 20,
                     f"({start_point[0]:.1f}, {start_point[1]:.1f})",
                     color='green')
            plt.text(end_point[0], end_point[1] + 20,
                     f"({end_point[0]:.1f}, {end_point[1]:.1f})",
                     color='red')

            plt.legend()
            plt.title('Transformed Generated Path')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.xlim(0, SCREEN_WIDTH)
            plt.ylim(0, SCREEN_HEIGHT)
            plt.gca().invert_yaxis()
            plt.grid(True)

            # Save the plot to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()

            # Encode the image in base64
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            # Add the image to the response
            response_data['plot'] = img_base64

        return jsonify(response_data), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

# ====================================
# 5. Run the App
# ====================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)