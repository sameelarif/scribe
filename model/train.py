import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime  # For timestamp

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define screen dimensions for normalization
SCREEN_WIDTH = 1920   # Replace with your screen width
SCREEN_HEIGHT = 1080  # Replace with your screen height

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================================
# 1. Data Loading and Preprocessing
# ====================================

class MouseMovementDataset(Dataset):
    def __init__(self, data_file, filter_data=False, deviation_threshold=0.05):
        """
        Args:
            data_file (str): Path to the data file.
            filter_data (bool): Whether to apply data filtering/cleaning.
            deviation_threshold (float): Threshold for deviation when filtering data.
        """
        self.data = []
        self.load_data(data_file)
        self.normalize_data()
        if filter_data:
            self.clean_data(deviation_threshold)

    def load_data(self, data_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)

    def normalize_data(self):
        for record in self.data:
            # Normalize start point
            x1 = record['start']['x']
            y1 = record['start']['y']
            x2 = record['end']['x']
            y2 = record['end']['y']

            record['start']['x'] = x1 / SCREEN_WIDTH
            record['start']['y'] = y1 / SCREEN_HEIGHT
            # Normalize end point
            record['end']['x'] = x2 / SCREEN_WIDTH
            record['end']['y'] = y2 / SCREEN_HEIGHT

            # Normalize path points and compute time intervals
            path_points = []
            prev_time = None
            for point in record['path']:
                x = point['x']
                y = point['y']
                t = int(point['timestamp'])

                x_norm = x / SCREEN_WIDTH
                y_norm = y / SCREEN_HEIGHT

                # Compute delta_time
                if prev_time is None:
                    delta_time = 0.0
                else:
                    delta_time = (t - prev_time) / 1000.0  # Convert ms to seconds

                prev_time = t

                # Collect the normalized point and delta_time
                path_points.append({'x': x_norm, 'y': y_norm, 'delta_time': delta_time})

            # If the first and second points are very distant, remove the first point
            if len(path_points) >= 2:
                x0, y0 = path_points[0]['x'], path_points[0]['y']
                x1, y1 = path_points[1]['x'], path_points[1]['y']
                dx = x1 - x0
                dy = y1 - y0
                distance = np.sqrt(dx**2 + dy**2)
                threshold = 0.05  # Threshold can be adjusted as needed
                if distance > threshold:
                    # Remove the first point
                    path_points.pop(0)
                    # Adjust delta_time of the new first point
                    path_points[0]['delta_time'] = 0.0

            # Update record['path'] with the modified path_points
            record['path'] = path_points

    def clean_data(self, deviation_threshold):
        """
        Remove samples where the path deviates significantly from a straight line.
        """
        cleaned_data = []
        for record in self.data:
            start = np.array([record['start']['x'], record['start']['y']])
            end = np.array([record['end']['x'], record['end']['y']])
            path = np.array([[p['x'], p['y']] for p in record['path']])

            if len(path) < 2:
                continue  # Skip if path is too short to evaluate

            # Calculate deviations
            deviations = self.calculate_deviation(start, end, path)

            # Mean squared deviation
            msd = np.mean(deviations ** 2)

            if msd <= deviation_threshold:
                cleaned_data.append(record)
            else:
                continue  # Exclude sample

        self.data = cleaned_data

    def calculate_deviation(self, start, end, path):
        """
        Calculate the perpendicular distance of each path point from the straight line between start and end.
        """
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return np.zeros(len(path))
        line_unitvec = line_vec / line_len

        path_vecs = path - start
        projections = np.dot(path_vecs, line_unitvec)
        projections = np.clip(projections, 0, line_len)
        projections_vec = np.outer(projections, line_unitvec)
        closest_points = start + projections_vec
        deviations = np.linalg.norm(path - closest_points, axis=1)
        return deviations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        # Start and end points
        start_point = np.array([record['start']['x'], record['start']['y']], dtype=np.float32)
        end_point = np.array([record['end']['x'], record['end']['y']], dtype=np.float32)
        # Path points and time intervals
        path_points = []
        time_intervals = []
        for p in record['path']:
            path_points.append([p['x'], p['y']])
            time_intervals.append(p['delta_time'])
        path_points = np.array(path_points, dtype=np.float32)  # Shape: (seq_len, 2)
        time_intervals = np.array(time_intervals, dtype=np.float32)  # Shape: (seq_len,)
        return {
            'start_point': start_point,    # Shape: (2,)
            'end_point': end_point,        # Shape: (2,)
            'path': path_points,           # Shape: (seq_len, 2)
            'time_intervals': time_intervals  # Shape: (seq_len,)
        }

# ====================================
# 2. Dataset and DataLoader Preparation
# ====================================

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    """
    start_points = []
    end_points = []
    paths = []
    time_intervals = []
    seq_lengths = []

    for item in batch:
        start_points.append(item['start_point'])
        end_points.append(item['end_point'])
        paths.append(torch.tensor(item['path']))
        time_intervals.append(torch.tensor(item['time_intervals']))
        seq_lengths.append(len(item['path']))

    # Pad sequences to the maximum length in the batch
    max_seq_len = max(seq_lengths)
    padded_paths = torch.zeros(len(batch), max_seq_len, 2)
    padded_times = torch.zeros(len(batch), max_seq_len)

    for i, (path, times) in enumerate(zip(paths, time_intervals)):
        seq_len = seq_lengths[i]
        padded_paths[i, :seq_len, :] = path
        padded_times[i, :seq_len] = times

    return {
        'start_points': torch.tensor(start_points),     # Shape: (batch_size, 2)
        'end_points': torch.tensor(end_points),         # Shape: (batch_size, 2)
        'paths': padded_paths,                          # Shape: (batch_size, max_seq_len, 2)
        'time_intervals': padded_times,                 # Shape: (batch_size, max_seq_len)
        'seq_lengths': seq_lengths
    }

# ====================================
# 3. Model Definition
# ====================================

class CVAE(nn.Module):
    def __init__(self, input_size=2, condition_size=4, hidden_size=128, latent_size=32):
        super(CVAE, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size  # Added to access input_size in methods

        # Encoder
        self.encoder_lstm = nn.LSTM(input_size + 1, hidden_size, batch_first=True)  # +1 for time intervals
        self.fc_mu = nn.Linear(hidden_size + condition_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size + condition_size, latent_size)

        # Decoder
        # Corrected input size: input_size + latent_size + condition_size + 1
        self.decoder_lstm = nn.LSTM(input_size + latent_size + condition_size + 1, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def encode(self, x, time_intervals, condition):
        batch_size = x.size(0)
        x = torch.cat([x, time_intervals.unsqueeze(-1)], dim=-1)
        packed_input = nn.utils.rnn.pack_padded_sequence(x, batch_first=True, lengths=[x.size(1)] * batch_size, enforce_sorted=False)
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

    def decode(self, z, condition, x_seq, time_intervals, seq_len):
        batch_size = z.size(0)
        outputs = []
        hidden = None

        # Initialize x_t with zeros or start tokens
        x_t = torch.zeros(batch_size, self.input_size).to(z.device)

        for t in range(seq_len):
            z_t = z  # Shape: (batch_size, latent_size)
            cond_t = condition  # Shape: (batch_size, condition_size)
            delta_t = time_intervals[:, t].unsqueeze(1)  # Shape: (batch_size, 1)

            # Include x_t in the decoder input
            decoder_input = torch.cat([x_t, z_t, cond_t, delta_t], dim=-1).unsqueeze(1)
            # Shape: (batch_size, 1, input_size + latent_size + condition_size + 1)

            output, hidden = self.decoder_lstm(decoder_input, hidden)
            x_t = self.output_layer(output.squeeze(1))  # Shape: (batch_size, input_size)
            outputs.append(x_t.unsqueeze(1))

            # Teacher forcing: use ground truth x_t during training
            if self.training and x_seq is not None:
                x_t = x_seq[:, t, :]

        outputs = torch.cat(outputs, dim=1)  # Shape: (batch_size, seq_len, input_size)
        return outputs

    def forward(self, x, time_intervals, condition, seq_len):
        mu, logvar = self.encode(x, time_intervals, condition)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, condition, x_seq=x, time_intervals=time_intervals, seq_len=seq_len)
        return recon_x, mu, logvar

# ====================================
# 4. Training Loop
# ====================================

def loss_function(recon_x, x, mu, logvar, seq_lengths):
    MSE = 0
    total_length = sum(seq_lengths)
    for i in range(len(seq_lengths)):
        seq_len = seq_lengths[i]
        MSE += nn.functional.mse_loss(recon_x[i, :seq_len, :], x[i, :seq_len, :], reduction='sum')
    MSE /= total_length
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= len(seq_lengths)
    return MSE + KLD

def train_model(model, dataloader, num_epochs=20, learning_rate=0.001, model_save_path='models'):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    # Ensure the models directory exists
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            start_points = batch['start_points'].to(device)
            end_points = batch['end_points'].to(device)
            paths = batch['paths'].to(device)
            time_intervals = batch['time_intervals'].to(device)
            seq_lengths = batch['seq_lengths']
            batch_size = paths.size(0)
            max_seq_len = paths.size(1)

            condition = torch.cat([start_points, end_points], dim=-1)  # Shape: (batch_size, 4)
            optimizer.zero_grad()
            recon_paths, mu, logvar = model(paths, time_intervals, condition, max_seq_len)
            loss = loss_function(recon_paths, paths, mu, logvar, seq_lengths)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

    # Save the finalized model after training
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'model_final_{timestamp}.pt'
    model_path = os.path.join(model_save_path, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f'Final model saved to {model_path}')

# ====================================
# 5. Inference Function
# ====================================

def generate_path(model, start_point, end_point, seq_len=50):
    model.eval()
    with torch.no_grad():
        condition = torch.cat([start_point, end_point], dim=-1).unsqueeze(0).to(device)  # Shape: (1, 4)
        z = torch.randn(1, model.latent_size).to(device)  # Sample from standard normal

        # Use average time intervals or a constant value
        average_delta_time = 0.05  # Adjust based on your data (in seconds)
        time_intervals = torch.full((1, seq_len), average_delta_time).to(device)

        outputs = []
        hidden = None

        # Initialize x_t with the start point
        x_t = start_point.unsqueeze(0).to(device)  # Shape: (1, 2)

        for t in range(seq_len):
            delta_t = time_intervals[:, t].unsqueeze(1)  # Shape: (1, 1)

            # Include x_t in the decoder input
            decoder_input = torch.cat([x_t, z, condition, delta_t], dim=-1).unsqueeze(1)
            # Shape: (1, 1, input_size + latent_size + condition_size + 1)

            output, hidden = model.decoder_lstm(decoder_input, hidden)
            x_t = model.output_layer(output.squeeze(1))  # Shape: (1, input_size)
            outputs.append(x_t.squeeze(0).cpu().numpy())

            # x_t is used in the next iteration

        generated_path = np.array(outputs)
        return generated_path

# ====================================
# 6. Main Function
# ====================================

def main():
    # Path to your data file
    DATA_FILE = 'main.data.json'  # Replace with your data file path

    # Create dataset and dataloader
    dataset = MouseMovementDataset(DATA_FILE, filter_data=True, deviation_threshold=0.05)
    if len(dataset) == 0:
        print("No data available after filtering. Adjust your deviation_threshold or check your data.")
        return

    print("Remaining data length after filtering:", len(dataset))

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Initialize the model
    model = CVAE(input_size=2, condition_size=4, hidden_size=128, latent_size=32).to(device)

    # Train the model
    train_model(model, dataloader, num_epochs=20, learning_rate=0.001, model_save_path='models')

    # Example inference
    # Use normalized coordinates for start and end points
    start_point = np.array([0.4, 0.4], dtype=np.float32)
    end_point = np.array([0.4, 0.6], dtype=np.float32)

    start_point_tensor = torch.tensor(start_point).to(torch.float32).to(device)
    end_point_tensor = torch.tensor(end_point).to(torch.float32).to(device)

    generated_path = generate_path(model, start_point_tensor, end_point_tensor, seq_len=50)

    # Denormalize the coordinates for visualization or further processing
    generated_path_denormalized = generated_path * np.array([SCREEN_WIDTH, SCREEN_HEIGHT])

    # Print or visualize the generated path
    print("Generated Path:")
    print(generated_path_denormalized)

    # Denormalize the start and end points for printing
    start_point_denormalized = start_point * np.array([SCREEN_WIDTH, SCREEN_HEIGHT])
    end_point_denormalized = end_point * np.array([SCREEN_WIDTH, SCREEN_HEIGHT])

    # Print the denormalized start and end points
    print("Start Point (Denormalized):")
    print(start_point_denormalized)
    print("End Point (Denormalized):")
    print(end_point_denormalized)

    # Optionally, visualize the path using matplotlib
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(generated_path_denormalized[:, 0], generated_path_denormalized[:, 1], marker='o', label='Generated Path')
        plt.scatter(start_point_denormalized[0], start_point_denormalized[1], color='green', label='Start Point')
        plt.scatter(end_point_denormalized[0], end_point_denormalized[1], color='red', label='End Point')
        plt.legend()
        plt.title('Generated Mouse Movement Path')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.show()
    except ImportError:
        print("Matplotlib is not installed. Install it to visualize the path.")

if __name__ == '__main__':
    main()
