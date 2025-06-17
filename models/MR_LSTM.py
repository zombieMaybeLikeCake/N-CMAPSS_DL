import torch
import torch.nn as nn
import pywt
import numpy as np
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_resolutions):
        super(Model, self).__init__()
        self.num_resolutions = num_resolutions

        # First-layer LSTMs (one for each resolution)
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True) for _ in range(num_resolutions)
        ])

        # Attention mechanism
        self.attention_fc = nn.Linear(hidden_dim, 1)

        # Second-layer LSTM
        self.final_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, x):
        # Multi-resolution inputs
        multi_res_inputs = [x] * self.num_resolutions

        # Pass through LSTM layers
        lstm_outputs = []
        for i, lstm in enumerate(self.lstm_layers):
            out, _ = lstm(multi_res_inputs[i])
            lstm_outputs.append(out)

        # Attention mechanism
        attention_scores = [torch.tanh(self.attention_fc(out)) for out in lstm_outputs]
        attention_weights = [torch.softmax(score, dim=1) for score in attention_scores]
        weighted_outputs = [weights * out for weights, out in zip(attention_weights, lstm_outputs)]
        combined_representation = torch.stack(weighted_outputs, dim=1).sum(dim=1)

        # Final LSTM layer
        final_output, _ = self.final_lstm(combined_representation)

        # Fully connected layers
        x = torch.relu(self.fc1(final_output[:, -1, :]))
        x = torch.relu(self.fc2(x))
        predictions = self.fc3(x)

        return predictions