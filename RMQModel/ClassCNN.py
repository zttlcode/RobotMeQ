import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_pos


class Model(nn.Module):
    """
    Classic CNN model for time series forecasting.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # CNN Layers
        self.conv1 = nn.Conv1d(in_channels=configs.d_model, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * configs.seq_len, 512)
        self.fc2 = nn.Linear(512, configs.c_out)

        # Task-specific layers
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)

        # CNN Layers
        enc_out = enc_out.transpose(1, 2)  # [B, D, L] -> [B, L, D]
        x = F.relu(self.conv1(enc_out))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten and Fully Connected layers
        x = x.flatten(start_dim=1)  # Flatten the output to feed into FC layers
        x = F.relu(self.fc1(x))
        dec_out = self.fc2(x)

        return dec_out  # [B, L, D]

    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)

        # CNN Layers
        enc_out = enc_out.transpose(1, 2)  # [B, D, L] -> [B, L, D]
        x = F.relu(self.conv1(enc_out))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten and Fully Connected layers
        x = x.flatten(start_dim=1)  # Flatten the output to feed into FC layers
        x = self.fc1(x)
        dec_out = self.fc2(x)

        dec_out = dec_out * std_enc + mean_enc
        return dec_out  # [B, L, D]

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # CNN Layers
        enc_out = enc_out.transpose(1, 2)  # [B, D, L] -> [B, L, D]
        x = F.relu(self.conv1(enc_out))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten and Fully Connected layers
        x = x.flatten(start_dim=1)  # Flatten the output to feed into FC layers
        x = self.fc2(x)

        return x

    def anomaly_detection(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)

        # CNN Layers
        enc_out = enc_out.transpose(1, 2)  # [B, D, L] -> [B, L, D]
        x = F.relu(self.conv1(enc_out))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten and Fully Connected layers
        x = x.flatten(start_dim=1)  # Flatten the output to feed into FC layers
        x = self.fc2(x)

        return x

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)

        # CNN Layers
        enc_out = enc_out.transpose(1, 2)  # [B, D, L] -> [B, L, D]
        x = F.relu(self.conv1(enc_out))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten and Fully Connected layers
        x = x.flatten(start_dim=1)  # Flatten the output to feed into FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'short_term_forecast':
            dec_out = self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
