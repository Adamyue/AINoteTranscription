# -*- coding: utf-8 -*-
"""
GRU + Attention Model for Note Segmentation
Based on A-GRCNN paper architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionMechanism(nn.Module):
    """
    Self-Attention mechanism for focusing on important temporal features
    """
    def __init__(self, feature_dim, num_heads=8):
        super(AttentionMechanism, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.attention_dropout = nn.Dropout(0.1)
        self.output = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, x):
        batch_size, seq_len, feature_dim = x.size()
        
        # Compute Q, K, V
        Q = self.query(x)  # (batch, seq_len, feature_dim)
        K = self.key(x)    # (batch, seq_len, feature_dim)
        V = self.value(x)  # (batch, seq_len, feature_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch, num_heads, seq_len, seq_len)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)  # (batch, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, feature_dim)
        
        # Final output projection
        output = self.output(attention_output)
        
        return output, attention_weights


class GRU_Attention_Model(nn.Module):
    """
    GRU + Attention Model for Note Segmentation
    Processes CFP features with 1D CNN + GRU + Attention
    """
    
    def __init__(self, conv1_in_channel=9, num_classes=6, hidden_dim=256, num_layers=2):
        super(GRU_Attention_Model, self).__init__()
        
        self.conv1_in_channel = conv1_in_channel
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Feature extraction with 1D CNN (processes frequency dimension)
        # Input: (batch, 9, 522, 19)
        # After conv: (batch, channels, freq, time)
        
        self.conv1 = nn.Conv1d(conv1_in_channel, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        
        # After convolutions: (batch, 256, freq_dim, 19)
        # We'll flatten the frequency dimension
        # Note: freq_dim will be calculated dynamically in forward pass
        
        # Project to feature dimension for GRU (will be set in forward pass)
        self.freq_projection = None
        
        # GRU layers for temporal modeling
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, 
                          batch_first=True, bidirectional=False, dropout=0.2 if num_layers > 1 else 0)
        
        # Attention mechanism
        self.attention = AttentionMechanism(hidden_dim, num_heads=8)
        
        # Final classification layers
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: (batch, 9, 522, 19) - CFP features
        Returns:
            output: (batch, 6) - note segmentation predictions
        """
        batch_size = x.size(0)
        time_frames = x.size(3)  # 19
        
        # Process each time frame independently through CNN
        # Reshape to process frequency dimension: (batch * time, channels, freq)
        x = x.permute(0, 3, 1, 2)  # (batch, 19, 9, 522)
        x = x.reshape(batch_size * time_frames, x.size(2), x.size(3))  # (batch*19, 9, 522)
        
        # 1D CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))  # (batch*19, 32, 522)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch*19, 64, 261)
        x = F.relu(self.bn3(self.conv3(x)))  # (batch*19, 128, 131)
        x = F.relu(self.bn4(self.conv4(x)))  # (batch*19, 256, 66)
        
        # Flatten frequency dimension
        x = x.reshape(batch_size * time_frames, -1)  # (batch*19, 256*freq_dim)
        
        # Create projection layer if not exists (first forward pass)
        if self.freq_projection is None:
            freq_feature_dim = x.size(1)
            self.freq_projection = nn.Linear(freq_feature_dim, self.hidden_dim).to(x.device)
        
        # Project to hidden dimension
        x = self.freq_projection(x)  # (batch*19, hidden_dim)
        
        # Reshape for GRU: (batch, time, features)
        x = x.reshape(batch_size, time_frames, self.hidden_dim)
        
        # GRU for temporal modeling
        gru_out, _ = self.gru(x)  # (batch, time, hidden_dim)
        
        # Apply attention mechanism
        attended_out, _ = self.attention(gru_out)  # (batch, time, hidden_dim)
        
        # Global average pooling over time dimension
        # We want to aggregate temporal information
        pooled = attended_out.mean(dim=1)  # (batch, hidden_dim)
        
        # Classification layers
        out = self.fc1(pooled)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


# Alternative simpler architecture for comparison
class Simplified_GRU_Attention_Model(nn.Module):
    """
    Simplified GRU + Attention Model
    Processes CFP features with 1D CNN + GRU + Attention
    """
    
    def __init__(self, conv1_in_channel=9, num_classes=6, hidden_dim=128):
        super(Simplified_GRU_Attention_Model, self).__init__()
        
        self.conv1_in_channel = conv1_in_channel
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Simpler feature extraction
        self.conv1 = nn.Conv1d(conv1_in_channel, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Project to hidden dimension (will be calculated dynamically)
        self.freq_projection = None
        
        # GRU layer
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        
        # Attention
        self.attention = AttentionMechanism(hidden_dim, num_heads=4)
        
        # Output layers
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        batch_size = x.size(0)
        time_frames = x.size(3)  # 19
        
        # Process each time frame
        x = x.permute(0, 3, 1, 2)  # (batch, 19, 9, 522)
        x = x.reshape(batch_size * time_frames, x.size(2), x.size(3))
        
        # CNN features
        x = F.relu(self.bn1(self.conv1(x)))  # (batch*19, 64, 261)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch*19, 128, 131)
        
        # Flatten and project
        x = x.reshape(batch_size * time_frames, -1)
        
        # Create projection layer if not exists (first forward pass)
        if self.freq_projection is None:
            freq_feature_dim = x.size(1)
            self.freq_projection = nn.Linear(freq_feature_dim, self.hidden_dim).to(x.device)
        
        x = self.freq_projection(x)
        
        # Reshape for GRU
        x = x.reshape(batch_size, time_frames, self.hidden_dim)
        
        # GRU
        gru_out, _ = self.gru(x)
        
        # Attention
        attended_out, _ = self.attention(gru_out)
        
        # Pool and classify
        pooled = attended_out.mean(dim=1)
        out = self.fc_out(pooled)
        
        return out

