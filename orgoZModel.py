import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class T5Full(nn.Module):
    def __init__(self, base_model, base_model_output_size, n_classes=1, drop_rate=0.1, pooling='cls'):
        super(T5Full, self).__init__()
        D_in, D_out = base_model_output_size, n_classes
        self.model = base_model
        self.dropout = nn.Dropout(drop_rate)
        self.pooling = pooling

        # Instantiate a linear regressor
        self.linear_regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out)
        )
        self.num_embedding = nn.Linear(1, D_in)  # Linear layer for numerical values
        
        # Add Layer Normalization
        self.layer_norm = nn.LayerNorm(D_in)  # Layer normalization for the combined embeddings

    def intialize_weights(self):
        for layer in self.linear_regressor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_masks, numerical_values):
        # Convert numerical_values to Float if it's Double
        numerical_values = numerical_values.float()
        
        hidden_states = self.model(input_ids, attention_masks)
        last_hidden_state = hidden_states.last_hidden_state  # [batch_size, input_length, D_in]

        # Embed numerical values
        numerical_values = numerical_values.unsqueeze(-1) if numerical_values.dim() == 2 else numerical_values  # Ensure values have 3 dimensions
        num_embedded = self.num_embedding(numerical_values)


        # Create a mask for numerical tokens
        num_mask = (numerical_values == -1).unsqueeze(-1).float()

        # Ensure dimensions match
        if num_embedded.size(1) != last_hidden_state.size(1):
            num_embedded = num_embedded.expand(-1, last_hidden_state.size(1), -1)
            num_mask = num_mask.expand(-1, last_hidden_state.size(1), -1)

        # Ensure num_mask has the correct shape
        num_mask = num_mask.view(last_hidden_state.size(0), last_hidden_state.size(1),-1) # Reshape to [batch_size, sequence_length,D_in]
        
        # Ensure num_embedded has the same shape as last_hidden_state
        num_embedded = num_embedded.view(last_hidden_state.size(0), last_hidden_state.size(1), -1)  # Adjust shape if necessary

        # Ensure num_embedded has the same last dimension as last_hidden_state
        num_embedded = num_embedded.view(num_embedded.size(0), num_embedded.size(1), -1)  # Adjust as necessary
        
        # Check if num_embedded's last dimension matches last_hidden_state's last dimension
        if num_embedded.size(2) != last_hidden_state.size(2):
            raise ValueError(f"num_embedded last dimension {num_embedded.size(2)} does not match last_hidden_state last dimension {last_hidden_state.size(2)}")
        
        # Combine embeddings
        combined_embeddings = last_hidden_state * (1 - num_mask) + num_embedded * num_mask

        # Apply Layer Normalization
        combined_embeddings = self.layer_norm(combined_embeddings)  # Normalize combined embeddings

        if self.pooling == 'cls':
            input_embedding = combined_embeddings[:, 0, :]  # [batch_size, D_in] -- [CLS] pooling
        elif self.pooling == 'mean':
            input_embedding = combined_embeddings.mean(dim=1)  # [batch_size, D_in] -- mean pooling

        outputs = self.linear_regressor(input_embedding)  # [batch_size, D_out]

        return input_embedding, outputs