import torch
import torch.nn as nn
import torch.nn.functional as F


class ReadModule(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class ProcessModule(nn.Module):
    def __init__(self, hidden_dim=128, num_steps=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps

        self.lstm_cell = nn.LSTMCell(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim
        )

    def forward(self, memory):
        batch_size, set_size, hidden_dim = memory.shape

        h = torch.zeros(batch_size, hidden_dim, device=memory.device)
        c = torch.zeros(batch_size, hidden_dim, device=memory.device)
        r = torch.zeros(batch_size, hidden_dim, device=memory.device)

        q_star = torch.cat([h, r], dim=-1)

        for _ in range(self.num_steps):
            h, c = self.lstm_cell(q_star, (h, c))

            scores = torch.bmm(memory, h.unsqueeze(-1)).squeeze(-1)
            attn_weights = F.softmax(scores, dim=-1)

            r = torch.bmm(attn_weights.unsqueeze(1), memory).squeeze(1)

            q_star = torch.cat([h, r], dim=-1)

        return q_star


class WriteModule(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.decoder_cell = nn.LSTMCell(
            input_size=hidden_dim,
            hidden_size=hidden_dim
        )

        self.init_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.init_c = nn.Linear(hidden_dim * 2, hidden_dim)

        self.start_token = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, memory, context, target=None):
        batch_size, set_size, hidden_dim = memory.shape

        h = torch.tanh(self.init_h(context))
        c = torch.tanh(self.init_c(context))

        decoder_input = self.start_token.unsqueeze(0).expand(batch_size, -1)
        selected_mask = torch.zeros(
            batch_size,
            set_size,
            dtype=torch.bool,
            device=memory.device
        )
        batch_indices = torch.arange(batch_size, device=memory.device)

        logits_list = []

        for t in range(set_size):
            h, c = self.decoder_cell(decoder_input, (h, c))

            scores = torch.bmm(memory, h.unsqueeze(-1)).squeeze(-1)
            scores = scores.masked_fill(selected_mask, float("-inf"))
            logits_list.append(scores)

            if target is not None:
                next_index = target[:, t]
            else:
                next_index = scores.argmax(dim=-1)

            next_mask = torch.zeros_like(selected_mask)
            next_mask[batch_indices, next_index] = True
            selected_mask = selected_mask | next_mask
            decoder_input = memory[batch_indices, next_index]

        logits = torch.stack(logits_list, dim=1)

        return logits
