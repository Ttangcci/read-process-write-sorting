import torch
import torch.nn as nn


class PointerNetwork(nn.Module):
    """基线模型：LSTM encoder-decoder 加 pointer attention。"""

    def __init__(self, input_dim=1, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.decoder_cell = nn.LSTMCell(
            input_size=hidden_dim,
            hidden_size=hidden_dim
        )
        self.start_token = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x, target=None, mask_selected=True):
        # encoder 按输入顺序读取序列；这里作为和 RPW 对比的传统 Ptr-Net baseline。
        memory, (h_n, c_n) = self.encoder(x)
        batch_size, set_size, _ = memory.shape

        h = h_n[-1]
        c = c_n[-1]
        decoder_input = self.start_token.unsqueeze(0).expand(batch_size, -1)

        selected_mask = torch.zeros(
            batch_size,
            set_size,
            dtype=torch.bool,
            device=x.device
        )
        batch_indices = torch.arange(batch_size, device=x.device)

        logits_list = []

        for t in range(set_size):
            h, c = self.decoder_cell(decoder_input, (h, c))

            # Pointer attention：用 decoder hidden state 直接给每个输入位置打分。
            pointer_scores = torch.bmm(memory, h.unsqueeze(-1)).squeeze(-1)
            if mask_selected:
                pointer_scores = pointer_scores.masked_fill(
                    selected_mask,
                    float("-inf")
                )
            logits_list.append(pointer_scores)

            if target is not None:
                # 训练时 teacher forcing，保持和 RPW 的训练接口一致。
                next_index = target[:, t]
            else:
                next_index = pointer_scores.argmax(dim=-1)

            next_mask = torch.zeros_like(selected_mask)
            next_mask[batch_indices, next_index] = True
            selected_mask = selected_mask | next_mask
            decoder_input = memory[batch_indices, next_index]

        return torch.stack(logits_list, dim=1)
