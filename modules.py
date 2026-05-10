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
        # Read 阶段逐元素编码，不依赖输入元素的顺序。
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

        # h/c 是 process LSTM 的状态，r 是当前 attention read 得到的集合摘要。
        h = torch.zeros(batch_size, hidden_dim, device=memory.device)
        c = torch.zeros(batch_size, hidden_dim, device=memory.device)
        r = torch.zeros(batch_size, hidden_dim, device=memory.device)

        q_star = torch.cat([h, r], dim=-1)

        for _ in range(self.num_steps):
            h, c = self.lstm_cell(q_star, (h, c))

            # Process attention：在解码前多次读取整个集合，用于构建全局上下文。
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
        self.glimpse_refine = nn.Linear(hidden_dim * 2, hidden_dim)

        self.start_token = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, memory, context, target=None, mask_selected=True):
        batch_size, set_size, hidden_dim = memory.shape

        # context 来自 ProcessModule，用于初始化写阶段 decoder 的 LSTM 状态。
        h = torch.tanh(self.init_h(context))
        c = torch.tanh(self.init_c(context))

        decoder_input = self.start_token.unsqueeze(0).expand(batch_size, -1)
        # selected_mask 记录已经输出过的位置；推理时避免重复选择同一个元素。
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

            # Glimpse attention：在每个解码步先读取一次 memory，细化当前 decoder query。
            glimpse_scores = torch.bmm(memory, h.unsqueeze(-1)).squeeze(-1)
            if mask_selected:
                glimpse_scores = glimpse_scores.masked_fill(
                    selected_mask,
                    float("-inf")
                )
            glimpse_weights = F.softmax(glimpse_scores, dim=-1)
            glimpse = torch.bmm(
                glimpse_weights.unsqueeze(1),
                memory
            ).squeeze(1)
            pointer_query = torch.tanh(self.glimpse_refine(
                torch.cat([h, glimpse], dim=-1)
            ))

            # Pointer attention：最终输出每个输入位置的 logit，用交叉熵监督目标下标。
            pointer_scores = torch.bmm(
                memory,
                pointer_query.unsqueeze(-1)
            ).squeeze(-1)
            if mask_selected:
                pointer_scores = pointer_scores.masked_fill(
                    selected_mask,
                    float("-inf")
                )
            logits_list.append(pointer_scores)

            if target is not None:
                # 训练时使用 teacher forcing：下一步输入使用真实目标位置的 memory。
                next_index = target[:, t]
            else:
                # 推理时使用当前 pointer 分数最高的位置作为下一步输入。
                next_index = pointer_scores.argmax(dim=-1)

            next_mask = torch.zeros_like(selected_mask)
            next_mask[batch_indices, next_index] = True
            selected_mask = selected_mask | next_mask
            decoder_input = memory[batch_indices, next_index]

        logits = torch.stack(logits_list, dim=1)

        return logits
