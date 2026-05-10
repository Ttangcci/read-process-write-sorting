import torch.nn as nn
from modules import ReadModule, ProcessModule, WriteModule


class ReadProcessWriteModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, process_steps=3):
        super().__init__()

        self.read = ReadModule(input_dim=input_dim, hidden_dim=hidden_dim)
        self.process = ProcessModule(
            hidden_dim=hidden_dim,
            num_steps=process_steps
        )
        self.write = WriteModule(hidden_dim=hidden_dim)

    def forward(self, x, target=None):
        memory = self.read(x)
        context = self.process(memory)
        logits = self.write(memory, context, target=target)
        return logits