import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import SortingDataset
from model import ReadProcessWriteModel
import config


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SortingDataset(
        num_samples=config.TRAIN_SAMPLES,
        set_size=config.SET_SIZE
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    model = ReadProcessWriteModel(
        input_dim=1,
        hidden_dim=config.HIDDEN_DIM,
        process_steps=config.PROCESS_STEPS
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )

    for epoch in range(config.NUM_EPOCHS):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, target in dataloader:
            x = x.to(device)
            target = target.to(device)

            logits = model(x, target=target)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

            model.eval()
            with torch.no_grad():
                greedy_logits = model(x, target=None)
                pred = greedy_logits.argmax(dim=-1)
                correct = (pred == target).all(dim=1).sum().item()
            model.train()

            total_correct += correct
            total_samples += x.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss: {avg_loss:.4f} | "
            f"Exact Match Acc: {accuracy:.4f}"
        )

    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    train()
