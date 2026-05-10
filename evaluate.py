import torch
from torch.utils.data import DataLoader

from data import SortingDataset
from model import ReadProcessWriteModel
import config


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SortingDataset(
        num_samples=config.TEST_SAMPLES,
        set_size=config.SET_SIZE
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    model = ReadProcessWriteModel(
        input_dim=1,
        hidden_dim=config.HIDDEN_DIM,
        process_steps=config.PROCESS_STEPS
    ).to(device)

    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.eval()

    total_correct = 0
    total_valid = 0
    total_samples = 0
    valid_order = torch.arange(config.SET_SIZE, device=device)

    with torch.no_grad():
        for x, target in dataloader:
            x = x.to(device)
            target = target.to(device)

            logits = model(x, target=None)
            pred = logits.argmax(dim=-1)

            correct = (pred == target).all(dim=1).sum().item()
            valid = (
                pred.sort(dim=1).values == valid_order.unsqueeze(0)
            ).all(dim=1).sum().item()

            total_correct += correct
            total_valid += valid
            total_samples += x.size(0)

    accuracy = total_correct / total_samples
    valid_rate = total_valid / total_samples

    print(f"Test Exact Match Accuracy: {accuracy:.4f}")
    print(f"Valid Permutation Rate: {valid_rate:.4f}")


if __name__ == "__main__":
    evaluate()
