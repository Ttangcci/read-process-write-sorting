import argparse
import csv
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from baseline import PointerNetwork
from data import SortingDataset
from model import ReadProcessWriteModel


def parse_int_list(value):
    # 命令行参数形如 "5,10,15"，这里转成整数列表。
    return [int(item) for item in value.split(",") if item.strip()]


def make_loader(num_samples, set_size, batch_size, shuffle):
    dataset = SortingDataset(num_samples=num_samples, set_size=set_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_batch(model, x, target, optimizer, device):
    model.train()
    x = x.to(device)
    target = target.to(device)

    # 两类模型都实现了相同接口，因此训练逻辑可以复用。
    logits = model(x, target=target)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_model(model, dataloader, device, set_size):
    model.eval()
    total_correct = 0
    total_valid = 0
    total_samples = 0
    valid_order = torch.arange(set_size, device=device)

    with torch.no_grad():
        for x, target in dataloader:
            x = x.to(device)
            target = target.to(device)

            logits = model(x, target=None)
            pred = logits.argmax(dim=-1)

            correct = (pred == target).all(dim=1).sum().item()
            # valid_permutation 用于发现重复选择或漏选输入位置的问题。
            valid = (
                pred.sort(dim=1).values == valid_order.unsqueeze(0)
            ).all(dim=1).sum().item()

            total_correct += correct
            total_valid += valid
            total_samples += x.size(0)

    return {
        "exact_match": total_correct / total_samples,
        "valid_permutation": total_valid / total_samples,
    }


def train_model(model, train_loader, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    iterator = iter(train_loader)
    total_loss = 0.0

    for _ in range(args.train_steps):
        try:
            x, target = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, target = next(iterator)

        total_loss += train_one_batch(model, x, target, optimizer, device)

    return total_loss / args.train_steps


def write_result(path, fieldnames, row, write_header):
    with open(path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def make_result_row(
    model_name,
    train_set_size,
    test_set_size,
    process_steps,
    glimpses,
    train_loss,
    train_seconds,
    metrics,
    args
):
    row = {
        "model": model_name,
        "train_set_size": train_set_size,
        "test_set_size": test_set_size,
        "process_steps": process_steps,
        "glimpses": glimpses,
        "train_steps": args.train_steps,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "hidden_dim": args.hidden_dim,
        "train_loss": train_loss,
        "train_seconds": train_seconds,
    }
    row.update(metrics)
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Compare RPW and Pointer Network on out-of-sample sorting."
    )
    parser.add_argument("--train-set-size", type=int, default=config.SET_SIZE)
    parser.add_argument("--test-set-sizes", default="5,10,15")
    parser.add_argument("--process-steps", default="0,1,5,10")
    parser.add_argument("--glimpses", default="0,1")
    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument("--train-samples", type=int, default=config.TRAIN_SAMPLES)
    parser.add_argument("--test-samples", type=int, default=config.TEST_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--hidden-dim", type=int, default=config.HIDDEN_DIM)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--output", default="experiment_results.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set_sizes = parse_int_list(args.test_set_sizes)
    process_steps_list = parse_int_list(args.process_steps)
    glimpses_list = parse_int_list(args.glimpses)

    fieldnames = [
        "model",
        "train_set_size",
        "test_set_size",
        "process_steps",
        "glimpses",
        "train_steps",
        "train_samples",
        "test_samples",
        "hidden_dim",
        "train_loss",
        "exact_match",
        "valid_permutation",
        "train_seconds",
    ]

    with open(args.output, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    train_loader = make_loader(
        args.train_samples,
        args.train_set_size,
        args.batch_size,
        shuffle=True
    )
    test_loaders = {
        test_set_size: make_loader(
            args.test_samples,
            test_set_size,
            args.batch_size,
            shuffle=False
        )
        for test_set_size in test_set_sizes
    }

    for glimpses in glimpses_list:
        # Ptr-Net 不包含 process step，但也可以比较 glimpse=0/1。
        baseline = PointerNetwork(
            input_dim=1,
            hidden_dim=args.hidden_dim,
            glimpses=glimpses
        ).to(device)
        start_time = time.time()
        train_loss = train_model(baseline, train_loader, args, device)
        train_seconds = time.time() - start_time

        for test_set_size, test_loader in test_loaders.items():
            metrics = evaluate_model(
                baseline,
                test_loader,
                device,
                test_set_size
            )
            row = make_result_row(
                "PointerNetwork",
                args.train_set_size,
                test_set_size,
                "NA",
                glimpses,
                train_loss,
                train_seconds,
                metrics,
                args
            )
            write_result(args.output, fieldnames, row, write_header=False)
            print(row)

        for process_steps in process_steps_list:
            # RPW 对同一个训练长度测试不同 process step 和不同测试长度。
            rpw = ReadProcessWriteModel(
                input_dim=1,
                hidden_dim=args.hidden_dim,
                process_steps=process_steps,
                glimpses=glimpses
            ).to(device)
            start_time = time.time()
            train_loss = train_model(rpw, train_loader, args, device)
            train_seconds = time.time() - start_time

            for test_set_size, test_loader in test_loaders.items():
                metrics = evaluate_model(
                    rpw,
                    test_loader,
                    device,
                    test_set_size
                )
                row = make_result_row(
                    "ReadProcessWrite",
                    args.train_set_size,
                    test_set_size,
                    process_steps,
                    glimpses,
                    train_loss,
                    train_seconds,
                    metrics,
                    args
                )
                write_result(args.output, fieldnames, row, write_header=False)
                print(row)


if __name__ == "__main__":
    main()
