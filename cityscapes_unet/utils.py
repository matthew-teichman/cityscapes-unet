import torch
import torch.nn as nn
from typing import Tuple


def get_batch_size(
    model: nn.Module,
    device: torch.device,
    input_shape: Tuple[int, int, int],
    output_shape: Tuple[int],
    dataset_size: int,
    max_batch_size: int = None,
    num_iterations: int = 5,
) -> int:
    model.to(device)
    model.train(True)
    optimizer = torch.optim.Adam(model.parameters())

    print("Test batch size")
    batch_size = 2
    while True:
        if max_batch_size is not None and batch_size >= max_batch_size:
            batch_size = max_batch_size
            break
        if batch_size >= dataset_size:
            batch_size = batch_size // 2
            break
        try:
            for _ in range(num_iterations):
                # dummy inputs and targets
                inputs = torch.rand(*(batch_size, *input_shape), device=device)
                targets = torch.rand(*(batch_size, *output_shape), device=device)
                outputs = model(inputs)
                loss = F.mse_loss(targets, outputs)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            batch_size *= 2
            print(f"\tTesting batch size {batch_size}")
            sleep(3)
        except RuntimeError:
            print(f"\tOOM at batch size {batch_size}")
            batch_size //= 2
            break
    del model, optimizer
    torch.cuda.empty_cache()
    print(f"Final batch size {batch_size}")
    return batch_size
