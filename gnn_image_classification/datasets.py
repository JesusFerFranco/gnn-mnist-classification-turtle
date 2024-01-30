import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

import os
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset
)




#def build_mnist_superpixels_dataset(train: bool) -> TURTLESuperpixels:
 #   return TURTLESuperpixels(
  #      root="turtles-superpixels-dataset",
   #     train=train,
    #)


def build_collate_fn(device: str | torch.device):
    def collate_fn(original_batch: list[Data]):
        batch_node_features: list[torch.Tensor] = []
        batch_edge_indices: list[torch.Tensor] = []
        classes: list[int] = []

        for data in original_batch:
            node_features = torch.cat((data.x, data.pos), dim=-1).to(device)
            edge_indices = data.edge_index.to(device)
            class_ = int(data.y)

            batch_node_features.append(node_features)
            batch_edge_indices.append(edge_indices)
            classes.append(class_)

        collated = {
            "batch_node_features": batch_node_features,
            "batch_edge_indices": batch_edge_indices,
            "classes": torch.LongTensor(classes).to(device),
        }

        return collated

    return collate_fn


def build_dataloader(
    dataset: TURTLESuperpixels,
    batch_size: int,
    shuffle: bool,
    device: str | torch.device,
) -> DataLoader:
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=build_collate_fn(device=device),
    )

    return loader


def build_train_val_dataloaders(batch_size: int, device: str) -> tuple[DataLoader, DataLoader]:
  
    train_filename = "train_data_Turtle.pt"
    test_filename = "test_data_Turtle.pt"
    
    train_dataset =  torch.load(train_filename)
    val_dataset = torch.load(test_filename)

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        device=device,
    )

    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        device=device,
    )

    return train_loader, val_loader
