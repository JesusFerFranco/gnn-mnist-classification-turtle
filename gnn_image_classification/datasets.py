import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import requests
import os
from typing import Callable, List, Optional
from torch_geometric.datasets.mnist_superpixels import MNISTSuperpixels
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import io


from torch_geometric.data import (
    Data,
    InMemoryDataset
)

class TURTLESuperpixels(InMemoryDataset):
    def __init__(
        self, root
    ) -> None:
        super().__init__(root)

    @property
    def raw_file_names(self) -> list:
        url_3 = "https://raw.githubusercontent.com/JesusFerFranco/gnn-mnist-classification-turtle/master/gnn_image_classification/TURTLESUPERPIXEL.pt"
        response = requests.get(url_3)
        data_bytes = response.content
        # Cargar la lista de grafos directamente desde los bytes
        data_list = torch.load(io.BytesIO(data_bytes), map_location=torch.device('cpu'))
        return data_list


    @property
    def processed_file_names(self) -> List[str]:
        return ['train_data_Turtle.pt', 'test_data_Turtle.pt']

    def download(self):
      pass

    def process(self) -> None:
        inputs = torch.load(self.raw_paths[0])
        for i in range(len(inputs)):
            data_list = [Data(**data_dict) for data_dict in inputs[i]]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.save(data_list, self.processed_paths[i])


def build_mnist_superpixels_dataset() -> TURTLESuperpixels:
    return TURTLESuperpixels(
        root="gnn_image_classification"
    )


def build_collate_fn(device: str | torch.device):
    def collate_fn(original_batch: list[Data]):
        batch_node_features: list[torch.Tensor] = []
        batch_edge_indices: list[torch.Tensor] = []
        classes: list[int] = []
        for data in original_batch:
           # node_features = torch.cat((data.x, data.pos), dim=-1).to(device)
           # edge_indices = data.edge_index.to(device)
           # class_ = int(data.y)

            #batch_node_features.append(node_features)
            #batch_edge_indices.append(edge_indices)
            #classes.append(class_)
            node_features = torch.cat((data.x, data.pos), dim=-1).to(device)
            edge_indices = data.edge_index.to(device)
            class_ = data.y.to(device)

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

#SE INTENTARÁ TOMAR LA LISTA GLOBAL DATA_LIST OBTENIDA DEL TURTLESUPERPIXEL

#OBTENER DATA_LISt    
# Cargar la lista desde el archivo uun url
url_1 = "https://raw.githubusercontent.com/JesusFerFranco/gnn-mnist-classification-turtle/master/gnn_image_classification/TURTLESUPERPIXEL.pt"
response = requests.get(url_1)
data_bytess = response.content
    
# Deserializar el contenido obtener la lista de datos
data_listt = torch.load(io.BytesIO(data_bytess))

def build_dataloader(
    dataset: data_listt,
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

#OBTENER DATA_LISt    
# Cargar la lista desde el archivo usando pickle
    url_2 = "https://raw.githubusercontent.com/JesusFerFranco/gnn-mnist-classification-turtle/master/gnn_image_classification/test_data_Turtle.pt"
    response = requests.get(url_2)
    data_bytes = response.content
    val_dataset = torch.load(io.BytesIO(data_bytes))

    url_3 = "https://raw.githubusercontent.com/JesusFerFranco/gnn-mnist-classification-turtle/master/gnn_image_classification/train_data_Turtle.pt"
    response1 = requests.get(url_3)
    data_bytes1 = response1.content
    train_dataset = torch.load(io.BytesIO(data_bytes1))

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
