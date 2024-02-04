import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import requests
import os
from typing import Callable, List, Optional
from torch_geometric.datasets.mnist_superpixels import MNISTSuperpixels
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import pickle


import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset
)


import os
from typing import Callable, List, Optional

import torch

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
    def raw_file_names(self) -> str:
        url_3 = "https://raw.githubusercontent.com/JesusFerFranco/gnn-mnist-classification-turtle/master/gnn_image_classification/archivos.pkl"
        response = requests.get(url_2)
        data_bytess = response.content
        # Deserializar el contenido usando pickle para obtener la lista de datos
        data_listt = pickle.loads(data_bytess)
        return data_listt

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


#def build_mnist_superpixels_dataset() -> TURTLESuperpixels:
 #   return TURTLESuperpixels(
   #     root="gnn_image_classification"
  #  )


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
    dataset: TURTLESuperpixel,
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

    # URL del archivo crudo en GitHub
#    train_url = "https://raw.githubusercontent.com/JesusFerFranco/gnn-mnist-classification-turtle/master/gnn_image_classification/train_data_Turtle.pt"
 #   test_url = "https://raw.githubusercontent.com/JesusFerFranco/gnn-mnist-classification-turtle/master/gnn_image_classification/test_data_Turtle.pt"

    # Descargar el archivo y cargar los datos
  #  train_response = requests.get(train_url)
   # train_dataset = torch.load(train_response.content)

    #test_response = requests.get(test_url)
    #val_dataset = torch.load(test_response.content)

   # train_filename = "train_data_Turtle.pt"
  #  test_filename = "test_data_Turtle.pt"
#OBTENER DATA_LISt
    
# Cargar la lista desde el archivo usando pickle
    url_2 = "https://raw.githubusercontent.com/JesusFerFranco/gnn-mnist-classification-turtle/master/gnn_image_classification/archivos.pkl"
    response = requests.get(url_2)
    data_bytes = response.content
    # Deserializar el contenido usando pickle para obtener la lista de datos
    data_list = pickle.loads(data_bytes)
    # Define la proporción de datos para el conjunto de entrenamiento y el conjunto de prueba
    train_ratio = 0.8
    test_ratio = 0.2

# Calcula el tamaño de cada conjunto de datos
    num_data = len(data_list)
    num_train = int(train_ratio * num_data)
    num_test = num_data - num_train
# Divide el conjunto de datos en conjuntos de entrenamiento y prueba
    train_dataset, val_dataset = random_split(data_list, [num_train, num_test])
#    train_dataset =  torch.load(train_filename)
 #   val_dataset = torch.load(test_filename)
  #  train_dataset=build_mnist_superpixels_dataset()
   # val_dataset=build_mnist_superpixels_dataset()
    
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
