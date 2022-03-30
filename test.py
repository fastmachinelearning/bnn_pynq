import os

import numpy as np
import pandas as pd
import torch
import torchvision
from brevitas.export import FINNManager
from torchvision import transforms

from models.CNV import CNV

if __name__ == "__main__":
    net = CNV(
        weight_bit_width=1,
        act_bit_width=1,
        in_bit_width=8,
        num_classes=10,
        in_ch=3,
        cnv_out_ch_stride_pool=[
            (64, 1, False),
            (64, 1, True),
            (128, 1, False),
            (128, 1, True),
            (256, 1, False),
            (256, 1, False),
        ],
        int_fc_feat=[(512, 512)],
        pool_size=2,
        kern_size=3,
    )
    net.load_state_dict(torch.load("state_dict.pth")["models_state_dict"][0])
    net.eval()
    FINNManager.export(net, input_shape=(1, 3, 32, 32), export_path="cnv_finn.onnx")

    data_path = "energyrunner/datasets/ic01"
    df = pd.read_csv(
        os.path.join(data_path, "y_labels.csv"),
        names=["file_name", "num_classes", "label"],
    )
    X_test = np.zeros((len(df), 32, 32, 3))
    y_test = np.zeros((len(df)))
    for i, (file_name, label) in enumerate(zip(df["file_name"], df["label"])):
        with open(os.path.join(data_path, file_name), "rb") as f:
            image_bytes = f.read()
            data = np.frombuffer(image_bytes, np.uint8).reshape(32, 32, 3)
            X_test[i, :, :, :] = data
            y_test[i] = label
    X_test = np.moveaxis(X_test, -1, 1) / 255.0
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)
    with torch.no_grad():
        y = net(X_test)

    from model_def import accuracy_rate

    acc = accuracy_rate(y, y_test)
    print("subset accuracy", acc)

    transform = transforms.Compose([transforms.ToTensor()])
    valset = torchvision.datasets.CIFAR10(
        root="./", train=False, download=True, transform=transform
    )
    valloader = torch.utils.data.DataLoader(valset, batch_size=100)

    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            all_labels.append(labels)
            all_outputs.append(net(images))
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_acc = accuracy_rate(all_outputs, all_labels)
    print("total accuracy", all_acc)
