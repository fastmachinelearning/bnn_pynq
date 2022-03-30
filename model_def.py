"""
CNN on Cifar10 from Keras example:
https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
"""

import tempfile
from typing import Any, Dict, Sequence, Tuple, Union, cast

import torch
import torchvision
from determined import InvalidHP
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext
from torch.nn import functional as F
from torchvision import transforms

from models.bops_counter import calc_bops
from models.CNV import CNV
from models.losses import SqrHingeLoss
from models.bops_counter import calc_bops

from brevitas.export.onnx.generic.manager import BrevitasONNXManager
from finn.util.inference_cost import inference_cost

# Constants about the data set.
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
IN_BIT_WIDTH = 8

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

def apply_constraints(hparams, num_params):
    normal_skip_count = 0
    reduce_skip_count = 0
    normal_conv_count = 0
    for hp, val in hparams.items():
        if val == "skip_connect":
            if "normal" in hp:
                normal_skip_count += 1
            elif "reduce" in hp:
                reduce_skip_count += 1
        if val == "sep_conv_3x3":
            if "normal" in hp:
                normal_conv_count += 1

    # Reject if num skip_connect >= 3 or <1 in either normal or reduce cell.
    if normal_skip_count >= 3 or reduce_skip_count >= 3:
        print("First invalid execute")
        raise det.InvalidHP("too many skip_connect operations")
    if normal_skip_count == 0 or reduce_skip_count == 0:
        print("Second invalid execute")
        raise det.InvalidHP("too few skip_connect operations")
    # Reject if fewer than 3 sep_conv_3x3 in normal cell.
    if normal_conv_count < 3:
        print("Third invalid")
        raise det.InvalidHP("fewer than 3 sep_conv_3x3 operations in normal cell")
    # Reject if num_params > 4.5 million or < 2.5 million.
    if num_params < 2.5e6 or num_params > 4.5e6:
        print("Fourth invalid")
        raise det.InvalidHP(
            "number of parameters in architecture is not between 2.5 and 4.5 million"
        )

def accuracy_rate(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Return the accuracy rate based on dense predictions and sparse labels."""
    assert len(predictions) == len(
        labels
    ), "Predictions and labels must have the same length."
    assert len(labels.shape) == 1, "Labels must be a column vector."

    return (  # type: ignore
        float((predictions.argmax(1) == labels.to(torch.long)).sum())
        / predictions.shape[0]
    )


class CIFARTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context

        # Create a unique download directory for each rank so they don't overwrite each
        # other when doing distributed training.
        self.download_directory = tempfile.mkdtemp()

        # unwrap the model
        try:
            net = CNV(
                weight_bit_width=self.context.get_hparam("weight_bit_width"),
                act_bit_width=self.context.get_hparam("act_bit_width"),
                in_bit_width=IN_BIT_WIDTH,
                num_classes=NUM_CLASSES,
                in_ch=NUM_CHANNELS,
                cnv_out_ch_stride_pool=[
                    (
                        self.context.get_hparam("cnv_out_ch_0"),
                        self.context.get_hparam("cnv_stride_0"),
                        self.context.get_hparam("cnv_pool_0"),
                    ),
                    (
                        self.context.get_hparam("cnv_out_ch_1"),
                        self.context.get_hparam("cnv_stride_1"),
                        self.context.get_hparam("cnv_pool_1"),
                    ),
                    (
                        self.context.get_hparam("cnv_out_ch_2"),
                        self.context.get_hparam("cnv_stride_2"),
                        self.context.get_hparam("cnv_pool_2"),
                    ),
                    (
                        self.context.get_hparam("cnv_out_ch_3"),
                        self.context.get_hparam("cnv_stride_3"),
                        self.context.get_hparam("cnv_pool_3"),
                    ),
                    (
                        self.context.get_hparam("cnv_out_ch_4"),
                        self.context.get_hparam("cnv_stride_4"),
                        self.context.get_hparam("cnv_pool_4"),
                    ),
                    (
                        self.context.get_hparam("cnv_out_ch_5"),
                        self.context.get_hparam("cnv_stride_5"),
                        self.context.get_hparam("cnv_pool_5"),
                    ),
                ],
                int_fc_feat=[
                    (
                        self.context.get_hparam("int_fc_feat_1"),
                        self.context.get_hparam("int_fc_feat_2"),
                    )
                ],
                pool_size=self.context.get_hparam("pool_size"),
                kern_size=self.context.get_hparam("kern_size"),
            )
        except Exception as e:
            print("Received exception in model creation. Skipping.")
            print("Exception is of type: ", type(e))
            print("And reads: ", e)
            raise InvalidHP
        
        if "use_constraints" in self.hparams and self.hparams["use_constraints"]:
            apply_constraints(self.hparams, size)

        self.model_cost = net.calculate_model_cost()
        self.model = self.context.wrap_model(net)

        self.optimizer = self.context.wrap_optimizer(
            torch.optim.Adam(
                self.model.parameters(),
                lr=self.context.get_hparam("learning_rate"),
                weight_decay=self.context.get_hparam("learning_rate_decay"),
            )
        )

        self.criterion = SqrHingeLoss()

    def train_batch(
        self, batch: TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
        data, labels = batch

        output = self.model(data)

        labels_onehot = F.one_hot(labels, num_classes=NUM_CLASSES)
        labels_onehot.masked_fill_(labels_onehot == 0, -1)
        loss = self.criterion(output, labels_onehot)

        accuracy = accuracy_rate(output, labels)

        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        return {"loss": loss, "train_error": 1.0 - accuracy, "train_accuracy": accuracy}

    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        """
        Calculate validation metrics for a batch and return them as a dictionary.
        This method is not necessary if the user defines evaluate_full_dataset().
        """
        batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
        data, labels = batch

        output = self.model(data)

        labels_onehot = F.one_hot(labels, num_classes=NUM_CLASSES)
        labels_onehot.masked_fill_(labels_onehot == 0, -1)
        loss = self.criterion(output, labels_onehot)

        bops = calc_bops(self.model)
        accuracy = accuracy_rate(output, labels)
        validation_result = {
            "bops": bops,
            "validation_loss": loss,
            "validation_accuracy": accuracy,
            "validation_error": 1.0 - accuracy,
        }
        validation_result.update(self.model_cost)
        return validation_result

    def build_training_data_loader(self) -> Any:
        train_transforms_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        transform = transforms.Compose(train_transforms_list)
        trainset = torchvision.datasets.CIFAR10(
            root=self.download_directory, train=True, download=True, transform=transform
        )
        return DataLoader(trainset, batch_size=self.context.get_per_slot_batch_size())

    def build_validation_data_loader(self) -> Any:
        transform = transforms.Compose([transforms.ToTensor()])
        valset = torchvision.datasets.CIFAR10(
            root=self.download_directory,
            train=False,
            download=True,
            transform=transform,
        )

        return DataLoader(valset, batch_size=self.context.get_per_slot_batch_size())

    # rename this to build_validation_data_loader to use subset as valdiation
    def build_partial_validation_data_loader(self) -> Any:
        import os

        import numpy as np
        import pandas as pd

        # load partial dataset from energyrunner
        data_path = "energyrunner/datasets/ic01"
        df = pd.read_csv(
            os.path.join(data_path, "y_labels.csv"),
            names=["file_name", "num_classes", "label"],
        )
        X_test = np.zeros((len(df), IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        y_test = np.zeros((len(df)))
        for i, (file_name, label) in enumerate(zip(df["file_name"], df["label"])):
            with open(os.path.join(data_path, file_name), "rb") as f:
                image_bytes = f.read()
                data = np.frombuffer(image_bytes, np.uint8).reshape(
                    IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS
                )
                X_test[i, :, :, :] = data
                y_test[i] = label
        X_test = np.moveaxis(X_test, -1, 1) / 255.0
        X_test = torch.Tensor(X_test)
        y_test = torch.Tensor(y_test)

        # create dataset and dataloaders
        valset = torch.utils.data.TensorDataset(X_test, y_test)

        return DataLoader(valset, batch_size=self.context.get_per_slot_batch_size())
