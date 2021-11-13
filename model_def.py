"""
CNN on Cifar10 from Keras example:
https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
"""

import tempfile
from typing import Any, Dict, Sequence, Tuple, Union, cast

import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.nn import functional as F

from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext
from determined import InvalidHP

from models.CNV import CNV
from models.losses import SqrHingeLoss
from models.bops_counter import calc_BOPS

#from brevitas.export.onnx.generic.manager import BrevitasONNXManager
#from finn.util.inference_cost import inference_cost

# Constants about the data set.
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
IN_BIT_WIDTH = 8

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


def accuracy_rate(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Return the accuracy rate based on dense predictions and sparse labels."""
    assert len(predictions) == len(labels), "Predictions and labels must have the same length."
    assert len(labels.shape) == 1, "Labels must be a column vector."

    return (  # type: ignore
        float((predictions.argmax(1) == labels.to(torch.long)).sum()) / predictions.shape[0]
    )

class CIFARTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context

        # Create a unique download directory for each rank so they don't overwrite each
        # other when doing distributed training.
        self.download_directory = tempfile.mkdtemp()

        try: 
            net = CNV(weight_bit_width=self.context.get_hparam("weight_bit_width"),
                      act_bit_width=self.context.get_hparam("act_bit_width"),
                      in_bit_width=IN_BIT_WIDTH,
                      num_classes=NUM_CLASSES,
                      in_ch=NUM_CHANNELS,
                      cnv_out_ch_stride_pool=[(self.context.get_hparam("cnv_out_ch_0"), 
                                               self.context.get_hparam("cnv_stride_0"),
                                               self.context.get_hparam("cnv_pool_0")),
                                              (self.context.get_hparam("cnv_out_ch_1"), 
                                               self.context.get_hparam("cnv_stride_1"), 
                                               self.context.get_hparam("cnv_pool_1")),
                                              (self.context.get_hparam("cnv_out_ch_2"),
                                               self.context.get_hparam("cnv_stride_2"),
                                               self.context.get_hparam("cnv_pool_2")),
                                              (self.context.get_hparam("cnv_out_ch_3"),
                                               self.context.get_hparam("cnv_stride_3"),  
                                               self.context.get_hparam("cnv_pool_3")),
                                              (self.context.get_hparam("cnv_out_ch_4"), 
                                               self.context.get_hparam("cnv_stride_4"), 
                                               self.context.get_hparam("cnv_pool_4")),
                                              (self.context.get_hparam("cnv_out_ch_5"),
                                               self.context.get_hparam("cnv_stride_5"), 
                                               self.context.get_hparam("cnv_pool_5"))],
                      int_fc_feat=[(self.context.get_hparam("int_fc_feat_1"), self.context.get_hparam("int_fc_feat_2"))],
                      pool_size=self.context.get_hparam("pool_size"),
                      kern_size=self.context.get_hparam("kern_size"))
        except: 
            raise InvalidHP

        self.model = self.context.wrap_model(net)
        
        print("In __init__()")
        print(self.model)

        self.optimizer = self.context.wrap_optimizer(torch.optim.Adam(
            self.model.parameters(),
            lr=self.context.get_hparam("learning_rate"),
            weight_decay=self.context.get_hparam("learning_rate_decay"),
        ))

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
        print("In train_batch()")
        bops = calc_BOPS(self.model)

        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)
        
        #export_onnx_path = "models/model_export.onnx"
        #final_onnx_path = "models/model_final.onnx"
        #cost_dict_path = "models/model_cost.json"
        
        #BrevitasONNXManager.export(self.model.cpu(), input_t=torch.randn(1, 2, 1024), export_path=export_onnx_path);
        #inference_cost(export_onnx_path, output_json=cost_dict_path, output_onnx=final_onnx_path,
               #preprocess=True, discount_sparsity=True)
        #inference_cost_dict = json.loads(inference_cost)
        return {"BOPs": bops, "loss": loss, "train_error": 1.0 - accuracy, "train_accuracy": accuracy}

    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        """
        Calculate validation metrics for a batch and return them as a dictionary.
        This method is not necessary if the user defines evaluate_full_dataset().
        """
        batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
        data, labels = batch

        output = self.model(data)
        accuracy = accuracy_rate(output, labels)
        validation_result = {"validation_accuracy": accuracy, "validation_error": 1.0 - accuracy}
        validation_result.update(self.model_cost)
        return validation_result

    def build_training_data_loader(self) -> Any:
        train_transforms_list = [transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor()]
        transform = transforms.Compose(train_transforms_list)
        trainset = torchvision.datasets.CIFAR10(
            root=self.download_directory, train=True, download=True, transform=transform
        )
        return DataLoader(trainset, batch_size=self.context.get_per_slot_batch_size())

    def build_validation_data_loader(self) -> Any:
        transform = transforms.Compose([transforms.ToTensor()])
        valset = torchvision.datasets.CIFAR10(
            root=self.download_directory, train=False, download=True, transform=transform
        )

        return DataLoader(valset, batch_size=self.context.get_per_slot_batch_size())
