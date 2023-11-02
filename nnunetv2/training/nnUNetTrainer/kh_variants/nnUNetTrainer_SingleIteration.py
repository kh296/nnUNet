import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_SingleIteration(nnUNetTrainer):
    """
    Trainer initialised to run for a single iteration.

    This can be useful for testing.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_iterations_per_epoch = 1
        self.num_val_iterations_per_epoch = 1
        self.num_epochs = 1
