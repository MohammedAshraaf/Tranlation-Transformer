import torch


class CustomLoss(torch.nn.Module):
    def __init__(self, padding_idx):
        super().__init__()
        self.loss_cls = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)

    def forward(self, predicted, actual):
        """
        Calculates the custom loss between the predicted and actual outputs.

        Args:
            predicted: The predicted outputs of the neural network.
            actual: The actual outputs.

        Returns:
            The value of the custom loss.
        """
        loss = self.loss_cls(predicted, actual)

        return loss
