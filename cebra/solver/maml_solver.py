import torch
import copy
from torch.optim import Adam
from cebra import CEBRA
from cebra.solver import Solver  # Import Solver if needed
import numpy as np

def preprocess_data(data, max_length=None, target_num_channels=120):
    """
    Preprocesses the data to ensure that it has a consistent length and number of channels.
    """
    if isinstance(data, np.ndarray):
        data = torch.tensor(data)

    # Padding or truncating the time dimension to max_length
    if max_length is not None:
        if data.shape[0] < max_length:
            padding = max_length - data.shape[0]
            data = torch.cat([data, torch.zeros(padding, data.shape[1])], dim=0)
        else:
            data = data[:max_length, :]

    # Padding or downsampling channels to target_num_channels
    num_channels = data.shape[1]
    if num_channels < target_num_channels:
        padding = target_num_channels - num_channels
        data = torch.cat([data, torch.zeros(data.shape[0], padding)], dim=1)
    elif num_channels > target_num_channels:
        step = num_channels // target_num_channels
        data = data[:, ::step]

    return data

class CustomBatch:
    def __init__(self, data, labels, max_length=None, target_num_channels=120):
        self.reference = preprocess_data(data, max_length, target_num_channels)
        self.positive = torch.tensor(labels)  
        self.negative = torch.zeros_like(self.positive)  

class CustomLoader:
    def __init__(self, data, labels, batch_size, max_length=None, target_num_channels=120):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.max_length = max_length
        self.target_num_channels = target_num_channels

    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            batch_data = self.data[i:i + self.batch_size]
            batch_labels = self.labels[i:i + self.batch_size]
            yield CustomBatch(batch_data, batch_labels, self.max_length, self.target_num_channels)

    def get_indices(self):
        return torch.arange(len(self.data))


# Define the MAMLSolver class which inherits from Solver
class MAMLSolver(Solver):
    def _inference(self, batch):
        """Perform the forward pass using the reference data."""
        return self.model(batch.reference.float())  # Ensure float32 type during inference

    def maml_train(self, datas, labels, maml_steps=5, maml_lr=1e-3, save_frequency=None, logdir="./checkpoints", decode=False):
        """MAML training loop integrated with CEBRA's Solver."""
        
        # Move model to the appropriate device
        self.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.train()  # Set the model in training mode
        
        meta_optimizer = self.optimizer  # Use the outer-loop optimizer

        # Outer loop (MAML) iteration
        for epoch in range(1, maml_steps + 1):
            print(f"Epoch {epoch}: MAML Training")
            meta_optimizer.zero_grad()  # Reset outer loop optimizer
            
            meta_loss = 0.0  # Initialize meta-loss for this epoch

            # Loop over tasks (each task has its own data and labels)
            for task_data, task_labels in zip(datas, labels):
                # Create a task-specific data loader
                task_loader = CustomLoader(task_data, task_labels, batch_size=len(task_data))

                # Create a copy of the model for the inner loop
                model_copy = copy.deepcopy(self.model)
                inner_optimizer = torch.optim.SGD(model_copy.parameters(), lr=maml_lr)

                # Inner loop: perform task-specific updates
                for batch in task_loader:
                    stats = self.step(batch)  # Call step function for gradient update
                    loss = stats['total']
                    inner_optimizer.zero_grad()
                    loss.backward()  # Backpropagate the loss for this task
                    inner_optimizer.step()

                # Compute the meta-loss for this task
                task_meta_loss = self._meta_loss(task_loader, model_copy)
                meta_loss += task_meta_loss  # Add task meta-loss to the total meta-loss

            # Average meta-loss across all tasks (for the outer loop)
            meta_loss /= len(datas)  # Average loss over tasks
            meta_loss.backward()  # Backpropagate the meta-loss
            meta_optimizer.step()  # Update the model based on the meta-loss

            print(f"Meta Loss: {meta_loss.item():.6f}")

            # Optionally, save the model at regular intervals
            if save_frequency is not None and epoch % save_frequency == 0:
                self.save(logdir, f"checkpoint_{epoch:#07d}.pth")

            # Perform decoding or additional saving
            if decode:
                self.decode_history.append(self.decoding(datas, labels))

        # Optionally save the model after training
        self.save(logdir, f"checkpoint_final.pth")
        
    def step(self, batch):
        """Perform a single gradient update on the model"""
        self.optimizer.zero_grad()
        prediction = self._inference(batch)  # This uses batch.reference, batch.positive, etc.
        loss, align, uniform = self.criterion(prediction.reference,
                                              prediction.positive,
                                              prediction.negative)
        loss.backward()
        self.optimizer.step()

        stats = dict(
            pos=align.item(),
            neg=uniform.item(),
            total=loss.item(),
            temperature=self.criterion.temperature,
        )
        return stats

    def _meta_loss(self, batch, model):
        """Compute the meta-loss for the updated model"""
        model.eval()
        with torch.no_grad():
            prediction = model(batch.reference)  # Forward pass
            loss, _, _ = self.criterion(
                prediction,
                batch.positive,
                batch.negative,
            )
        return loss

