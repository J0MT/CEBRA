import torch
import copy
from torch.optim import Adam
from cebra import CEBRA
from cebra.solver import Solver  # Import Solver if needed

class CustomBatch:
    def __init__(self, data, labels):
        # Wrap data and labels as tensors with attributes
        self.reference = torch.tensor(data)  # Data used for training
        self.positive = torch.tensor(labels)  # Labels used for training
        self.negative = torch.zeros_like(self.positive)  # Placeholder for negative samples


# Define the MAMLSolver class which inherits from Solver
class MAMLSolver(Solver):
    def _inference(self, batch):
        """Perform the forward pass using the reference data."""
        # Assuming the model is set up to take batch.reference as input
        return self.model(batch.reference)  # Perform inference using reference data from the batch

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


class CustomLoader:
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.index = torch.arange(len(data))

    def __iter__(self):
        # Yield a CustomBatch object for each batch
        for i in range(0, len(self.data), self.batch_size):
            batch_data = self.data[i:i + self.batch_size]
            batch_labels = self.labels[i:i + self.batch_size]
            yield CustomBatch(batch_data, batch_labels)  # Yield CustomBatch object

    def get_indices(self):
        return self.index


