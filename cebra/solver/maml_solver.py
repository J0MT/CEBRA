import torch
import copy
import cebra
from torch.optim import SGD
from cebra import CEBRA
from torch import optim

class MAMLSolver:
    """Custom MAML Solver for CEBRA"""
    
    def __init__(self, model, criterion, optimizer, *args, **kwargs):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        # Initialize other attributes if needed

    def maml_train(self, datas, labels, maml_steps=5, maml_lr=1e-3, save_frequency=None, logdir="./checkpoints", decode=False):
        """MAML Training loop integrated with CEBRA"""
        
        # Move model to the appropriate device (CPU or GPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)  # Ensure model is on the correct device
        self.model.train()  # Set the model to training mode

        meta_optimizer = self.optimizer  # Use the outer-loop optimizer

        # Loop over MAML epochs (outer loop)
        for epoch in range(1, maml_steps + 1):
            print(f"Epoch {epoch}: MAML Training")
            meta_optimizer.zero_grad()  # Reset outer optimizer
            
            meta_loss = 0.0  # Initialize meta-loss
            
            # Loop over tasks (rat data)
            for task_data, task_labels in zip(datas, labels):
                task_loader = cebra.data.Loader(task_data, task_labels, batch_size=len(task_data))  # Create task loader

                # Create a copy of the model for the inner loop
                model_copy = copy.deepcopy(self.model)
                inner_optimizer = torch.optim.SGD(model_copy.parameters(), lr=maml_lr)

                # Inner loop: perform task-specific gradient updates
                for batch in task_loader:
                    stats = self.step(batch, model=model_copy)  # Perform gradient update on model_copy
                    loss = stats['total']
                    inner_optimizer.zero_grad()
                    loss.backward()  # Backpropagate loss for the current task
                    inner_optimizer.step()

                # Compute meta-loss for the current task
                task_meta_loss = self._meta_loss(task_loader, model_copy)
                meta_loss += task_meta_loss

            # Average the meta-loss across tasks
            meta_loss /= len(datas)
            meta_loss.backward()  # Backpropagate the total meta-loss
            meta_optimizer.step()  # Update model using meta-gradient

            print(f"Meta Loss: {meta_loss.item():.6f}")
            
            # Optionally save the model at regular intervals
            if save_frequency is not None and epoch % save_frequency == 0:
                self.save(logdir, f"checkpoint_{epoch:#07d}.pth")

            # Optionally perform decoding or additional saving
            if decode:
                self.decode_history.append(self.decoding(datas, labels))

        # Save final model checkpoint
        self.save(logdir, f"checkpoint_final.pth")

    def step(self, batch, model=None):
        """Perform a single gradient update on the model"""
        if model is None:
            model = self.model

        self.optimizer.zero_grad()
        prediction = self._inference(batch)
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

    def save(self, logdir, filename="checkpoint_last.pth"):
        """Save the model state"""
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        savepath = os.path.join(logdir, filename)
        torch.save(self.state_dict(), savepath)

    def state_dict(self):
        """Return the state of the solver"""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss": torch.tensor(self.history),
            "log": self.log,
        }

