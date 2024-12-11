import copy
import torch
from cebra.solver.base import Solver
from cebra.data import Loader  # You may need to adjust this import based on your data format

class MAMLSolver(Solver):
    """Custom solver for MAML."""
    
    def __init__(self, model, criterion, optimizer, *args, **kwargs):
        # Initialize the parent Solver class
        super(MAMLSolver, self).__init__(model, criterion, optimizer, *args, **kwargs)

    def maml_train(self, datas, labels, maml_steps=1, maml_lr=1e-3, save_frequency=None, valid_frequency=None, decode=False, logdir=None):
        """MAML training: Meta-train across tasks (rats)."""
        
        self.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.train()

        meta_optimizer = self.optimizer  # Outer-loop optimizer
        
        # Loop over MAML epochs (outer loop)
        for epoch in range(1, maml_steps + 1):
            print(f"Epoch {epoch}: MAML Training")
            meta_optimizer.zero_grad()  # Reset the meta-optimizer
            
            meta_loss = 0.0
            
            # Loop over tasks (Rats 1, 2, 3) for the inner loop
            for task_data, task_labels in zip(datas, labels):
                task_loader = Loader(task_data, task_labels, batch_size=len(task_data))  # Adjust according to your Loader setup
                
                # Create a copy of the model for inner-loop updates
                model_copy = copy.deepcopy(self.model)  
                inner_optimizer = torch.optim.SGD(model_copy.parameters(), lr=maml_lr)
                
                # Perform training on this task (inner loop)
                for batch in task_loader:
                    stats = self.step(batch, model=model_copy)  # Train using the copied model
                    loss = stats['total']
                    inner_optimizer.zero_grad()
                    loss.backward()
                    inner_optimizer.step()

                # Compute meta-loss on the task (evaluate model_copy)
                task_meta_loss = self._meta_loss(task_loader, model_copy)
                meta_loss += task_meta_loss

            # Outer loop update: average meta-loss across tasks
            meta_loss /= len(datas)  # Average meta-loss across tasks (rats)
            meta_loss.backward()  # Backpropagate the meta-loss
            meta_optimizer.step()  # Update the model based on the meta-loss
            
            print(f"Meta Loss: {meta_loss.item():.6f}")
        
        # After training, save the model if needed
        if save_frequency is not None:
            self.save(logdir, f"checkpoint_{epoch:#07d}.pth")
        
        if decode:
            self.decode_history.append(self.decoding(datas, labels))

        if save_hook:
            save_hook(epoch, self)

    def step(self, batch, model=None):
        """Perform a single gradient update.

        Args:
            batch: The input samples
            model: The model to be used for updates (defaults to `self.model`)

        Returns:
            Dictionary containing training metrics.
        """
        if model is None:
            model = self.model

        self.optimizer.zero_grad()
        prediction = self._inference(batch)
        loss, align, uniform = self.criterion(prediction.reference,
                                              prediction.positive,
                                              prediction.negative)

        loss.backward()
        self.optimizer.step()
        self.history.append(loss.item())
        stats = dict(
            pos=align.item(),
            neg=uniform.item(),
            total=loss.item(),
            temperature=self.criterion.temperature,
        )
        for key, value in stats.items():
            self.log[key].append(value)
        return stats

    def _meta_loss(self, batch, model):
        """Compute meta-loss for the updated (task-specific) model."""
        model.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            prediction = model(batch.reference)  # Forward pass
            loss, _, _ = self.criterion(
                prediction,
                batch.positive,
                batch.negative,
            )
        return loss

    def _inference(self, batch):
        """Override this method to perform inference with your model."""
        raise NotImplementedError
