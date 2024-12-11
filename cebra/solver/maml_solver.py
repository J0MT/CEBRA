import torch
import copy
import cebra
from torch.optim import SGD
from cebra import CEBRA

class MAMLSolver:
    def __init__(self, model, criterion, optimizer):
        self.model = model  # CEBRA model or multi-session model
        self.criterion = criterion  # CEBRA's contrastive loss or another criterion
        self.optimizer = optimizer  # Optimizer for the outer loop
    
    def maml_train(self, datas, labels, maml_steps=5, maml_lr=1e-3, save_frequency=None, logdir="./checkpoints", decode=False):
        """MAML training loop integrated with CEBRA's Solver."""
        
        self.to("cuda" if torch.cuda.is_available() else "cpu")  # Move model to the appropriate device
    
        # Set the actual model inside CEBRA in training mode
        self.model.model_[0].train()  # Access the model inside CEBRA and set it to training mode
    
        meta_optimizer = self.optimizer  # Use the outer-loop optimizer
    
        # Outer loop (MAML) iteration
        for epoch in range(1, maml_steps + 1):
            print(f"Epoch {epoch}: MAML Training")
            meta_optimizer.zero_grad()  # Reset outer loop optimizer
            
            meta_loss = 0.0  # Initialize meta-loss for this epoch
    
            # Loop over tasks (each task is a session or model in `model_`)
            for task_idx, (task_data, task_labels) in enumerate(zip(datas, labels)):
                task_loader = cebra.data.Loader(task_data, task_labels, batch_size=len(task_data))  # DataLoader for the task
                
                # Access the model for the current task (session) using model_[task_idx]
                task_model = self.model.model_[task_idx]  # Assuming model_[task_idx] holds a model for each task
                
                # Make a copy of the model for the inner-loop updates
                model_copy = copy.deepcopy(task_model)
                inner_optimizer = torch.optim.SGD(model_copy.parameters(), lr=maml_lr)  # Use SGD for inner-loop updates
                
                # Perform inner-loop training (gradient update per task)
                for batch in task_loader:
                    loss = self.step(batch, model=model_copy)  # Train using the copied model
                    inner_optimizer.zero_grad()
                    loss.backward()  # Backpropagate the loss
                    inner_optimizer.step()  # Update model for the task
                
                # Compute the task-specific meta-loss (evaluate model_copy)
                task_meta_loss = self._meta_loss(task_loader, model_copy)
                meta_loss += task_meta_loss  # Add to the total meta-loss for this epoch
    
            # Average the meta-loss across all tasks (outer loop)
            meta_loss /= len(datas)  # Average meta-loss over tasks
            meta_loss.backward()  # Backpropagate the meta-loss
            meta_optimizer.step()  # Update the model based on the meta-loss
            
            print(f"Meta Loss: {meta_loss.item():.6f}")
    
            # Optionally, save the model at regular intervals
            if save_frequency and epoch % save_frequency == 0:
                self.save(logdir, f"checkpoint_{epoch:#07d}.pth")
            
            if decode:
                self.decode_history.append(self.decoding(datas, labels))
    
        # Final model save
        self.save(logdir, "checkpoint_final.pth")


    def step(self, batch, model=None):
        """Perform a single gradient update for MAML."""
        if model is None:
            model = self.model
        self.optimizer.zero_grad()
        prediction = model(batch.reference)  # Forward pass
        loss, align, uniform = self.criterion(prediction.reference, prediction.positive, prediction.negative)
        loss.backward()  # Backpropagate
        self.optimizer.step()  # Update model parameters
        return loss

    def _meta_loss(self, batch, model):
        """Compute meta-loss for the task-specific model."""
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            prediction = model(batch.reference)  # Forward pass
            loss, _, _ = self.criterion(prediction, batch.positive, batch.negative)
        return loss

    def to_device(self, device):
        """Move model to the specified device (e.g., 'cuda' or 'cpu')."""
        self.model.to(device)
        self.criterion.to(device)

    def save(self, logdir, filename="checkpoint.pth"):
        """Save model parameters."""
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        savepath = os.path.join(logdir, filename)
        torch.save(self.model.state_dict(), savepath)
        print(f"Model saved to {savepath}")

    def decoding(self, datas, labels):
        """Optional function for decoding or additional metrics."""
        # Implement decoding logic if needed
        pass

