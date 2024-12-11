import torch
import copy
from cebra import CEBRA
from torch.optim import Adam
from cebra.solver import Solver

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Batch Class
class CustomBatch:
    def __init__(self, data, labels, device):
        # Ensure tensors are cast to float32 and moved to the correct device
        self.reference = torch.tensor(data, dtype=torch.float32).to(device)
        self.positive = torch.tensor(labels, dtype=torch.float32).to(device)
        self.negative = torch.zeros_like(self.positive).to(device)

# Custom Data Loader
class CustomLoader:
    def __init__(self, data, labels, batch_size, device):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            batch_data = self.data[i:i + self.batch_size].to(self.device)
            batch_labels = self.labels[i:i + self.batch_size].to(self.device)
            yield CustomBatch(batch_data, batch_labels, self.device)


# MAML Solver Implementation
class MAMLSolver(Solver):
    def _inference(self, batch):
        return self.model(batch.reference)

    def maml_train(self, datas, labels, maml_steps=5, maml_lr=1e-3, save_frequency=500, logdir="./checkpoints"):
        self.to(device)
        self.model.train()
        meta_optimizer = self.optimizer

        for epoch in range(1, maml_steps + 1):
            print(f"Epoch {epoch}: MAML Training")
            meta_optimizer.zero_grad()
            meta_loss = 0.0

            for task_data, task_labels in zip(datas, labels):
                task_loader = CustomLoader(task_data, task_labels, batch_size=len(task_data), device=device)
                model_copy = copy.deepcopy(self.model)
                inner_optimizer = torch.optim.SGD(model_copy.parameters(), lr=maml_lr)

                for batch in task_loader:
                    stats = self.step(batch)
                    loss = stats['total']
                    inner_optimizer.zero_grad()
                    loss.backward()
                    inner_optimizer.step()

                task_meta_loss = self._meta_loss(task_loader, model_copy)
                meta_loss += task_meta_loss

            meta_loss /= len(datas)
            meta_loss.backward()
            meta_optimizer.step()

            print(f"Meta Loss: {meta_loss.item():.6f}")
            if save_frequency and epoch % save_frequency == 0:
                self.save(logdir, f"checkpoint_{epoch:#07d}.pth")
        self.save(logdir, "checkpoint_final.pth")

    def step(self, batch):
        self.optimizer.zero_grad()
        prediction = self._inference(batch)
        loss, align, uniform = self.criterion(
            prediction.reference, prediction.positive, prediction.negative
        )
        loss.backward()
        self.optimizer.step()

        stats = dict(pos=align.item(), neg=uniform.item(), total=loss.item())
        return stats

    def _meta_loss(self, loader, model):
        model.eval()
        with torch.no_grad():
            for batch in loader:
                prediction = model(batch.reference)
                loss, _, _ = self.criterion(
                    prediction, batch.positive, batch.negative
                )
        return loss

