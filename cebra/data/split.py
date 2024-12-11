def split_tasks_by_individuals(data_list, label_list):
    """
    Split the data into tasks based on individuals and return as CEBRA-compatible Batch objects.

    Args:
        data_list (list): List of neural data arrays for each individual.
        label_list (list): List of label arrays corresponding to each individual.

    Returns:
        tasks (list): List of CEBRA Batch objects for each individual.
    """
    tasks = []
    for i, (data, labels) in enumerate(zip(data_list, label_list)):
        # Convert data and labels to torch tensors
        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        # Create a CEBRA Batch object
        task_batch = Batch(
            reference=data_tensor,     # Data for the "reference" input
            positive=labels_tensor,    # Labels treated as "positive" inputs
            negative=labels_tensor     # Duplicate labels for "negative" input for simplicity
        )
        tasks.append(task_batch)
    return tasks
