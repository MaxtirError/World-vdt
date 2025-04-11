import time
import torch

class CUDATimer:
    """
    A context manager to measure and log the running time of CUDA operations.
    """
    def __init__(self, description: str):
        """
        Initialize the context manager with a description of the operation.

        Args:
            description (str): A description of the operation being timed.
        """
        self.description = description
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        # Record the start time
        print(f"Starting: {self.description}")
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Record the end time and synchronize
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time = self.start_event.elapsed_time(self.end_event)  # Time in milliseconds
        print(f"Completed: {self.description} in {elapsed_time:.2f} ms")
        
if __name__ == "__main__":
    # Example usage
    # Example CUDA operation
    with CUDATimer("Example CUDA operation"):
        # Simulate a CUDA operation
        # matrix multiplication or any other CUDA operation
        a = torch.randn(1000, 1000).cuda()
        b = torch.randn(1000, 1000).cuda()
        c = torch.matmul(a, b)
        # Simulate some delay