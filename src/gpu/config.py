import torch

class GPUConfig:
    # Automatic mixed precision
    AMP_ENABLED = True

    # Batch sizes for different operations
    BATCH_SIZES = {
        'fft': 1024,
        'lsb': 4096,
        'echo': 512
    }

    # Device configuration
    @staticmethod
    def auto_select_device():
        if torch.cuda.is_available():
            return f'cuda:{torch.cuda.current_device()}'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    # Optimization flags
    OPTIMIZATION = {
        'cudnn_benchmark': True,
        'memory_pinning': True,
        'tensor_cores': True
    }

    # Algorithm parameters
    ALGORITHM_PARAMS = {
        'lsb': {'num_bits': 2},
        'fft': {'strength': 0.01},
        'echo': {'delay': 0.1, 'decay': 0.3}
    }
