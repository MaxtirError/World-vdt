import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from core.trainers.utils import get_model_cls, show_supported_models
from core.schemas import Args
import torch
# memory_fraction = 0.5
# torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)
## for debug
def main():
    args = Args.parse_args()
    trainer_cls = get_model_cls(args.model_name, args.training_type)
    trainer = trainer_cls(args)
    trainer.fit()


if __name__ == "__main__":
    main()
