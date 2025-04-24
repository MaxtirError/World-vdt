from core.trainers.base import Trainer
from typing_extensions import override

class FramePackSFTTrainer(Trainer):
    """Trainer class for FramePack SFT (Supervised Fine-Tuning) training."""
    
    @override
    def load_components(self):
        
        