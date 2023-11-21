from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm


class LightProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        """Disable validation tqdm"""
        bar = Tqdm(
            disable=True,
        )
        return bar
