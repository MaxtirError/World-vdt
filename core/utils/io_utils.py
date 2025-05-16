import imageio
from typing import Union, IO
import numpy as np
import os
def write_video(path: Union[str, os.PathLike, IO], video: np.ndarray, fps: int = 10):
    """
    Write a video, input uint8 RGB array of shape (T, H, W, 3).
    """
    writer = imageio.get_writer(path, fps=fps)
    for frame in video:
        writer.append_data(frame)
    writer.close()