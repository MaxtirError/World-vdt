import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from datasets.tartanair import TartanairLoader
from utils import *
from renderers.gaussian_renderer import GaussianRenderer
import imageio
from torchvision.utils import save_image
from diffusers import AutoencoderKLCogVideoX

def save_to_video(images : torch.Tensor, save_path : str, value_range : Tuple[float, float] = (0, 1)):   
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    images = (images - value_range[0]) / (value_range[1] - value_range[0])
    images = (images * 255).astype(np.uint8)
    imageio.mimsave(save_path, images, fps=10)
import click
@click.command()
@click.argument('root', type=str, default='~/data/tartanair')
def main(root):
    np.random.seed(10)
    root = os.path.expanduser(root)
    loader = TartanairLoader(root,
        window_size=100,
        max_num_history=4,
        frame_size=49,
        image_size=(480, 720),
        max_depth_range=128,
        debug=True)
    data = loader(200, num_history=1)
    pc_scene = data['history_scene']
    pc_scene = pc_scene.to('cuda')
    renderer = GaussianRenderer(resolution=(480, 720))
    video = pc_scene.warp_to_rgba(renderer)
    save_to_video(video, 'output.mp4')
    # pretrained_model_name_or_path = "THUDM/CogVideoX-5b-I2V"
    # vae = AutoencoderKLCogVideoX.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    # vae = vae.to('cuda')
    # vae.eval()
    # vae.enable_slicing()
    # vae.enable_tiling()
    # original_video = data['frames']
    # save_to_video(original_video, 'input.mp4')
    # [F, 3, H, W] -> [3, F, H, W]
    # with torch.no_grad():
    #     video_latent = vae.encode(original_video.transpose(0, 1)[None, ...].cuda()).latent_dist.sample()
    #     print(video_latent.shape)
    #     frames = vae.decode(video_latent).sample
    #     print(frames.shape)
    # save_to_video(frames[0].transpose(0, 1), 'output_vae.mp4')

    
    
if __name__ == '__main__':
    main()    

