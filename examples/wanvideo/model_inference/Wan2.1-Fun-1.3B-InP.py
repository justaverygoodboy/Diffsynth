import argparse
import numpy as np
from PIL import Image
import imageio
import torch
from diffsynth import save_video
from diffsynth.audio_utils import load_audio_features
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


def extract_first_frame(video_path: str) -> Image.Image:
    reader = imageio.get_reader(video_path)
    frame = reader.get_data(0)
    reader.close()
    return Image.fromarray(frame).convert("RGB")


def concat_input(rgb: Image.Image, depth: Image.Image) -> Image.Image:
    width, height = rgb.size
    depth = depth.resize((width, height), Image.LANCZOS)
    canvas = Image.new("RGB", (width, height * 2))
    canvas.paste(rgb, (0, 0))
    canvas.paste(depth, (0, height))
    return canvas


def main(args):
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=args.ckpt, offload_device="cpu"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-1.3B-InP", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-1.3B-InP", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-1.3B-InP", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
        ],
    )
    pipe.enable_vram_management()

    rgb = extract_first_frame(args.rgb_video).resize((256, 256), Image.LANCZOS)
    depth = extract_first_frame(args.depth_video).resize((256, 256), Image.LANCZOS)
    input_image = concat_input(rgb, depth)

    audio_feats = load_audio_features(args.audio_path, fps=args.fps, num_frames=args.num_frames)

    video = pipe(
        prompt="A person is talking.",
        input_image=input_image,
        audio_features=audio_feats,
        inference_mode="joint",
        seed=0,
        tiled=True,
        height=512,
        width=256,
        num_frames=args.num_frames,
    )

    if isinstance(video, dict):
        save_video(video["rgb_video"], "video_rgb.mp4", fps=args.fps, quality=5)
        save_video(video["depth_video"], "video_depth.mp4", fps=args.fps, quality=5)
    else:
        save_video(video, "video_concat.mp4", fps=args.fps, quality=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_video", type=str, required=True)
    parser.add_argument("--depth_video", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="./models/train/Wan2.1-Fun-1.3B-InP_full-concat/epoch-1.safetensors")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--num_frames", type=int, default=81)
    args = parser.parse_args()
    main(args)
