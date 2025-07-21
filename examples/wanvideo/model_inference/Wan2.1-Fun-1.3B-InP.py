import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
from diffsynth.models.model_manager import ModelManager
import imageio
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path="./models/train/Wan2.1-Fun-1.3B-InP_full-concat/epoch-1.safetensors", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-1.3B-InP", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-1.3B-InP", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-1.3B-InP", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
    ],
)
def extract_first_frame_from_video(video_path):
    """从视频文件中提取第一帧"""
    reader = imageio.get_reader(video_path)
    first_frame = reader.get_data(0)  # 获取第一帧
    reader.close()
        
    # 转换为PIL图像
    frame_image = Image.fromarray(first_frame).convert("RGB")
    print(f"✅ 成功提取第一帧: {video_path}")
    return frame_image

def create_concatenated_input_image(rgb_img, depth_img):
    """创建垂直拼接的输入图像：RGB上 + Depth下"""
    # 确保两个图像尺寸一致
    width, height = rgb_img.size
    depth_img = depth_img.resize((width, height), Image.LANCZOS)
    
    # 创建拼接图像：256x256 + 256x256 → 256x512
    concatenated_height = height * 2
    concatenated_image = Image.new('RGB', (width, concatenated_height))
    
    # RGB在上半部分
    concatenated_image.paste(rgb_img, (0, 0))
    
    # 深度在下半部分
    concatenated_image.paste(depth_img, (0, height))
    
    print(f"🔧 创建拼接输入图像: {width}x{height} + {width}x{height} → {width}x{concatenated_height}")
    return concatenated_image
def separate_concatenated_video(video_frames, rgb_height=256):
    """
    分离垂直拼接的视频帧
    
    Args:
        video_frames: 拼接的视频帧列表
        rgb_height: RGB部分的高度
    
    Returns:
        rgb_frames, depth_frames: 分离后的RGB和深度视频帧
    """
    rgb_frames = []
    depth_frames = []
    
    for frame in video_frames:
        # 获取frame的numpy数组
        frame_array = np.array(frame)
        height, width = frame_array.shape[:2]
        
        # 分离RGB（上半部分）和深度（下半部分）
        rgb_part = frame_array[:rgb_height, :]
        depth_part = frame_array[rgb_height:, :]
        
        # 转换回PIL图像
        rgb_frame = Image.fromarray(rgb_part)
        depth_frame = Image.fromarray(depth_part)
        
        rgb_frames.append(rgb_frame)
        depth_frames.append(depth_frame)
    
    return rgb_frames, depth_frames
# print("加载联合训练权重...")
# model_manager = ModelManager()
# model_manager.load_model("./models/train/Wan2.1-Fun-1.3B-InP_full-joint/epoch-0.safetensors")
# pipe.dit = model_manager.fetch_model("wan_video_dit")
# print("加载模型完成...")
pipe.enable_vram_management()

rgb_video_path = "/gemini/platform/public/zqni/DiffSynth/data/HDTF_MEAD_train/M040_front_neutral_level_1_007.mp4"
depth_video_path = "/gemini/platform/public/zqni/DiffSynth/data/HDTF_MEAD_depth/M040_front_neutral_level_1_007_depth.mp4"
rgb_first_frame = extract_first_frame_from_video(rgb_video_path)
depth_first_frame = extract_first_frame_from_video(depth_video_path)
if rgb_first_frame and depth_first_frame:
    # 调整图像尺寸为256x256
    rgb_first_frame = rgb_first_frame.resize((256, 256), Image.LANCZOS)
    depth_first_frame = depth_first_frame.resize((256, 256), Image.LANCZOS)
        
    print(f"🖼️ RGB第一帧尺寸: {rgb_first_frame.size}")
    print(f"🖼️ 深度第一帧尺寸: {depth_first_frame.size}")
    # 保存第一帧图像用于查看
    rgb_first_frame.save("extracted_rgb_frame.png")
    depth_first_frame.save("extracted_depth_frame.png")
    print("💾 已保存提取的第一帧图像: extracted_rgb_frame.png, extracted_depth_frame.png")
    # 创建垂直拼接的输入图像
    input_image = create_concatenated_input_image(rgb_first_frame, depth_first_frame)
    # 保存拼接图像用于查看
    input_image.save("concatenated_input_image.png")
    print("💾 已保存拼接输入图像: concatenated_input_image.png")
    # 生成视频
    print("🎬 开始生成视频...")

    # First and last frame to video
    video = pipe(
        prompt="A people is talking.",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=input_image,
        seed=0, tiled=True,
        height=512, width=256,
        # inference_mode="joint",  # "joint" for joint output, "rgb" for RGB only
        # You can input `end_image=xxx` to control the last frame of the video.
        # The model will automatically generate the dynamic content between `input_image` and `end_image`.
    )
# 修复字典键名访问
# if isinstance(video, dict):
#     rgb_video = video["rgb_video"]  # 不是 "rgb_video"
#     depth_video = video["depth_video"]  # 不是 "depth_video"
#     save_video(rgb_video, "video_rgb.mp4", fps=15, quality=5)
#     save_video(depth_video, "video_depth.mp4", fps=15, quality=5)
# else:
    # 如果不是联合输出，直接保存
    save_video(video, "video_concat.mp4", fps=15, quality=5)
