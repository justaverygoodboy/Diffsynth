#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
import argparse
import time
from pathlib import Path
from datetime import datetime
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from tqdm import tqdm
import multiprocessing as mp
from queue import Queue
import threading

def read_prompt_list(prompt_file):
    """读取prompt列表文件"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    return prompts

def create_output_filename(prompt, index, output_dir):
    """创建输出文件名"""
    # 【修改】简化文件名，避免prompt过长问题
    filename = f"wan2.1_1.3B_{index:04d}.mp4"
    return os.path.join(output_dir, filename)

def setup_pipeline(model_id, device_id=0):
    """设置WAN视频生成管道"""
    print(f"Setting up pipeline on GPU {device_id}...")
    
    # 设置CUDA设备
    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"
    
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model.safetensors", offload_device="cpu"),
            ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id=model_id, origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )
    pipe.enable_vram_management()
    
    print(f"✅ Pipeline ready on GPU {device_id}")
    return pipe

def generate_single_video(pipe, prompt, output_file, args, device_id):
    """生成单个视频"""
    try:
        # 【修改】显示完整prompt，不裁剪
        print(f"[GPU {device_id}] Generating: {prompt}")
        
        # 【修改】使用完整的prompt，不做任何处理
        video = pipe(
            prompt=prompt,  # 完整prompt
            negative_prompt=args.negative_prompt,
            seed=args.seed + hash(prompt) % 10000,  # 基于prompt的hash生成不同seed
            tiled=args.tiled,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        )
        
        # 保存视频
        save_video(video, output_file, fps=args.fps, quality=args.quality)
        
        # 检查文件是否成功生成
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"[GPU {device_id}] ✅ Generated: {output_file} ({file_size:.1f}MB)")
            return True
        else:
            print(f"[GPU {device_id}] ❌ File not created: {output_file}")
            return False
            
    except Exception as e:
        print(f"[GPU {device_id}] 💥 Error generating video: {e}")
        return False

def worker_process(gpu_id, prompt_queue, result_queue, args):
    """工作进程函数"""
    try:
        # 设置管道
        pipe = setup_pipeline(args.model_id, gpu_id)
        
        success_count = 0
        total_count = 0
        
        while True:
            try:
                # 从队列获取任务
                item = prompt_queue.get(timeout=5)
                if item is None:  # 结束信号
                    break
                
                prompt_index, prompt = item
                total_count += 1
                
                # 创建输出文件路径
                output_file = create_output_filename(prompt, prompt_index, args.output_dir)
                
                # 检查文件是否已存在
                if os.path.exists(output_file):
                    print(f"[GPU {gpu_id}] File exists, skipping: {output_file}")
                    success_count += 1
                    success = True
                else:
                    # 生成视频
                    success = generate_single_video(pipe, prompt, output_file, args, gpu_id)
                    if success:
                        success_count += 1
                
                # 【修改】返回结果时显示完整prompt
                result_queue.put({
                    'gpu_id': gpu_id,
                    'index': prompt_index,
                    'prompt': prompt,  # 完整prompt
                    'output_file': output_file,
                    'success': success
                })
                
            except Exception as e:
                print(f"[GPU {gpu_id}] Worker error: {e}")
                break
        
        print(f"[GPU {gpu_id}] Worker finished. Success: {success_count}/{total_count}")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Worker setup failed: {e}")

def batch_generate_videos(args):
    """批量生成视频的主函数"""
    # 读取prompt列表
    print(f"Reading prompts from: {args.prompt_file}")
    prompts = read_prompt_list(args.prompt_file)
    print(f"Found {len(prompts)} prompts")
    
    if args.max_prompts > 0:
        prompts = prompts[:args.max_prompts]
        print(f"Limited to {len(prompts)} prompts")
    
    # 【修改】确保输出目录为 gen_istock
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建队列
    prompt_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # 将prompts放入队列
    for i, prompt in enumerate(prompts):
        prompt_queue.put((i, prompt))
    
    # 添加结束信号
    for _ in range(args.num_gpus):
        prompt_queue.put(None)
    
    # 启动工作进程
    print(f"Starting {args.num_gpus} worker processes...")
    processes = []
    for gpu_id in range(args.num_gpus):
        p = mp.Process(target=worker_process, args=(gpu_id, prompt_queue, result_queue, args))
        p.start()
        processes.append(p)
        print(f"Started worker process for GPU {gpu_id}")
        time.sleep(2)  # 避免同时初始化模型
    
    # 监控进度
    completed = 0
    success_count = 0
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"Starting video generation...")
    print(f"{'='*80}")
    
    while completed < len(prompts):
        try:
            result = result_queue.get(timeout=300)
            completed += 1
            if result['success']:
                success_count += 1
            
            # 计算统计信息
            elapsed = time.time() - start_time
            avg_time = elapsed / completed
            eta = avg_time * (len(prompts) - completed)
            
            # 【修改】显示完整prompt，但限制显示长度以避免终端混乱
            display_prompt = result['prompt']
            if len(display_prompt) > 80:
                display_prompt = display_prompt[:77] + "..."
            
            print(f"Progress: {completed:4d}/{len(prompts)} ({completed/len(prompts)*100:5.1f}%) "
                  f"Success: {success_count:4d} "
                  f"ETA: {eta/60:6.1f}min "
                  f"[GPU {result['gpu_id']}] {display_prompt}")
            
        except Exception as e:
            print(f"Error monitoring progress: {e}")
            break
    
    # 等待所有进程完成
    print("\nWaiting for all processes to finish...")
    for p in processes:
        p.join(timeout=600)
        if p.is_alive():
            print(f"Force terminating process {p.pid}")
            p.terminate()
            p.join()
    
    # 输出最终统计
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Batch generation completed!")
    print(f"{'='*80}")
    print(f"Total prompts: {len(prompts)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(prompts) - success_count}")
    print(f"Success rate: {success_count/len(prompts)*100:.1f}%")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per video: {total_time/len(prompts):.1f} seconds")
    print(f"Output directory: {args.output_dir}")
    
    # 列出生成的文件
    generated_files = list(Path(args.output_dir).glob("wan2.1_1.3B_*.mp4"))
    print(f"Generated {len(generated_files)} video files")

def main():
    parser = argparse.ArgumentParser(description='Batch video generation using Wan2.1-T2V-1.3B')
    
    # 输入输出
    parser.add_argument('--prompt_file', 
                       default='/gemini/platform/public/zqni/istock_output/istock_copied_short.txt',
                       help='Path to the prompt list file')
    parser.add_argument('--output_dir', default='/gemini/platform/public/zqni/istock_output/istock_gen',  # 【修改】默认输出目录
                       help='Output directory for generated videos')
    parser.add_argument('--model_id', default='Wan-AI/Wan2.1-T2V-1.3B',
                       help='Model ID for WAN pipeline')
    
    # GPU设置
    parser.add_argument('--num_gpus', type=int, default=8,
                       help='Number of GPUs to use')
    parser.add_argument('--max_prompts', type=int, default=-1,
                       help='Maximum number of prompts to process (-1 for all)')
    
    # 生成参数
    parser.add_argument('--height', type=int, default=480,
                       help='Video height')
    parser.add_argument('--width', type=int, default=832,
                       help='Video width')
    parser.add_argument('--num_frames', type=int, default=81,
                       help='Number of frames')
    parser.add_argument('--fps', type=int, default=25,
                       help='Frames per second')
    parser.add_argument('--quality', type=int, default=5,
                       help='Video quality (1-10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base seed for generation')
    parser.add_argument('--tiled', action='store_true', default=True,
                       help='Use tiled generation')
    
    # Prompt设置
    parser.add_argument('--negative_prompt', 
                       default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                       help='Negative prompt for generation')
    
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.exists(args.prompt_file):
        print(f"Error: Prompt file not found: {args.prompt_file}")
        return 1
    
    # 检查GPU数量
    available_gpus = torch.cuda.device_count()
    if args.num_gpus > available_gpus:
        print(f"Warning: Requested {args.num_gpus} GPUs, but only {available_gpus} available")
        args.num_gpus = available_gpus
    
    print(f"Configuration:")
    print(f"  Prompt file: {args.prompt_file}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Model ID: {args.model_id}")
    print(f"  Number of GPUs: {args.num_gpus}")
    print(f"  Video size: {args.width}x{args.height}")
    print(f"  Frames: {args.num_frames}")
    print(f"  FPS: {args.fps}")
    print(f"  Quality: {args.quality}")
    print("")
    
    # 开始批量生成
    try:
        batch_generate_videos(args)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == '__main__':
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    exit(main())