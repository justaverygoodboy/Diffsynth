accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/HDTF_MEAD_train \
  --dataset_metadata_path data/metadata_with_depth.csv \
  --height 256 \
  --width 256 \
  --dataset_repeat 10 \
  --learning_rate 1e-5 \
  --num_epochs 10 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-Fun-1.3B-InP_full-concat" \
  --trainable_models "dit" \
  --extra_inputs "input_image,end_image" \
  --depth_base_path data/HDTF_MEAD_depth \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-1.3B-InP:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --training_mode "rgb" \
  # --model_paths '["./models/PAI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors", "./models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth", "./models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", "./models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"]' \