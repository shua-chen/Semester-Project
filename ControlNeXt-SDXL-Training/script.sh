python run_controlnext.py \
 --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
 --pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
 --variant fp16 \
 --output_dir="test" \
 --validation_image "examples/vidit_depth/condition_1.png" \
 --validation_prompt "a high resolution human face image"  \
 --controlnet_model_name_or_path "train/example/checkpoints/checkpoint-25000/controlnet.safetensors" \
 --unet_model_name_or_path "train/example/checkpoints/checkpoint-25000/unet_weight_increasements.safetensors" \
 --num_validation_images 1



