# 这是另一种finetune模型的方法，名为textual inversion，效果一般，仅内置一份供参考。
# 提示：该方法训练出的概念编码只能在diffusers使用。
#!/sbin/bash

accelerate launch ../tools/textual_inversion.py \
  --pretrained_model_name_or_path="./xxx/" \
  --train_data_dir="./datasets/xxx/" \
  --learnable_property="style" \
  --placeholder_token="<xxx>" --initializer_token="xxx" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=6000 \
  --learning_rate=1.0e-05 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --save_steps=200 \
  --mixed_precision="fp16" \
  --output_dir="output_model" 