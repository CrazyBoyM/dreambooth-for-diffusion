# 这是另一种finetune模型的方法，名为textual inversion，效果一般，仅内置一份供参考。
# 提示：该方法训练出的概念编码只能在diffusers使用。暂时不支持在diffusers之外的推理框架使用。（如webui）
#!/sbin/bash
export LOG_DIR="/root/tf-logs"

accelerate launch ./tools/train_textual_inversion.py \
  --pretrained_model_name_or_path="./model/" \
  --train_data_dir="./datasets/test" \
  --learnable_property="style" \
  --placeholder_token="<xxx-girl>" --initializer_token="girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --save_steps=200 \
  --max_train_steps=3000 \
  --mixed_precision="fp16" \
  --logging_dir=$LOG_DIR \
  --output_dir="output_model" 

  # --learnable_property为style时训练特定风格，为object时训练特定物体/人物。
  # --placeholder_token为训练时的占位符，--initializer_token为训练时的初始化词。
  # --resolution为训练时的分辨率，--train_batch_size为训练时的batch size，--gradient_accumulation_steps为梯度累积步数。
  # --learning_rate为训练时的学习率，--scale_lr为是否对学习率进行缩放，--lr_scheduler为学习率调度器，--lr_warmup_steps为学习率预热步数。
  # --save_steps为保存模型的步数，--max_train_steps为最大训练步数，--mixed_precision为混合精度训练模式。
  # --logging_dir为日志保存路径，--output_dir为模型保存路径。
  # --pretrained_model_name_or_path为预训练模型路径，--train_data_dir为训练数据路径，必须为文件夹，文件夹内为处理后的图片。