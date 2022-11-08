export MODEL_NAME="./model"
export INSTANCE_DIR="./datasets/test2"
export OUTPUT_DIR="./new_model"
export CLASS_DIR="class"
export LOG_DIR="/root/tf-logs"

rm -rf $CLASS_DIR/*
rm -rf $LOG_DIR/*
# tensorboard --logdir=$LOG_DIR/dreambooth --port=6007 & #系统默认启动了tensorboard，如果没有可以手动启动

accelerate launch tools/train_dreambooth.py \
  --train_text_encoder \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --mixed_precision="fp16" \
  --instance_data_dir=$INSTANCE_DIR \
  --instance_prompt="a photo of <xxx> cat" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a photo of cat" \
  --output_dir=$OUTPUT_DIR \
  --logging_dir=$LOG_DIR \
  --center_crop \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=1000 \
  --save_model_every_n_steps=300
  
# 如果max_train_steps改大了，请记得把save_model_every_n_steps也改大
# 不然磁盘很容易中间就满了


# 以下是核心参数介绍：
# 主要的几个
# --train_text_encoder 训练文本编码器
# --mixed_precision="fp16" 混合精度训练
# - center_crop 
# 是否裁剪图片，一般如果你的数据集不是正方形的话，需要裁剪
# - resolution 
# 图片的分辨率，一般是512，使用该参数会自动缩放输入图像
# 可以配合center_crop使用，达到裁剪成正方形并缩放到512*512的效果
# - instance_prompt
# 如果你希望训练的是特定的人物，使用该参数
# 如 --instance_prompt="a photo of <xxx> girl"
# - class_prompt
# 如果你希望训练的是某个特定的类别，使用该参数可能提升一定的训练效果
# - use_txt_as_label
# 是否读取与图片同名的txt文件作为label
# 如果你要训练的是整个大模型的图像风格，那么可以使用该参数
# 该选项会忽略instance_prompt参数传入的内容
# - learning_rate
# 学习率，一般是2e-6，是训练中需要调整的关键参数
# 太大会导致模型不收敛，太小的话，训练速度会变慢
# - max_train_steps
# 训练的最大步数，一般是1000，如果你的数据集比较大，那么可以适当增大该值
# - save_model_every_n_steps
# 每多少步保存一次模型，方便查看中间训练的结果找出最优的模型，也可以用于断点续训