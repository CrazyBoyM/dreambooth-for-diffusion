# 用于训练特定物体/人物的方法（只需单一标签）
export MODEL_NAME="./model"
export INSTANCE_DIR="./datasets/test2"
export OUTPUT_DIR="./new_model"
export CLASS_DIR="./datasets/class" # 用于存放模型生成的先验知识的图片文件夹，请勿改动
export LOG_DIR="/root/tf-logs"
export TEST_PROMPTS_FILE="./test_prompts_object.txt"

rm -rf $CLASS_DIR/* # 如果你要训练与上次不同的特定物体/人物，需要先清空该文件夹。其他时候可以注释掉这一行（前面加#）
rm -rf $LOG_DIR/*

accelerate launch tools/train_dreambooth_all.py \
  --train_text_encoder \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --mixed_precision="fp16" \
  --instance_data_dir=$INSTANCE_DIR \
  --instance_prompt="dog" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_prompt="dog" \
  --class_data_dir=$CLASS_DIR \
  --num_class_images=200 \
  --output_dir=$OUTPUT_DIR \
  --logging_dir=$LOG_DIR \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --auto_test_model \
  --test_prompts_file=$TEST_PROMPTS_FILE \
  --test_seed=123 \
  --test_num_per_prompt=3 \
  --max_train_steps=1000 \
  --save_model_every_n_steps=500 

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
# - lr_scheduler, 可选项有constant, linear, cosine, cosine_with_restarts, cosine_with_hard_restarts
# 学习率调整策略，一般是constant，即不调整，如果你的数据集很大，可以尝试其他的，但是可能会导致模型不收敛，需要调整学习率
# - lr_warmup_steps，如果你使用的是constant，那么这个参数可以忽略，
# 如果使用其他的，那么这个参数可以设置为0，即不使用warmup
# 也可以设置为其他的值，比如1000，即在前1000个step中，学习率从0慢慢增加到learning_rate的值
# 一般不需要设置, 除非你的数据集很大，训练收敛很慢
# - max_train_steps
# 训练的最大步数，一般是1000，如果你的数据集比较大，那么可以适当增大该值
# - save_model_every_n_steps
# 每多少步保存一次模型，方便查看中间训练的结果找出最优的模型，也可以用于断点续训

# --with_prior_preservation，--prior_loss_weight=1.0，分别是使用先验知识保留和先验损失权重
# 如果你的数据样本比较少，那么可以使用这两个参数，可以提升训练效果，还可以防止过拟合（即生成的图片与训练的图片相似度过高）

# --auto_test_model, --test_prompts_file, --test_seed, --test_num_per_prompt
# 分别是自动测试模型（每save_model_every_n_steps步后）、测试的文本、随机种子、每个文本测试的次数
# 测试的样本图片会保存在模型输出目录下的test文件夹中