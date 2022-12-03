# Dreambooth Stable Diffusion 集成化环境训练
如果你是在autodl上的机器可以直接使用封装好的镜像创建实例，开箱即用  
如果是本地或者其他服务器上也可以使用，需要手动安装一些pip包

## 如何运行
直接在autodl使用镜像运行：https://www.codewithgpu.com/i/CrazyBoyM/dreambooth-for-diffusion/dreambooth-for-diffusion  

如果你不熟悉notebook代码的训练方式，也可以直接使用封装好的webui在线镜像（含稳定Dreambooth、dreamArtist训练插件，已fix）：  
https://www.codewithgpu.com/i/CrazyBoyM/sd_dreambooth_extension_webui/dreambooth-dreamartist-for-webui

## 注意
本项目仅供用于学习、测试人工智能技术使用  
请勿用于训练生成不良或侵权图片内容

## 关于项目
在autodl封装的镜像名称为：dreambooth-for-diffusion  
可在创建实例时直接选择公开的算法镜像使用。  
在autodl内蒙A区A5000的机器上封装，如遇到问题且无法自行解决的朋友请使用同一环境。  
白菜写教程时做了尽可能多的测试，但仍然无法确保每一个环节都完全覆盖    
如有小错误可尝试手动解决，或者访问git项目地址查看最新的README  
项目地址：https://github.com/CrazyBoyM/dreambooth-for-diffusion

如果遇到问题可到b站主页找该教程对应训练演示的视频：https://space.bilibili.com/291593914
（因为现在写时视频还没做）

## 强烈建议
1.用vscode的ssh功能远程连接到本服务器，训练体验更好，autodl自带的notebook也不错，有文件上传、下载功能。  
2.(重要)先把/root/目录下dreambooth-for-diffusion文件夹整个移动到/root/autodl-tmp/路径下(数据盘)，避免系统盘空间满

## 进入工作文件夹
```
cd /root/autodl-tmp/dreambooth-for-diffusion
```

## 转换ckpt检查点文件为diffusers官方权重
已经内置了两个基础模型，可以根据自己数据集的特性选择。    
- sd_1-5.ckpt是偏真实风格  
- nd_lastest.ckpt是偏二次元风格  
开始转换二次元模型：
```
# 该步需要运行大约一分钟 
!python tools/ckpt2diffusers.py \
    --checkpoint_path=./ckpt_models/nd_lastest.ckpt \
    --dump_path=./model \
    --vae_path=./ckpt_models/animevae.pt \
    --original_config_file=./ckpt_models/model.yaml \
    --scheduler_type="ddim"
```
转换写实风格模型：
```
# 该步需要运行大约一分钟 
!python tools/ckpt2diffusers.py \
    --checkpoint_path=./ckpt_models/sd_1-5.ckpt \
    --dump_path=./model \
    --original_config_file=./ckpt_models/model.yaml \
    --scheduler_type="ddim"
```
这里后面跟的两个文件分别是你的ckpt文件和转换后的输出路径。

## 转换diffusers官方权重为ckpt检查点文件
```
python tools/diffusers2ckpt.py ./new_model ./ckpt_models/newModel_half.ckpt --half
```
如需保存为float16版精度，添加--half参数，权重大小会减半。

## 准备数据集
请按照训练任务准备好对应的数据集。
### 图像裁剪为512*512
我在tools/handle_images.py中提供了一份批量处理的代码用于参考  
自动center crop图像，并缩放尺寸
```
python tools/handle_images.py ./datasets/test ./datasets/test2 --width=512 --height=512
```
test为未处理的原始图像文件夹，test2为输出处理图像的路径  
如需处理透明背景png图为黑色/白色底jpg，可以添加--png参数。

### 图像自动标注
使用deepdanbooru生成tags label.
```
!python tools/label_images.py  --path=./datasets/test2 
```
第二个参数--path为你需要标注的图像文件夹路径   

注：如提示deepdanbooru找不到，可自行参考以下仓库进行编译  
https://github.com/KichangKim/DeepDanbooru  

我在other文件夹下也提供了一份编译好的版本：
```
pip install other/deepdanbooru-1.0.0-py3-none-any.whl 
```

## 训练以及常用命令总结
### 配置训练环境（可选）
如果你不是在封装好的镜像上直接使用，则需要做以下配置：
```
pip install accelerate
```
运行以下命令，并选择本地运行、NO、NO
```
accelerate config
```

### 开始训练 
请打开train.sh文件，参考其中的具体参数说明。  
如果需要训练特定人、事物： 
（推荐准备3~5张风格统一、特定对象的图片）

```
sh train_object.sh
```

如果要Finetune训练自己的大模型： 
（推荐准备3000+张图片，包含尽可能的多样性，数据决定训练出的模型质量）
```
sh train_style.sh
```
A5000的训练速度大概8分钟/1000步

### 测试训练效果
打开train/test_model.py文件修改其中的model_path和prompt，然后执行：
```
python test_model.py
```

### 其他常用命令
如需后台任务训练：  
```
nohup sh train_style.sh &
```
推荐晚上这样挂后台跑着，不需要担心连接中断导致的训练停止。
白菜个人推荐的省钱训练小妙招:
```
nohup sh back_train.sh &
```
(训练完直接自动关机)

训练日志会输出到nohup.out文件中，可以vscode直接打开或下载查看。  
查看日志后十行：  
```
tail -n 10 nohup.out
```

查看当前磁盘占有率：  
（记得清理不要的文件，不然经常容易磁盘几十个g空间满导致模型保存失败！！）  
```
df -h
```

## 如果你是在其他服务器上执行，没有使用集成环境
提示缺少一些包可以自行安装：
```
pip install diffusers
pip install ftfy
pip install tensorflow-gpu
pip install pytorch_lightning
pip install OmegaConf
... 以及其他的一些
```

## 学术加速（可选）
如果你需要拉取git上一些内容，发现速度很慢，以下内容或许对你有帮助。  
请根据机器所在区域执行以下命令：
```
北京A区的实例¶
export http_proxy=http://100.72.64.19:12798 && export https_proxy=http://100.72.64.19:12798

内蒙A区的实例¶
export http_proxy=http://192.168.1.174:12798 && export https_proxy=http://192.168.1.174:12798

泉州A区的实例¶
export http_proxy=http://10.55.146.88:12798 && export https_proxy=http://10.55.146.88:12798
```

## xformers(可选)
由于A5000实测训练和推理的速度已经很快了，就没有安装。  
如果你使用的是其他显卡或者实在有需要，可以参考下面的地址进行编译使用：  
https://github.com/facebookresearch/xformers  
（我猜到你可能想要尝试，已经在train/other目录下放了一个提前编译好的版本啦）  
注：需要升级pytorch版本到1.12.x以上才能安装使用（好懒）(更新：我已经升级好并帮你装好啦~！)

## 升级pytorch版本到1.12.x
```
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

# 关于autodl的使用心得

## 服务器的数据迁移
经常关机后再开机发现机器资源被占用了，这时候你只能另外开一台机器了  
但是对于已经关机的机器在菜单上有个功能是“跨实例拷贝数据”，  
可以很方便地同步/root/autodl-tmp文件夹下的内容到其他已开机的机器（所以推荐工作文件都放这）  
（注意，只适用于同一区域的机器之间）
数据迁移教程：https://www.autodl.com/docs/migrate_instance/

## 传输文件的方式
### 方式一 使用VScode
直接从vscode拖动上传、下载文件，速度慢，也最简单。  

### 方式二 使用autodl的用户网盘
在autodl的网盘界面初始化一个同区域的网盘，然后重启一下服务器实例  
会发现多了一个文件夹/root/autodl-nas/, 你可以在网页界面进行权重和数据的上传  
训练完，把生成的权重文件移动到该路径下，就可以去网页上进行下载了。  
（对应网页：https://www.autodl.com/console/netdisk）
注意：初始化的网盘一定要和服务器处于同一区域.  

### 方式三 使用对象存储
有条件的朋友也可以尝试使用cos或oss进行文件中转，速度更快。  
在train/tools文件夹中我也放置了一份上传到cos的代码供参考，请有经验的朋友自行使用。  

autodl官网也有一些推荐的方式可以参考，https://www.autodl.com/docs/scp/

# 其他内容
感谢diffusers、deepdanbooru等开源项目  
风格训练代码来自nbardy的PR进行修改  
打tags标签的部分代码来自crosstyan、Nyanko Lepsoni、AUTOMATC1111  
如果感兴趣欢迎加QQ群探讨交流，455521885  
封装整理by - 白菜 
