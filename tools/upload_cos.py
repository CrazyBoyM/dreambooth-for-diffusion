# -*- coding: UTF-8 -*-
# by ruochen
# 需要先执行 pip install -U cos-python-sdk-v5
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client

secret_id = 'abc123'  # 替换为用户的 secretId
secret_key = 'abc123'  # 替换为用户的 secretKey
region = 'ap-guangzhou'  # 替换为用户的 Region

config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)
client = CosS3Client(config)

response = client.upload_file(
    Bucket='xxx', # 替换为存储桶名称
    LocalFilePath='../ckpt_models/newModel.ckpt',  # 本地文件的路径
    Key='newModel.ckpt',  # 上传之后的文件名
)
print(response['ETag'])