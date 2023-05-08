import os, cv2, argparse
import numpy as np

# 修改透明背景为白色
def transparence2white(img):
    sp=img.shape
    width=sp[0]
    height=sp[1]
    for yh in range(height):
        for xw in range(width):
            color_d=img[xw,yh]
            if(color_d[3]==0):
                img[xw,yh]=[255,255,255,255]
    return img

# 修改透明背景为黑色
def transparence2black(img):
    sp = img.shape
    width = sp[0]
    height = sp[1]
    for yh in range(height):
        for xw in range(width):
            color_d = img[xw, yh]  
            if (color_d[3] == 0):
                img[xw, yh] = [0, 0, 0, 255] 
    return img

# 中心裁剪
def center_crop(img, crop_size):
    h, w = img.shape[:2]
    th, tw = crop_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return img[i:i + th, j:j + tw] 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--origin_image_path", default=None, type=str, help="Path to the images to convert.")
    parser.add_argument("--output_image_path", default=None, type=str, help="Path to the 1:1 output images.")
    parser.add_argument("--output_image_path_0", default=None, type=str, help="Path to the 3:2 output images.")
    parser.add_argument("--output_image_path_1", default=None, type=str, help="Path to the 2:3 output images.")
    parser.add_argument("--width", default=512, type=int, help="Width of the output images.")
    parser.add_argument("--height", default=512, type=int, help="Height of the output images.")
    parser.add_argument("--png", action="store_true", help="convert the transparent background to white/black.")

    
    args = parser.parse_args()
    
    path = args.origin_image_path
    save_path = args.output_image_path
    save_path_0 = args.output_image_path_0
    save_path_1 = args.output_image_path_1
    if save_path!=None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    if save_path_0!=None:
        if not os.path.exists(save_path_0):
            os.makedirs(save_path_0)
    if save_path_1!=None:
        if not os.path.exists(save_path_1):
            os.makedirs(save_path_1)
    else:
        print('The folder already exists, please check the path.')

# 只读取png、jpg、jpeg、bmp、webp格式
allow_suffix = ['png', 'jpg', 'jpeg', 'bmp', 'webp']
image_list = os.listdir(path)
image_list = [os.path.join(path, image) for image in image_list if image.split('.')[-1] in allow_suffix]

width = args.width
height = args.height
ratio = 3 / 2
for file, i in zip(image_list, range(1, len(image_list)+1)):
    print('Processing image: %s' % file)
    try:
        img = cv2.imread(file)
        if width==height:
           # 对图像进行center crop, 保证图像的长宽比为1:1, crop_size为图像的较短边
            crop_size = min(img.shape[:2])
            img = center_crop(img, (crop_size, crop_size))
            # 缩放图像
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            if args.png:
                img = transparence2black(img)
            cv2.imwrite(os.path.join(save_path, str(i).zfill(4) + ".jpg"), img)
        else:
            height_temp, width_temp, _ = img.shape
            # 如果宽度大于高度，则裁剪成3:2的宽高比
            if width_temp > height_temp:
                new_width = width_temp
                new_height = int(width_temp / ratio)
                left = 0
                top = 0
                img = img[top:top+new_height, left:left+new_width]
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                if args.png:
                    img = transparence2black(img)
                cv2.imwrite(os.path.join(save_path_0, str(i).zfill(4) + ".jpg"), img)
            else:
                # 反之，则裁剪成2:3的宽高比
                new_height = height_temp
                new_width = int(height_temp * ratio)
                left = 0
                top = 0
                img = img[top:top+new_height, left:left+new_width]
                img = cv2.resize(img, (height,width), interpolation=cv2.INTER_AREA)
                if args.png:
                    img = transparence2black(img)
                cv2.imwrite(os.path.join(save_path_1, str(i).zfill(4) + ".jpg"), img)
        # img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        
        # # 如果是透明图，将透明背景转换为白色或者黑色
        # if args.png:
        #     img = transparence2black(img)
            
        # cv2.imwrite(os.path.join(save_path, str(i).zfill(4) + ".jpg"), img)
    except Exception as e:
        print(e)
        os.remove(path+file) # 删除无效图片
        print("删除无效图片: " + path+file)