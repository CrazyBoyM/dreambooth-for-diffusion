from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--origin_image_path", default=None, type=str, help="Path to the images to convert.")
parser.add_argument("--output_image_path", default=None, type=str, help="Path to the output images.")
parser.add_argument("--max_side_length", default=None, type=int, help="Path to the images to convert.")
args = parser.parse_args()


def resize_images(folder_path,output_folder_path, max_side_length):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder_path, filename)
            
            image = Image.open(image_path)
            image = image.convert('RGB')
            
            width, height = image.size
            
            scale = min(max_side_length / width, max_side_length / height)
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
            
            resized_image.save(output_path)

input_path = args.origin_image_path
output_path = args.output_image_path
if output_path!=None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

max_side_length = args.max_side_length # 最大边长 24G <= 786
resize_images(input_path,output_path, max_side_length)
