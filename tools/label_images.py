# from AUTOMATC1111
# maybe modified by Nyanko Lepsoni
# modified by crosstyan
import os.path
import re
import tempfile
import argparse
import glob
import zipfile
import deepdanbooru as dd
import tensorflow as tf
import numpy as np

from basicsr.utils.download_util import load_file_from_url
from PIL import Image
from tqdm import tqdm

re_special = re.compile(r"([\\()])")

def get_deepbooru_tags_model(model_path: str):
    if not os.path.exists(os.path.join(model_path, "project.json")):
        is_abs = os.path.isabs(model_path)
        if not is_abs:
            model_path = os.path.abspath(model_path)
            
        load_file_from_url(
            r"https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip",
            model_path,
        )
        with zipfile.ZipFile(
            os.path.join(model_path, "deepdanbooru-v3-20211112-sgd-e28.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall(model_path)
        os.remove(os.path.join(model_path, "deepdanbooru-v3-20211112-sgd-e28.zip"))

    tags = dd.project.load_tags_from_project(model_path)
    model = dd.project.load_model_from_project(model_path, compile_model=False)
    return model, tags


def get_deepbooru_tags_from_model(
    model,
    tags,
    pil_image,
    threshold,
    alpha_sort=False,
    use_spaces=True,
    use_escape=True,
    include_ranks=False,
):
    width = model.input_shape[2]
    height = model.input_shape[1]
    image = np.array(pil_image)
    image = tf.image.resize(
        image,
        size=(height, width),
        method=tf.image.ResizeMethod.AREA,
        preserve_aspect_ratio=True,
    )
    image = image.numpy()  # EagerTensor to np.array
    image = dd.image.transform_and_pad_image(image, width, height)
    image = image / 255.0
    image_shape = image.shape
    image = image.reshape((1, image_shape[0], image_shape[1], image_shape[2]))

    y = model.predict(image)[0]

    result_dict = {}

    for i, tag in enumerate(tags):
        result_dict[tag] = y[i]

    unsorted_tags_in_theshold = []
    result_tags_print = []
    for tag in tags:
        if result_dict[tag] >= threshold:
            if tag.startswith("rating:"):
                continue
            unsorted_tags_in_theshold.append((result_dict[tag], tag))
            result_tags_print.append(f"{result_dict[tag]} {tag}")

    # sort tags
    result_tags_out = []
    sort_ndx = 0
    if alpha_sort:
        sort_ndx = 1

    # sort by reverse by likelihood and normal for alpha, and format tag text as requested
    unsorted_tags_in_theshold.sort(key=lambda y: y[sort_ndx], reverse=(not alpha_sort))
    for weight, tag in unsorted_tags_in_theshold:
        tag_outformat = tag
        if use_spaces:
            tag_outformat = tag_outformat.replace("_", " ")
        if use_escape:
            tag_outformat = re.sub(re_special, r"\\\1", tag_outformat)
        if include_ranks:
            tag_outformat = f"({tag_outformat}:{weight:.3f})"

        result_tags_out.append(tag_outformat)

    # print("\n".join(sorted(result_tags_print, reverse=True)))

    return ", ".join(result_tags_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=".")
    parser.add_argument("--threshold", type=int, default=0.75)
    parser.add_argument("--alpha_sort", type=bool, default=False)
    parser.add_argument("--use_spaces", type=bool, default=True)
    parser.add_argument("--use_escape", type=bool, default=True)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--include_ranks", type=bool, default=False)

    args = parser.parse_args()

    global model_path
    model_path:str
    if args.model_path == "":
        script_path = os.path.realpath(__file__)
        default_model_path = os.path.join(os.path.dirname(script_path), "deepdanbooru-models")
        # print("No model path specified, using default model path: {}".format(default_model_path))
        model_path = default_model_path
    else:
        model_path = args.model_path

    types = ('*.jpg', '*.png', '*.jpeg', '*.gif', '*.webp', '*.bmp') 
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(os.path.join(args.path, files)))
        # print(glob.glob(args.path + files))
        
    model, tags = get_deepbooru_tags_model(model_path)
    for image_path in tqdm(files_grabbed, desc="Processing"):
        image = Image.open(image_path).convert("RGB")
        prompt = get_deepbooru_tags_from_model(
            model,
            tags,
            image,
            args.threshold,
            alpha_sort=args.alpha_sort,
            use_spaces=args.use_spaces,
            use_escape=args.use_escape,
            include_ranks=args.include_ranks,
        )
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        txt_filename = os.path.join(args.path, f"{image_name}.txt")
        # print(f"writing {txt_filename}: {prompt}")
        with open(txt_filename, 'w') as f:
            f.write(prompt)
        
