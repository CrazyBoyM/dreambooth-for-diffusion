from diffusers import StableDiffusionPipeline
import torch
from diffusers import DDIMScheduler

model_path = "./new_model"  
prompt = "a cute girl, blue eyes, brown hair"

pipe = StableDiffusionPipeline.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        scheduler=DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=True,
        ),
        safety_checker=None
    )

# def dummy(images, **kwargs):
#     return images, False
# pipe.safety_checker = dummy
pipe = pipe.to("cuda")
images = pipe(prompt, num_inference_steps=30, num_images_per_prompt=3).images
for i, image in enumerate(images):
    image.save(f"test-{i}.png")
