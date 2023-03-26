# Requirement:
#   pip install --upgrade diffusers[torch] transformers
# More info: https://github.com/huggingface/diffusers/tree/main

from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
