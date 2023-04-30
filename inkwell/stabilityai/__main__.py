import base64
import os
import logging
from pathlib import Path
from io import BytesIO

import fire
import requests
from PIL import Image

from inkwell.constants import (
    INKY_FRAME_7_3_HEIGHT,
    INKY_FRAME_7_3_WIDTH,
)


STYLE_PRESETS = ["digital-art", "neon-punk"]
ENGINE_ID = os.getenv("ENGINE_ID", "stable-diffusion-768-v2-1")
API_KEY = os.getenv("STABILITY_API_KEY")


def main(prompt, dest):
    if API_KEY is None:
        raise Exception("Missing the env variable: STABILITY_API_KEY")

    for style in STYLE_PRESETS:
        logging.info(f"Send a request to Stability AI with style_preset:{style}")
        response = requests.post(
            f"https://api.stability.ai/v1/generation/{ENGINE_ID}/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            },
            # MEMO(shotarok) (576, 960) is the minimum size of (width, height) satisfied with
            # * InkyFrame 7.3"'s aspect ratio: 3 x 5
            # * Stability AI API's minimum area size
            json={
                "text_prompts": [
                    {
                        "text": prompt
                    }
                ],
                "cfg_scale": 7,
                "clip_guidance_preset": "FAST_BLUE",
                "height": 576,
                "width": 960,
                "samples": 1,
                "steps": 25,
                "style_preset": style,
            },
        )
        logging.info("Get a response from Stability AI")

        if response.status_code != 200:
            raise Exception("Non-200 response: " + str(response.text))

        data = response.json()

        # bytes of a generated PNG image file
        raw = base64.b64decode(data["artifacts"][0]["base64"])

        # Open the PNG image file to save it as JPG
        img = Image.open(BytesIO(raw))
        img_rgb = img.convert('RGB')
        img_resized = img_rgb.resize((INKY_FRAME_7_3_WIDTH, INKY_FRAME_7_3_HEIGHT))

        img_path = str((Path(dest) / f"{style}.jpg").absolute())
        img_resized.save(img_path, 'JPEG')
        logging.info(f"Succeeded to save {img_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
