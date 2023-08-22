import random

import polars as pl


# See https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts
PARQUET_FILE_URL = "https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts/resolve/main/data/train.parquet"

FAVORITE_ARTISTS = [
    "Charles Angrand",
    "Van Gogh",
    "Jackson Pollock",
    "Francis Bacon",
]


def main():
    artist = random.choice(FAVORITE_ARTISTS)
    suffix = f", written by {artist}."

    df = pl.read_parquet(PARQUET_FILE_URL)
    print(df.sample(1)[0, "Prompt"], suffix)

if __name__ == "__main__":
    main()
