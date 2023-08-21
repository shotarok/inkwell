import polars as pl


# See https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts
PARQUET_FILE_URL = "https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts/resolve/main/data/train.parquet"


def main():
    df = pl.read_parquet(PARQUET_FILE_URL)
    print(df.sample(1)[0, "Prompt"])

if __name__ == "__main__":
    main()
