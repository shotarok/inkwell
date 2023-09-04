import polars as pl


# See https://huggingface.co/datasets/poloclub/diffusiondb
PARQUET_FILE_URL = "https://huggingface.co/datasets/poloclub/diffusiondb/raw/main/metadata.parquet"


def main():
    df = pl.read_parquet(PARQUET_FILE_URL)
    print(df.sample(1)[0, "prompt"])

if __name__ == "__main__":
    main()
