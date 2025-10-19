import polars as pl

def load_company_dict() -> pl.DataFrame:
    return pl.read_csv("src/entity/dictionaries/companies.csv")

def resolve_ticker(df_norm: pl.DataFrame) -> pl.DataFrame:
    comp = load_company_dict()
    joined = df_norm.join(
        comp.select(["ticker","domain","brand","name"]),
        left_on="domain", right_on="domain", how="left"
    )
    return joined.with_columns([
        pl.when(pl.col("ticker").is_null()).then(None).otherwise(pl.col("ticker")).alias("ticker"),
        pl.when(pl.col("ticker").is_null()).then(0.0).otherwise(1.0).alias("confidence"),
    ])
