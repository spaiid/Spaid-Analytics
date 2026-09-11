import polars as pl

def add_forward_return(
    df: pl.DataFrame | pl.LazyFrame,
    horizon: int,
    *,
    date_col: str = "asof",
    entity_col: str = "entity",
    price_col: str = "close",
    sector_col: str | None = None,
    benchmark_entity: str | None = None,   # e.g., "SPY"
    sector_relative: bool = False,
    use_log_returns: bool = True,
    prefix: str | None = None,
) -> pl.LazyFrame:
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    base = prefix or f"fwd{horizon}"

    ret_expr = (
        (pl.col(price_col).shift(-horizon) / pl.col(price_col)).log()
        if use_log_returns
        else (pl.col(price_col).shift(-horizon) / pl.col(price_col) - 1.0)
    )

    lf_targets = (
        lf.sort([entity_col, date_col])
          .with_columns(
              pl.when(pl.col(price_col) > 0)
                .then(ret_expr)
                .otherwise(None)
                .alias(f"{base}_ret")
          )
    )

    if benchmark_entity:
        bench = (
            lf_targets
            .filter(pl.col(entity_col) == benchmark_entity)
            .select(date_col, pl.col(f"{base}_ret").alias(f"{base}_ret_mkt"))
        )
        lf_targets = (
            lf_targets
            .join(bench, on=date_col, how="left")
            .with_columns(
                (pl.col(f"{base}_ret") - pl.col(f"{base}_ret_mkt")).alias(f"{base}_ret_ex_mkt")
            )
            .drop(f"{base}_ret_mkt")
        )

    if sector_relative:
        if not sector_col:
            raise ValueError("sector_relative=True requires sector_col.")
        sec_mean = (
            lf_targets
            .group_by([date_col, sector_col])
            .agg(pl.col(f"{base}_ret").mean().alias(f"{base}_ret_sec_mean"))
        )
        lf_targets = (
            lf_targets
            .join(sec_mean, on=[date_col, sector_col], how="left")
            .with_columns(
                (pl.col(f"{base}_ret") - pl.col(f"{base}_ret_sec_mean")).alias(f"{base}_ret_ex_sec")
            )
            .drop(f"{base}_ret_sec_mean")
        )

    return lf_targets
