import polars as pl

class EntityResolver:
    def __init__(self, registry_csv: str):
        self.df = pl.read_csv(registry_csv).with_columns([
            pl.col("ticker").str.to_uppercase(),
            pl.col("domain").str.to_lowercase()
        ])

    def by_domain(self, domain: str) -> str | None:
        domain = domain.lower()
        m = self.df.filter(pl.col("domain")==domain)
        return (m["ticker"][0] if len(m) else None)

    def by_name(self, name: str) -> str | None:
        name = name.lower()
        m = self.df.filter(
            (pl.col("company_name").str.to_lowercase()==name) |
            (pl.col("aliases").str.contains(name, literal=False))
        )
        return (m["ticker"][0] if len(m) else None)
