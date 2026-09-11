"""Provider adapters.

Each provider translates one external source into the canonical schema declared
in `spaid.storage.schema`. Nothing downstream knows which provider a number came
from except through the `source` column, which is what makes a provider
replaceable rather than load-bearing.
"""
