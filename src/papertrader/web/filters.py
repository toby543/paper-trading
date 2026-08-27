"""Template filters for the dashboard."""
from __future__ import annotations


def indian_currency(value) -> str:
    """Format a number with Indian digit grouping (lakhs/crores), e.g.
    100000 -> "1,00,000.00", matching the en-IN formatting the dashboard's
    client-side JS already uses everywhere else."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    negative = value < 0
    value = abs(value)
    int_part, dec_part = f"{value:.2f}".split(".")

    if len(int_part) <= 3:
        grouped = int_part
    else:
        last3 = int_part[-3:]
        rest = int_part[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        grouped = ",".join(groups) + "," + last3

    return ("-" if negative else "") + f"{grouped}.{dec_part}"
