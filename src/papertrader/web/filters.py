"""Template filters for the dashboard."""
from __future__ import annotations

from jinja2 import pass_context


@pass_context
def indian_currency(ctx, value) -> str:
    """Format a number with Indian digit grouping (lakhs/crores), e.g.
    100000 -> "1,00,000.00", matching the en-IN formatting the dashboard's
    client-side JS already uses everywhere else.

    Context-aware (hence pass_context) so a book denominated in a
    non-INR currency gets plain thousands grouping instead -- 100000 ->
    "100,000.00". Reading `quote_currency` off the render context keeps
    every existing `{{ x|inr }}` call site unchanged; the JS half picks
    the matching locale via NUM_LOCALE."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    negative = value < 0
    value = abs(value)
    int_part, dec_part = f"{value:.2f}".split(".")

    if str(ctx.get("quote_currency", "INR")).upper() != "INR":
        grouped = f"{int(int_part):,}"
    elif len(int_part) <= 3:
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
