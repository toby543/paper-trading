"""Optional LLM-generated plain-English rationale for BUY trades.

This module never influences which trades happen -- momentum_52w_high,
cross_sectional_momentum and consolidation_breakout make every buy/sell
decision on their own, deterministically, exactly as before this existed.
All this does is take the numbers a strategy already computed for a BUY
that has ALREADY been executed and recorded, and ask Claude to turn them
into one readable sentence for the trade log. It runs strictly after the
fact, and the request contains no lever that could feed back into a
decision -- it is one-way narration.

Deliberately left out of the backtester. A backtest can place dozens of
buys per profile per run and exists to be re-run repeatedly while tuning a
strategy or reproducing a result -- attaching a paid, network-dependent,
non-deterministic API call to every one of those buys would make backtests
slow, costly, occasionally-different-each-run, and dependent on an API key
being present just to test strategy logic. Live trading places at most a
handful of buys per scan, so this only ever narrates trades an account is
actually holding.

Any failure -- no package installed, no API key, network error, rate
limit, timeout -- is caught and logged here, never raised. A rationale is
a nice-to-have annotation on a trade that has already executed; nothing
in this module may affect whether that trade happened.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You add a one-sentence, plain-English rationale to a stock trade "
    "already made by a rule-based momentum trading system. You do not "
    "decide, endorse, or second-guess the trade -- it already happened; "
    "you are only explaining, in plain language, the numbers the "
    "strategy computed. Respond with exactly one sentence, at most 30 "
    "words. No caveats, no disclaimers, no restating the ticker symbol "
    "or price, no hedging language, no use of the words 'I' or "
    "'recommend'."
)


def generate_buy_rationale(
    symbol: str,
    strategy_mode: str,
    structured_reason: str,
    model: str = "claude-opus-5",
    timeout_seconds: float = 8.0,
) -> str | None:
    """One plain-English sentence explaining an already-executed BUY, or
    None if generation isn't possible or fails for any reason.

    `structured_reason` is the same reason string already attached to the
    trade record (e.g. "momentum_52w_high score=105.4 3.2% off 52w-high,
    24.1% 252d return, volume 1.8x baseline") -- the only input, since it
    already carries every number the strategy used to decide.
    """
    try:
        import anthropic
    except ImportError:
        log.debug(
            "`anthropic` package not installed; skipping buy rationale for %s "
            "(pip install anthropic to enable intelligence.enabled)", symbol,
        )
        return None

    try:
        client = anthropic.Anthropic(timeout=timeout_seconds, max_retries=0)
        response = client.messages.create(
            model=model,
            max_tokens=100,
            # A one-sentence annotation on a trade already sitting in the
            # ledger doesn't warrant deep reasoning -- low effort keeps this
            # fast and cheap without disabling thinking outright (which has
            # its own failure modes on Opus-tier models; see the model
            # docs). Latency here can never delay a trade: it runs after
            # the buy is already recorded.
            output_config={"effort": "low"},
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    f"Strategy: {strategy_mode}\n"
                    f"Symbol: {symbol}\n"
                    f"Signal: {structured_reason}"
                ),
            }],
        )
    except Exception as exc:  # noqa: BLE001 - see module docstring: a rationale
        # is best-effort decoration on a trade that has already executed;
        # nothing here may ever propagate into the trading engine.
        log.warning("Buy rationale generation failed for %s (%s): %s", symbol, strategy_mode, exc)
        return None

    text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()
    return text or None
