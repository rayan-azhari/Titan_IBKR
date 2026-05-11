"""Daily portfolio summary — passive Slack/Telegram notifier.

Adds a once-per-day rollup of account state, open positions, per-strategy
equity, and halt status. Doesn't trade. Subscribes to AUD/JPY H1 bars
(already in the champion portfolio's instrument list) as a clock-tick
source, then checks at each H1 close whether it's time to send today's
summary.
"""
