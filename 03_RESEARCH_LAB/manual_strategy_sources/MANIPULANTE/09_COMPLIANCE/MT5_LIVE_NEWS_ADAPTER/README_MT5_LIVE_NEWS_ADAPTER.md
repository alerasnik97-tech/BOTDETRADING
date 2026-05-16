# MT5 Live News Adapter

This adapter is the approved Phase37 path for Live News Fortress.

- Source: MT5/MQL5 Economic Calendar.
- Currencies: EUR and USD.
- Impact: HIGH only.
- Guard window: 30 minutes before and 30 minutes after.
- Missing cache: NO_TRADE.
- Stale cache: NO_TRADE.
- Unknown timezone: NO_TRADE.

The exporter is read-only. It must not use `OrderSend`, `CTrade`, `trade.Buy`,
`trade.Sell`, or account-changing functions.
