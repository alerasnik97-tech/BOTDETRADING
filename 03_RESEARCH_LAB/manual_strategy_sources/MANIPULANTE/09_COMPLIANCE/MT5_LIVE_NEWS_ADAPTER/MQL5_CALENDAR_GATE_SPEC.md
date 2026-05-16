# MQL5 Calendar Gate Spec

Use official MQL5 Economic Calendar functions:

- `CalendarValueHistory`
- `CalendarEventById`
- `CalendarCountryById`
- `CalendarEventByCurrency`

Important: MQL5 calendar datetime values use trade server time, not local PC time. The adapter must write both server timestamp and UTC timestamp. If the conversion to UTC/NY is uncertain, output `NO_TRADE_TIMEZONE_ERROR`.

Allowed statuses:

- `ALLOW`
- `NO_TRADE_NEWS_WINDOW`
- `NO_TRADE_NEWS_SOURCE_UNAVAILABLE`
- `NO_TRADE_TIMEZONE_ERROR`
- `NO_TRADE_UNKNOWN_IMPACT`

The MQL5 adapter is watch-only and cache-only. It must not contain `OrderSend`, `CTrade`, `PositionOpen`, `trade.Buy` or `trade.Sell`.
