#property strict

input int ExportEverySeconds = 600;
input int DaysAhead = 7;

string JsonEscape(string value)
{
   StringReplace(value, "\\", "\\\\");
   StringReplace(value, "\"", "\\\"");
   StringReplace(value, "\r", " ");
   StringReplace(value, "\n", " ");
   return value;
}

string ImportanceText(const ENUM_CALENDAR_EVENT_IMPORTANCE importance)
{
   if(importance == CALENDAR_IMPORTANCE_HIGH) return "HIGH";
   if(importance == CALENDAR_IMPORTANCE_MODERATE) return "MEDIUM";
   if(importance == CALENDAR_IMPORTANCE_LOW) return "LOW";
   return "UNKNOWN";
}

bool TargetCurrency(const string currency)
{
   return (currency == "EUR" || currency == "USD");
}

string UtcText(datetime value)
{
   return TimeToString(value, TIME_DATE | TIME_SECONDS) + "Z";
}

void ExportRange(datetime from_time, datetime to_time, string file_name)
{
   MqlCalendarValue values[];
   int count = CalendarValueHistory(values, from_time, to_time, NULL, NULL);
   int handle = FileOpen("MANIPULANTE\\" + file_name, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(handle == INVALID_HANDLE)
   {
      Print("MANIPULANTE CalendarBridge cannot open ", file_name);
      return;
   }
   FileWriteString(handle, "{\n");
   FileWriteString(handle, "  \"source_type\": \"MT5_MQL5_CALENDAR\",\n");
   FileWriteString(handle, "  \"verified_by_mt5\": true,\n");
   FileWriteString(handle, "  \"generated_at_utc\": \"" + UtcText(TimeGMT()) + "\",\n");
   FileWriteString(handle, "  \"server_time\": \"" + TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS) + "\",\n");
   FileWriteString(handle, "  \"timezone_basis\": \"TimeGMT plus MT5 server TimeCurrent\",\n");
   FileWriteString(handle, "  \"events\": [\n");
   bool first = true;
   for(int i = 0; i < count; i++)
   {
      MqlCalendarEvent event;
      MqlCalendarCountry country;
      if(!CalendarEventById(values[i].event_id, event)) continue;
      if(!CalendarCountryById(event.country_id, country)) continue;
      if(!TargetCurrency(country.currency)) continue;
      if(event.importance != CALENDAR_IMPORTANCE_HIGH) continue;
      if(!first) FileWriteString(handle, ",\n");
      first = false;
      FileWriteString(handle, "    {\n");
      FileWriteString(handle, "      \"event_id\": \"" + IntegerToString((int)values[i].event_id) + "\",\n");
      FileWriteString(handle, "      \"event_name\": \"" + JsonEscape(event.name) + "\",\n");
      FileWriteString(handle, "      \"currency\": \"" + country.currency + "\",\n");
      FileWriteString(handle, "      \"impact\": \"" + ImportanceText(event.importance) + "\",\n");
      FileWriteString(handle, "      \"event_time_utc\": \"" + UtcText(values[i].time) + "\",\n");
      FileWriteString(handle, "      \"event_time_server\": \"" + TimeToString(values[i].time, TIME_DATE | TIME_SECONDS) + "\",\n");
      FileWriteString(handle, "      \"source\": \"MT5_MQL5_CALENDAR\"\n");
      FileWriteString(handle, "    }");
   }
   FileWriteString(handle, "\n  ]\n");
   FileWriteString(handle, "}\n");
   FileClose(handle);
}

void ExportNews()
{
   datetime now_utc = TimeGMT();
   MqlDateTime dt;
   TimeToStruct(now_utc, dt);
   dt.hour = 0;
   dt.min = 0;
   dt.sec = 0;
   datetime start_today = StructToTime(dt);
   datetime end_today = start_today + 86400;
   datetime end_week = start_today + (DaysAhead * 86400);
   ExportRange(start_today, end_today, "ftmo_news_today.json");
   ExportRange(start_today, end_week, "ftmo_news_week.json");
   ExportRange(start_today, end_week, "ftmo_news_gate_status.json");
}

int OnInit()
{
   EventSetTimer(MathMax(300, ExportEverySeconds));
   ExportNews();
   return INIT_SUCCEEDED;
}

void OnTimer()
{
   ExportNews();
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}
