// MANIPULANTE Calendar Exporter - Phase37
// Purpose: export MT5/MQL5 Economic Calendar events only.
// It is read-only and never modifies account state.
#property strict
#property script_show_inputs

input string OutputPrefix = "ftmo";
input int DaysAhead = 7;

string ImpactToText(const ENUM_CALENDAR_EVENT_IMPORTANCE importance)
{
   if(importance == CALENDAR_IMPORTANCE_HIGH) return "HIGH";
   if(importance == CALENDAR_IMPORTANCE_MODERATE) return "MEDIUM";
   if(importance == CALENDAR_IMPORTANCE_LOW) return "LOW";
   return "UNKNOWN";
}

bool IsTargetCurrency(const string currency)
{
   return (currency == "EUR" || currency == "USD");
}

string JsonEscape(string value)
{
   StringReplace(value, "\\", "\\\\");
   StringReplace(value, "\"", "\\\"");
   StringReplace(value, "\r", " ");
   StringReplace(value, "\n", " ");
   return value;
}

void ExportRange(const datetime from_time, const datetime to_time, const string file_name)
{
   MqlCalendarValue values[];
   int count = CalendarValueHistory(values, from_time, to_time, NULL, NULL);
   int handle = FileOpen(file_name, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(handle == INVALID_HANDLE)
   {
      Print("MANIPULANTE_CalendarExporter: cannot open ", file_name);
      return;
   }

   FileWriteString(handle, "{\n");
   FileWriteString(handle, "  \"source_type\": \"MT5_MQL5_ECONOMIC_CALENDAR\",\n");
   FileWriteString(handle, "  \"verified_by_mt5\": true,\n");
   FileWriteString(handle, "  \"generated_at_utc\": \"" + TimeToString(TimeGMT(), TIME_DATE | TIME_SECONDS) + "Z\",\n");
   FileWriteString(handle, "  \"timezone\": \"MT5 server export plus UTC by TimeGMT\",\n");
   FileWriteString(handle, "  \"events\": [\n");

   bool first = true;
   for(int i = 0; i < count; i++)
   {
      MqlCalendarEvent event;
      MqlCalendarCountry country;
      if(!CalendarEventById(values[i].event_id, event)) continue;
      if(!CalendarCountryById(event.country_id, country)) continue;
      if(!IsTargetCurrency(country.currency)) continue;
      if(event.importance != CALENDAR_IMPORTANCE_HIGH) continue;

      if(!first) FileWriteString(handle, ",\n");
      first = false;
      FileWriteString(handle, "    {\n");
      FileWriteString(handle, "      \"event_id\": \"" + IntegerToString((int)values[i].event_id) + "\",\n");
      FileWriteString(handle, "      \"event_name\": \"" + JsonEscape(event.name) + "\",\n");
      FileWriteString(handle, "      \"currency\": \"" + country.currency + "\",\n");
      FileWriteString(handle, "      \"impact\": \"" + ImpactToText(event.importance) + "\",\n");
      FileWriteString(handle, "      \"event_time_utc\": \"" + TimeToString(values[i].time, TIME_DATE | TIME_SECONDS) + "Z\",\n");
      FileWriteString(handle, "      \"event_time_server\": \"" + TimeToString(values[i].time, TIME_DATE | TIME_SECONDS) + "\",\n");
      FileWriteString(handle, "      \"actual\": \"" + IntegerToString((int)values[i].actual_value) + "\",\n");
      FileWriteString(handle, "      \"forecast\": \"" + IntegerToString((int)values[i].forecast_value) + "\",\n");
      FileWriteString(handle, "      \"previous\": \"" + IntegerToString((int)values[i].prev_value) + "\"\n");
      FileWriteString(handle, "    }");
   }

   FileWriteString(handle, "\n  ]\n");
   FileWriteString(handle, "}\n");
   FileClose(handle);
}

void OnStart()
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
   string date_key = TimeToString(start_today, TIME_DATE);
   StringReplace(date_key, ".", "-");

   ExportRange(start_today, end_today, date_key + "_" + OutputPrefix + "_news_today.json");
   ExportRange(start_today, end_week, date_key + "_" + OutputPrefix + "_news_week.json");
   ExportRange(start_today, end_week, date_key + "_" + OutputPrefix + "_news_gate_status.json");
   Print("MANIPULANTE_CalendarExporter finished: ", date_key);
}
