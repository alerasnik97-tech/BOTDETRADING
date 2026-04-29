//+------------------------------------------------------------------+
//|                                MANIPULANTE_CalendarBootstrapEA   |
//|                                  Copyright 2026, Antigravity Bot |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Antigravity Bot"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("MANIPULANTE BOOTSTRAP EA: Iniciando...");
   
   // Ejecutar la exportación
   ExportAll();
   
   Print("MANIPULANTE BOOTSTRAP EA: Exportación completada. Autoremoviendo...");
   
   // Autoremover el EA
   ExpertRemove();
   return(INIT_SUCCEEDED);
}

void ExportAll()
{
   datetime now_utc_time = TimeGMT();
   MqlDateTime dt;
   TimeToStruct(now_utc_time, dt);
   dt.hour = 0; dt.min = 0; dt.sec = 0;
   datetime today_start = StructToTime(dt);
   
   int dow = dt.day_of_week;
   int offset = (dow == 0) ? 6 : dow - 1;
   datetime week_start = today_start - (offset * 86400);
   datetime week_end = week_start + (7 * 86400) - 1;
   
   datetime today_end = today_start + 86400 - 1;
   
   ExportToFiles(week_start, week_end, "week");
   ExportToFiles(today_start, today_end, "today");
}

void ExportToFiles(datetime start_time, datetime end_time, string label)
{
   MqlCalendarValue values[];
   int count = CalendarValueHistory(values, start_time, end_time);
   
   if(count < 0)
   {
      Print("MANIPULANTE BOOTSTRAP EA: Error al obtener calendario. Código: ", GetLastError());
      return;
   }
   
   string folder = "MANIPULANTE\\";
   string filename_csv = folder + "ftmo_news_" + label + ".csv";
   string filename_json = folder + "ftmo_news_" + label + ".json";
   
   int handle_csv = FileOpen(filename_csv, FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_SHARE_READ, ',');
   int handle_json = FileOpen(filename_json, FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_SHARE_READ);
   
   if(handle_csv == INVALID_HANDLE || handle_json == INVALID_HANDLE)
   {
      if(handle_csv != INVALID_HANDLE) FileClose(handle_csv);
      if(handle_json != INVALID_HANDLE) FileClose(handle_json);
      return;
   }
   
   FileWrite(handle_csv, "event_name", "currency", "country", "importance", "time_utc", "source", "generated_at_utc");
   
   string generated_at = TimeToString(TimeGMT(), TIME_DATE|TIME_MINUTES|TIME_SECONDS);
   FileWriteString(handle_json, "{\n  \"source\": \"MT5_MQL5_CALENDAR_BOOTSTRAP_EA\",\n  \"generated_at_utc\": \"" + generated_at + "\",\n  \"events\": [\n");
   
   int exported_count = 0;
   for(int i=0; i<count; i++)
   {
      MqlCalendarEvent event;
      if(!CalendarEventById(values[i].event_id, event)) continue;
      
      MqlCalendarCountry country;
      if(!CalendarCountryById(event.country_id, country)) continue;
      
      if(country.currency != "EUR" && country.currency != "USD") continue;
      
      string imp_str = "LOW";
      if(event.importance == CALENDAR_IMPORTANCE_MODERATE) imp_str = "MODERATE";
      if(event.importance == CALENDAR_IMPORTANCE_HIGH) imp_str = "HIGH";
      
      datetime t_utc = values[i].time;
      string t_str = TimeToString(t_utc, TIME_DATE|TIME_MINUTES);
      
      FileWrite(handle_csv, event.name, country.currency, country.code, imp_str, t_str, "MT5_MQL5_CALENDAR_BOOTSTRAP_EA", generated_at);
      
      if(exported_count > 0) FileWriteString(handle_json, ",\n");
      FileWriteString(handle_json, "    {\n");
      FileWriteString(handle_json, "      \"name\": \"" + event.name + "\",\n");
      FileWriteString(handle_json, "      \"currency\": \"" + country.currency + "\",\n");
      FileWriteString(handle_json, "      \"country\": \"" + country.code + "\",\n");
      FileWriteString(handle_json, "      \"importance\": \"" + imp_str + "\",\n");
      FileWriteString(handle_json, "      \"time_utc\": \"" + t_str + "\"\n");
      FileWriteString(handle_json, "    }");
      
      exported_count++;
   }
   
   FileWriteString(handle_json, "\n  ]\n}");
   
   FileClose(handle_csv);
   FileClose(handle_json);
   
   if(label == "today")
   {
      int handle_gate = FileOpen(folder + "ftmo_news_gate_status.json", FILE_WRITE|FILE_TXT|FILE_ANSI);
      if(handle_gate != INVALID_HANDLE)
      {
         FileWriteString(handle_gate, "{\n  \"status\": \"BOOTSTRAP_DONE\",\n  \"last_update_utc\": \"" + generated_at + "\",\n  \"today_count\": " + (string)exported_count + "\n}");
         FileClose(handle_gate);
      }
   }
   
   Print("MANIPULANTE BOOTSTRAP EA: Exportados ", exported_count, " eventos para ", label);
}
