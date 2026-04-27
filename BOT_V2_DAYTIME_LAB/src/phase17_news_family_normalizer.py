
import re

class NewsFamilyNormalizer:
    def __init__(self):
        # Canonical mappings using regex
        self.mappings = {
            "CPI": [
                r"cpi m/m", r"cpi y/y", r"core cpi m/m", r"core cpi y/y",
                r"consumer price index"
            ],
            "NFP": [
                r"non-farm employment change", r"non-farm payrolls",
                r"adp non-farm employment change"
            ],
            "ECB": [
                r"main refinancing rate", r"ecb press conference",
                r"monetary policy statement", r"ecb monetary policy statement"
            ],
            "FOMC": [
                r"fomc statement", r"fomc press conference", 
                r"federal funds rate", r"fomc meeting minutes"
            ],
            "ISM": [
                r"ism manufacturing pmi", r"ism services pmi", r"ism non-manufacturing pmi"
            ],
            "GDP": [
                r"advance gdp q/q", r"prelim gdp q/q", r"final gdp q/q", r"gdp q/q"
            ],
            "RETAIL": [
                r"core retail sales m/m", r"retail sales m/m"
            ],
            "JOBLESS": [
                r"unemployment claims"
            ]
        }
        
        self.allowed_families = ["CPI", "NFP", "ECB"]
        self.rejected_families = ["FOMC", "ISM", "GDP"]
        self.watchlist_families = ["RETAIL", "JOBLESS"]

    def normalize(self, event_name):
        event_name = event_name.lower().strip()
        for family, patterns in self.mappings.items():
            for pattern in patterns:
                if re.search(pattern, event_name):
                    return family
        return "UNKNOWN"

    def is_allowed(self, family):
        return family in self.allowed_families

    def classify_event(self, event_row):
        """
        event_row: dict or Series with 'event_name_normalized', 'impact_level', 'currency'
        """
        raw_name = event_row.get('event_name_normalized', '')
        impact = event_row.get('impact_level', '').upper()
        currency = event_row.get('currency', '').upper()
        
        if impact != "HIGH":
            return "REJECTED_LOW_IMPACT", None
            
        if currency not in ["USD", "EUR"]:
            return "REJECTED_WRONG_CURRENCY", None
            
        family = self.normalize(raw_name)
        
        if family == "UNKNOWN":
            return "REJECTED_UNKNOWN_FAMILY", None
            
        if self.is_allowed(family):
            return "ALLOWED", family
        elif family in self.rejected_families:
            return "REJECTED_PROHIBITED_FAMILY", family
        else:
            return "WATCHLIST", family

if __name__ == "__main__":
    # Small test
    norm = NewsFamilyNormalizer()
    tests = [
        "Core CPI m/m",
        "Non-Farm Employment Change",
        "FOMC Statement",
        "ISM Manufacturing PMI",
        "Main Refinancing Rate",
        "Some random event"
    ]
    for t in tests:
        print(f"'{t}' -> {norm.normalize(t)}")
