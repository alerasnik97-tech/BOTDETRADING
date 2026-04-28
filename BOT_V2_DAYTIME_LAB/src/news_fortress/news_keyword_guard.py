
from .critical_news_taxonomy import AMBIGUOUS_KEYWORDS, CRITICAL_USD_FAMILIES, CRITICAL_EUR_FAMILIES

class KeywordGuard:
    def __init__(self):
        pass

    def evaluate_title(self, title, currency):
        t = str(title).lower()
        c = str(currency).upper()
        
        # 1. Critical USD/EUR Keywords
        if c == 'USD':
            for k in CRITICAL_USD_FAMILIES:
                if k in t:
                    return True, f"CRITICAL_USD_KEYWORD: {k}"
        elif c == 'EUR':
            for k in CRITICAL_EUR_FAMILIES:
                if k in t:
                    return True, f"CRITICAL_EUR_KEYWORD: {k}"
                    
        # 2. Ambiguous Keywords
        for k in AMBIGUOUS_KEYWORDS:
            if k in t:
                # Ambiguous keywords block if currency is USD or EUR
                if c in ['USD', 'EUR']:
                    return True, f"AMBIGUOUS_CRITICAL_KEYWORD: {k}"
                
        return False, "OK"
