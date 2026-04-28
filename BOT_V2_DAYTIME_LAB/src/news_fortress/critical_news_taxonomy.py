
CRITICAL_USD_FAMILIES = [
    'nfp', 'non-farm payrolls', 'payrolls', 'unemployment rate',
    'cpi', 'core cpi', 'inflation', 'pce', 'core pce',
    'fomc', 'fomc statement', 'fomc minutes', 'fed rate decision',
    'federal funds rate', 'fed chair speech', 'powell speech',
    'retail sales', 'gdp', 'advance gdp', 'ism manufacturing',
    'ism services', 'pmi', 'jobless claims'
]

CRITICAL_EUR_FAMILIES = [
    'ecb rate decision', 'ecb interest rate', 'ecb monetary policy',
    'ecb press conference', 'lagarde speech', 'ecb president speech',
    'eurozone cpi', 'core cpi eur', 'german cpi', 'ecb minutes'
]

AMBIGUOUS_KEYWORDS = [
    'speech', 'testimony', 'press conference', 'statement',
    'emergency', 'unscheduled', 'crisis', 'liquidity', 'war'
]

ULTRA_CRITICAL_FAMILIES = [
    'fomc', 'fed rate decision', 'powell', 'nfp', 'cpi',
    'ecb rate decision', 'ecb press conference', 'lagarde'
]

def classify_event(event_title, currency):
    title = str(event_title).lower()
    curr = str(currency).upper()
    
    is_critical = False
    family = 'unknown'
    
    if curr == 'USD':
        for f in CRITICAL_USD_FAMILIES:
            if f in title:
                is_critical = True
                family = f
                break
    elif curr == 'EUR':
        for f in CRITICAL_EUR_FAMILIES:
            if f in title:
                is_critical = True
                family = f
                break
                
    # Keyword check for ambiguity
    is_ambiguous = any(k in title for k in AMBIGUOUS_KEYWORDS)
    
    is_ultra = any(u in title for u in ULTRA_CRITICAL_FAMILIES)
    
    return {
        "is_critical": is_critical,
        "is_ultra": is_ultra,
        "is_ambiguous": is_ambiguous,
        "family": family
    }
