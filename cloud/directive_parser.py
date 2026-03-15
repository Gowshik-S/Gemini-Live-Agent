import re
from typing import Tuple

def parse_directives(text: str) -> Tuple[str, dict]:
    """Parse OpenClaw-style inline directives from text.
    
    Example: 'Open chrome /model:pro /thinking:high'
    Returns: ('Open chrome', {'model': 'pro', 'thinking': 'high'})
    """
    directives = {}
    
    # Match /key:value or /key
    pattern = r'/([a-zA-Z0-9_-]+)(?::([a-zA-Z0-9_-]+))?'
    
    def replace_match(match):
        key = match.group(1).lower()
        val = match.group(2)
        if val is None:
            val = "on"  # Default for flag-style e.g., /thinking
        else:
            val = val.lower()
        directives[key] = val
        return ""

    cleaned_text = re.sub(pattern, replace_match, text).strip()
    # Cleanup extra spaces left by removal
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text, directives
