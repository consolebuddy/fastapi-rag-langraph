from typing import List, Dict

def budget_chunks(chunks: List[Dict], max_chars: int = 12000) -> List[Dict]:
    sel, total = [], 0
    for c in chunks:
        t = c.get("text", "")
        if total + len(t) <= max_chars:
            sel.append(c)
            total += len(t)
        else:
            break
    return sel