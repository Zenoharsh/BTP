import re

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def normalize_text(s):
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s

def anls_score(pred, answers, threshold=0.5):
    pred = normalize_text(pred)
    if not pred:
        return 0.0
    
    max_score = 0.0
    for ans in answers:
        ans = normalize_text(str(ans))
        dist = levenshtein(pred, ans)
        max_len = max(len(pred), len(ans))
        if max_len == 0:
            score = 1.0
        else:
            score = 1.0 - (dist / max_len)
            
        if score >= threshold:
            max_score = max(max_score, score)
    return max_score

def _clean_short(x):
    """lowercase, strip whitespace, surrounding quotes and a trailing period ("True." -> "true")."""
    x = str(x).strip().lower().strip('"\'').strip()
    return x[:-1].strip() if x.endswith(".") else x

def relaxed_acc_score(pred, answers):
    pred = _clean_short(pred)
    
    def parse_num(x):
        x = x.replace("%", "").replace(",", "").strip()
        try:
            return float(x)
        except ValueError:
            return None

    pred_num = parse_num(pred)
    
    for ans in answers:
        ans = _clean_short(ans)
        if pred == ans:
            return 1.0
            
        ans_num = parse_num(ans)
        if pred_num is not None and ans_num is not None:
            if ans_num == 0:
                if pred_num == 0:
                    return 1.0
            else:
                if abs(pred_num - ans_num) / abs(ans_num) <= 0.05:
                    return 1.0
    return 0.0

def boolean_acc_score(pred, answers):
    pred = _clean_short(pred)
    # Map pred
    if pred in ["true", "yes"]:
        p_val = "true"
    elif pred in ["false", "no"]:
        p_val = "false"
    else:
        p_val = None
        
    for ans in answers:
        ans = str(ans).strip().lower()
        if ans in ["true", "yes"]:
            a_val = "true"
        elif ans in ["false", "no"]:
            a_val = "false"
        else:
            a_val = ans # fallback
            
        if p_val is not None and p_val == a_val:
            return 1.0
    return 0.0

def exact_match(pred, answers):
    pred = normalize_text(pred)
    for ans in answers:
        if pred == normalize_text(str(ans)):
            return 1.0
    return 0.0

def compute_metric(task, pred, answers):
    if task == "document_ocr":
        return anls_score(pred, answers)
    elif task == "chart_qa":
        return relaxed_acc_score(pred, answers)
    elif task == "spatial_reasoning":
        return boolean_acc_score(pred, answers)
    else:
        return exact_match(pred, answers)
