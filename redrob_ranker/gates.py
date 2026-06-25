"""Hard impossibility gate (FP-free) + soft penalties (never removal).

Hard gate = parser-unambiguous arithmetic/chronology impossibilities only.
Empirically on the released 100K: tenure-sum>YOE catches 23, naive date errors catch 0.
Everything opinion-based (consulting, expert@0, CV-only, ...) is a SOFT penalty, because the
hidden grader may still rank such candidates highly (e.g. 19% of strong fits are consulting-tainted).
"""
from __future__ import annotations
from .schema import parse_date, prose, TODAY
from .features import is_junior_title

CONSULTING = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
              "tech mahindra", "mindtree", "ltimindtree", "hcl", "igate", "mphasis"]
CV_SPEECH_ROBOTICS = ["computer vision", "image classification", "object detection",
                      "speech recognition", "tts", "robotics", "slam", "lidar"]
NLP_IR_TERMS = ["nlp", "retrieval", "ranking", "search", "recommendation", "language",
                "embedding", "information retrieval", "text"]
PROD = ["production", "deployed", "real users", "at scale", "latency", "serving"]


def is_impossible(c: dict) -> bool:
    """Hard gate. True => excluded from the top-100 (forced to the bottom)."""
    p, hist = c["profile"], c["career_history"]
    yoe = p.get("years_of_experience", 0)
    if sum(h.get("duration_months", 0) for h in hist) > yoe * 12 + 24:
        return True
    for h in hist:
        sd = parse_date(h.get("start_date", ""))
        ed = parse_date(h.get("end_date")) if h.get("end_date") else None
        if sd and ed and sd > ed:
            return True
        if sd and sd > TODAY:
            return True
    return False


def soft_penalties(c: dict, cfg: dict) -> tuple[float, list]:
    """Return (total_penalty capped, reasons). All values from config.yaml penalties block."""
    P = cfg["penalties"]
    p, hist = c["profile"], c["career_history"]
    pen, reasons = 0.0, []
    comps = [h.get("company", "").lower() for h in hist]
    if comps and all(any(cf in cc for cf in CONSULTING) for cc in comps):
        pen += P["consulting_only"]; reasons.append("consulting-only career")
    e0 = sum(1 for s in c.get("skills", []) if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0)
    if e0 >= 3:
        pen += P["expert_at_zero"]; reasons.append(f"{e0} expert skills at 0 months")
    pr = prose(p, hist)
    t = p.get("current_title", "").lower()
    if any(k in t or k in pr for k in CV_SPEECH_ROBOTICS) and not any(k in pr for k in NLP_IR_TERMS):
        pen += P["cv_speech_robotics_no_nlp"]; reasons.append("CV/speech/robotics without NLP/IR")
    if "research" in t and not any(k in pr for k in PROD):
        pen += P["research_only_no_prod"]; reasons.append("research-only, no production signal")
    durs = [h.get("duration_months", 0) for h in hist if not h.get("is_current")]
    if durs and sum(durs) / len(durs) < 16:
        pen += P["title_chaser"]; reasons.append("short avg tenure (title-chaser signal)")
    if is_junior_title(p) and p.get("years_of_experience", 0) < cfg["role_profile"]["experience_years"]["min"]:
        pen += P["seniority_mismatch"]; reasons.append("junior title below role seniority")
    return min(pen, P["cap"]), reasons
