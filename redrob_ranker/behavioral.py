"""Behavioral availability — mild, normalized, bounded.

The 23 signals are near-random in this dataset (response-rate mean 0.44, inactivity maxes at
269 days), so they get a SMALL weight. A strong effect applies ONLY to the JD's explicitly-named
"not actually available" profile.
"""
from __future__ import annotations
from .schema import parse_date, TODAY


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def availability_norm(rs: dict) -> float:
    """Availability in [0,1] from response rate, recency, open-to-work, interview completion."""
    resp = rs.get("recruiter_response_rate", 0.5)
    la = parse_date(rs.get("last_active_date", ""))
    days = (TODAY - la).days if la else 120
    recency = _clip(1 - days / 270.0)
    openw = 1.0 if rs.get("open_to_work_flag") else 0.0
    icr = rs.get("interview_completion_rate", 0.5)
    return _clip(0.4 * resp + 0.3 * recency + 0.2 * openw + 0.1 * icr)


def availability_multiplier(rs: dict, floor: float = 0.78, reach: float = 0.40) -> float:
    """Graded, bounded availability penalty (replaces the old binary 0.5x cliff).

    Candidates with availability >= `reach` are untouched (multiplier 1.0). Below that, the
    multiplier scales SMOOTHLY down toward `floor` as availability -> 0. This 'down-weights
    appropriately' (per the JD) without burying a top technical match: the worst case loses
    (1-floor), not half its score. No cliff -> lower variance, fully explainable.
    """
    a = availability_norm(rs)
    unavail = max(0.0, (reach - a) / reach) if reach > 0 else 0.0   # 0 above reach, ->1 as a->0
    return 1.0 - (1.0 - floor) * unavail


def context_multiplier(profile: dict, rs: dict, cfg: dict) -> float:
    """Bounded location + notice-period fit, straight from the JD's location/comp/notice section.

    The role is a Pune/Noida hybrid in India, no visa sponsorship, short notice preferred. We also
    factor in identity verification (a verified profile is a more reachable, trustworthy lead). The
    effect is modest and capped: it lifts in-region, relocate-willing, short-notice, verified
    candidates and eases down clear mismatches. It never zeroes anyone.
    """
    cf = cfg.get("context_fit")
    if not cf or not cf.get("enabled", True):
        return 1.0
    loc = (profile.get("location", "") or "").lower()
    country = (profile.get("country", "") or "").lower()
    relocate = bool(rs.get("willing_to_relocate"))
    in_india = country == "india"

    if in_india and any(c in loc for c in cf["preferred_cities"]):
        loc_f = cf["loc_preferred"]
    elif in_india and (any(c in loc for c in cf["ok_india_cities"]) or relocate):
        loc_f = cf["loc_ok_or_relocate"]
    elif in_india:
        loc_f = cf["loc_other_india"]
    elif relocate:
        loc_f = cf["loc_foreign_relocate"]
    else:
        loc_f = cf["loc_foreign_norelocate"]

    n = rs.get("notice_period_days", 60)
    if n <= 30:
        notice_f = cf["notice_30"]
    elif n <= 60:
        notice_f = cf["notice_60"]
    elif n <= 90:
        notice_f = cf["notice_90"]
    else:
        notice_f = cf["notice_long"]

    # Identity verification: a verified, reachable profile is a stronger lead than an unverified one.
    # Bounded: a fully-unverified profile loses at most `verification_weight`; fully verified loses nothing.
    vw = cf.get("verification_weight", 0.0)
    verified = sum([bool(rs.get("verified_email")), bool(rs.get("verified_phone")),
                    bool(rs.get("linkedin_connected"))]) / 3.0
    ver_f = 1.0 - vw * (1.0 - verified)

    return loc_f * notice_f * ver_f


def is_extreme_unavailable(rs: dict) -> bool:
    """JD's explicit 'not actually available' profile — kept for reporting/audit only."""
    la = parse_date(rs.get("last_active_date", ""))
    days = (TODAY - la).days if la else 0
    return days >= 170 and rs.get("recruiter_response_rate", 1) <= 0.1 and not rs.get("open_to_work_flag")
