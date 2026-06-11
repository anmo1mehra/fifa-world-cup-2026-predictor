"""Canonical team-name mapping for the 48 qualified 2026 FIFA World Cup teams.

The same nation often appears under different names across our three datasets:
    - schedule.xlsx uses 3-letter FIFA codes (e.g. "USA", "KOR", "TUR")
    - fifa_ranking.csv uses FIFA's official long names ("USA", "Korea Republic", "Türkiye")
    - results.csv uses common English names ("United States", "South Korea", "Turkey")

We pick ONE canonical name per team (matching results.csv since it's the largest
dataset, ~49k rows) and translate the other sources into it.
"""
from __future__ import annotations

CODE_TO_CANONICAL: dict[str, str] = {
    "ALG": "Algeria",
    "ARG": "Argentina",
    "AUS": "Australia",
    "AUT": "Austria",
    "BEL": "Belgium",
    "BIH": "Bosnia and Herzegovina",
    "BRA": "Brazil",
    "CAN": "Canada",
    "CIV": "Ivory Coast",
    "COD": "DR Congo",
    "COL": "Colombia",
    "CPV": "Cape Verde",
    "CRO": "Croatia",
    "CUW": "Curaçao",
    "CZE": "Czech Republic",
    "ECU": "Ecuador",
    "EGY": "Egypt",
    "ENG": "England",
    "ESP": "Spain",
    "FRA": "France",
    "GER": "Germany",
    "GHA": "Ghana",
    "HAI": "Haiti",
    "IRN": "Iran",
    "IRQ": "Iraq",
    "JOR": "Jordan",
    "JPN": "Japan",
    "KOR": "South Korea",
    "KSA": "Saudi Arabia",
    "MAR": "Morocco",
    "MEX": "Mexico",
    "NED": "Netherlands",
    "NOR": "Norway",
    "NZL": "New Zealand",
    "PAN": "Panama",
    "PAR": "Paraguay",
    "POR": "Portugal",
    "QAT": "Qatar",
    "RSA": "South Africa",
    "SCO": "Scotland",
    "SEN": "Senegal",
    "SUI": "Switzerland",
    "SWE": "Sweden",
    "TUN": "Tunisia",
    "TUR": "Turkey",
    "URU": "Uruguay",
    "USA": "United States",
    "UZB": "Uzbekistan",
}

# Aliases that may appear in the rankings or results dataset -> canonical name.
# Any name not in here is assumed to already match the canonical form.
ALIASES: dict[str, str] = {
    # From fifa_ranking.csv
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Cabo Verde": "Cape Verde",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Aotearoa New Zealand": "New Zealand",
    "Republic of Ireland": "Ireland",
    "Brunei Darussalam": "Brunei",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Chinese Taipei": "Taiwan",
    "Hong Kong, China": "Hong Kong",
    # Common results.csv variants we want to fold together
    "United States of America": "United States",
}

CANONICAL_TO_CODE: dict[str, str] = {v: k for k, v in CODE_TO_CANONICAL.items()}
WC_TEAMS: list[str] = sorted(CODE_TO_CANONICAL.values())

# Map FIFA 3-letter code -> ISO 3166-1 alpha-2 (lowercase) for flag rendering.
# England and Scotland use flagcdn's subdivision codes ("gb-eng", "gb-sct"),
# which work for the PNG CDN but not for emoji flags (those use a fallback).
ISO_CODES: dict[str, str] = {
    "ALG": "dz", "ARG": "ar", "AUS": "au", "AUT": "at", "BEL": "be",
    "BIH": "ba", "BRA": "br", "CAN": "ca", "CIV": "ci", "COD": "cd",
    "COL": "co", "CPV": "cv", "CRO": "hr", "CUW": "cw", "CZE": "cz",
    "ECU": "ec", "EGY": "eg", "ENG": "gb-eng", "ESP": "es", "FRA": "fr",
    "GER": "de", "GHA": "gh", "HAI": "ht", "IRN": "ir", "IRQ": "iq",
    "JOR": "jo", "JPN": "jp", "KOR": "kr", "KSA": "sa", "MAR": "ma",
    "MEX": "mx", "NED": "nl", "NOR": "no", "NZL": "nz", "PAN": "pa",
    "PAR": "py", "POR": "pt", "QAT": "qa", "RSA": "za", "SCO": "gb-sct",
    "SEN": "sn", "SUI": "ch", "SWE": "se", "TUN": "tn", "TUR": "tr",
    "URU": "uy", "USA": "us", "UZB": "uz",
}

_REGIONAL_OFFSET = ord("\U0001F1E6") - ord("a")  # 'A' regional indicator base


def iso_code(team_name: str) -> str | None:
    """Return the (lowercase) ISO/flagcdn code for a canonical team name."""
    code = name_to_code(team_name)
    return ISO_CODES.get(code) if code else None


def flag_emoji(team_name: str) -> str:
    """Return a unicode flag emoji for a team. Falls back to '' for subdivisions
    (England, Scotland) which don't have standard regional-indicator emoji."""
    iso = iso_code(team_name)
    if not iso or len(iso) != 2 or "-" in iso:
        return ""
    return "".join(chr(ord(c) + _REGIONAL_OFFSET) for c in iso)


def flag_url(team_name: str, size: str = "40x30") -> str | None:
    """Return a flagcdn.com PNG URL for a team. Always works (incl. ENG/SCO)."""
    iso = iso_code(team_name)
    if not iso:
        return None
    return f"https://flagcdn.com/{size}/{iso}.png"


def with_flag(team_name: str) -> str:
    """'Argentina' -> '🇦🇷 Argentina' (emoji + name). Used for Plotly labels."""
    emoji = flag_emoji(team_name)
    return f"{emoji} {team_name}" if emoji else f"   {team_name}"


def canonical(name: str | None) -> str | None:
    """Normalize any team name into our canonical form."""
    if name is None:
        return None
    name = str(name).strip()
    return ALIASES.get(name, name)


def code_to_name(code: str) -> str:
    """Convert a FIFA 3-letter code to the canonical team name."""
    return CODE_TO_CANONICAL[code.strip().upper()]


def name_to_code(name: str) -> str | None:
    """Convert a canonical team name to its FIFA 3-letter code (or None)."""
    return CANONICAL_TO_CODE.get(canonical(name))


# 2026 group composition, derived from schedule.xlsx
GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Colombia", "Uzbekistan"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
