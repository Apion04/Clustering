"""Configuration and normalization constants for supplier clustering."""

import os
import json
from dataclasses import dataclass, field
from typing import Any, Set, List, Dict, FrozenSet

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is optional at runtime; real environment variables still work.
    pass

LEGAL_SUFFIXES: List[str] = [
    "inc", "incorporated", "ltd", "limited", "llc", "llp", "lp",
    "corp", "corporation", "gmbh", "kg", "gbr",
    "lc", "l.c.", "commv",
    "sas", "s.a.s.", "s.a.", "sa", "spa", "srl", "s.r.l.",
    "bv", "b.v.", "ag", "a.g.", "plc", "p.l.c.", "kft",
    "private limited", "pvt ltd", "pvt. ltd.",
    "sp. z o.o.", "sp.zo.o.", "sp. z o.o", "sp zoo",
    "sa de cv", "s.a. de c.v.", "oy", "ab", "as", "aps",
    "sl", "s.l.", "nv", "n.v.", "se", "co", "company",
    "holdings", "holding", "group", "grp",
]

STATUS_WORDS: Set[str] = {
    "deactivated", "deacivated", "inactive", "old", "blocked", "gesperrt",
    "gesperrte", "geschlossen", "closed", "do not use", "do-not-use", "obsolete",
    "legacy", "former", "duplicate", "dup", "test", "testing", "old vendor",
    "xxxxx", "xxxx", "xxx",
}

OPERATIONAL_PREFIX_TOKENS: Set[str] = {
    "avisor",
}

GENERIC_DOMAINS: Set[str] = {
    # Global free email / ISP email providers. Exact or subdomain matches are treated as weak.
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.uk", "yahoo.co.in", "yahoo.co.jp",
    "yahoo.fr", "yahoo.de", "yahoo.es", "yahoo.it", "ymail.com", "rocketmail.com",
    "hotmail.com", "hotmail.co.uk", "hotmail.fr", "hotmail.de", "hotmail.it", "hotmail.es",
    "outlook.com", "outlook.in", "live.com", "live.co.uk", "msn.com",
    "icloud.com", "me.com", "mac.com", "aol.com", "aim.com",
    "protonmail.com", "proton.me", "pm.me", "mail.com", "email.com", "usa.com", "consultant.com",
    "gmx.com", "gmx.net", "gmx.de", "web.de", "freenet.de", "t-online.de",
    "mail.ru", "inbox.ru", "list.ru", "bk.ru", "rambler.ru", "yandex.ru", "yandex.com", "ya.ru",
    "qq.com", "163.com", "126.com", "sina.com", "sohu.com", "yeah.net", "foxmail.com",
    "rediffmail.com", "indiatimes.com", "sify.com", "vsnl.com",
    "orange.fr", "free.fr", "wanadoo.fr", "laposte.net", "sfr.fr", "neuf.fr",
    "libero.it", "virgilio.it", "alice.it", "tin.it", "tiscali.it",
    "uol.com.br", "bol.com.br", "terra.com.br", "ig.com.br", "globo.com",
    "naver.com", "daum.net", "hanmail.net", "nate.com",
    "bigpond.com", "bigpond.net.au", "optusnet.com.au", "iiNet.net.au",
    "btinternet.com", "btopenworld.com", "sky.com", "talktalk.net", "virginmedia.com",
    "comcast.net", "verizon.net", "att.net", "bellsouth.net", "cox.net", "charter.net", "earthlink.net",
    "rogers.com", "shaw.ca", "sympatico.ca", "telus.net",
    "zoho.com", "hushmail.com",
    # Temporary/disposable email domains often seen in poor quality vendor data.
    "mailinator.com", "yopmail.com", "10minutemail.com", "guerrillamail.com", "tempmail.com",
    "temp-mail.org", "trashmail.com", "sharklasers.com", "getnada.com", "dispostable.com",
}

ADDRESS_ABBREVIATIONS: Dict[str, str] = {
    "st": "street", "str": "street", "str.": "street", "strasse": "street", "straße": "street",
    "ave": "avenue", "av": "avenue", "blvd": "boulevard", "bld": "boulevard",
    "rd": "road", "dr": "drive", "ln": "lane", "ct": "court", "cir": "circle",
    "pl": "place", "sq": "square", "pkwy": "parkway", "hwy": "highway",
    "ste": "suite", "fl": "floor", "bldg": "building", "apt": "apartment", "unit": "unit",
    "po box": "pobox", "p.o. box": "pobox", "nr": "number", "no": "number",
    "allee": "allee", "alle": "allee",
}

KNOWN_BRANDS: Set[str] = {
    "subway", "dq", "dairy queen", "popeyes", "shoppers drug mart", "esso",
    "shell", "petro canada", "co-op", "canadian tire", "iga", "sobeys", "fas gas",
    "apple auto glass", "dhl", "kantar", "ogilvy", "ricoh", "jones lang lasalle",
    "cbre", "iqvia", "danone", "nutricia", "eurofins", "thermo fisher",
    "thermofisher", "fisher clinical", "patheon", "sigma aldrich", "millipore",
    "merck", "icon", "pra", "molecularmd", "convergint", "icd security",
    "bell", "telus", "rogers", "rbc", "royal bank", "td", "scotiabank", "bmo",
    "pitney bowes", "alberta health services", "ahs", "university of",
    "research institute", "ppc", "vynova ppc", "rain carbon", "ruetgers", "rutgers",
    "potasse", "elektrophysik", "elektro physik", "naegele",
}

STORE_NUMBER_PATTERNS: List[str] = [
    r"\s*#\d+", r"\s*no\s*\d+", r"\s*nr\s*\d+", r"\s*store\s*\d+",
    r"\s*outlet\s*\d+", r"\s*\d{3,5}$",
]



INVALID_TAX_VALUES: Set[str] = {
    "", "na", "n/a", "none", "null", "nan", "blank", "unknown", "notavailable", "notprovided",
    "notapplicable", "notspecified", "subjecttotax", "subjecttotaxes", "taxexempt", "exempt",
    "notregistered", "pending", "tbd", "-", "--", "000000", "0000000", "00000000",
    "000000000", "999999", "9999999", "99999999", "999999999",
}

COMMON_FIRST_NAMES: Set[str] = {
    "aaron", "abdul", "abigail", "achim", "adam", "adolf", "adrian", "ahmad", "albert", "alfred",
    "aline", "alice", "alicia", "alex", "alexander", "alexandra", "alexandre", "alexandru",
    "alessandro", "amit", "anita", "angela", "andrea", "andreas", "andre", "andree",
    "andrei", "andrew", "ann", "anna", "anne", "annette", "anja", "anke", "annika",
    "annegret", "antje", "anton", "antonia", "antoine", "armin", "arthur", "axel",
    "robert", "roberto", "peter", "michael", "john", "david", "thomas", "martin", "maria",
    "jose", "josef", "mohamed", "mohammad", "muhammad", "ahmed", "ali", "anna", "anne",
    "sandra", "susan", "suzanne", "paul", "mark", "marc", "frank", "stephan", "stefan",
    "benjamin", "benoit", "bernd", "bernhard", "birgit", "bruno", "carol", "carolina", "caroline",
    "carsten", "cedric", "chris", "christ", "christa", "christan", "christel", "christelle",
    "christen", "christian", "christiana", "christianah", "christiane", "christoph", "chrissy",
    "daniel", "daniela", "dieter", "dirk", "dominic", "dominik", "dominique", "doris",
    "edgar", "eduard", "elena", "elisabeth", "emmanuel", "erich", "erika", "ernst", "eva",
    "felix", "florian", "francesca", "frederic", "friedrich", "richard", "james", "william", "george",
    "charles", "patrick", "philippe", "jean", "pierre", "marie", "luis", "carlos", "juan",
    "georg", "giovanni", "gregor", "guido", "guillaume", "antonio", "marco", "rahul", "rohit",
    "raj", "ravi", "sunil", "sanjay", "hans", "heike", "heinrich", "herbert", "holger",
    "horst", "ingo", "ines", "jan", "joachim", "johannes", "julia", "julian", "karl",
    "katrin", "klaus", "kevin", "kristin", "laurent", "lars", "lena", "leon", "leopold", "linda", "lisa", "lukas",
    "manuel", "manuela", "mantas", "marcus", "maren", "margareta", "margit", "mario",
    "marko", "markus", "matthias", "maximilian", "michel", "nicolas", "nicole", "oliver", "olivier",
    "pascal", "ralf", "reiner", "rene", "roland", "sebastian", "simon", "sophia", "steffen",
    "thierry", "tim", "tobias", "vincent", "werner", "juergen", "jurgen", "wolfgang", "franz", "gerhard",
    "joerg", "jorg", "ulrich", "uwe", "helmut", "manfred", "rainer", "guenter", "gunter",
    "claudia", "monika", "petra", "sabine", "angelika", "karin", "renate", "brigitte",
    "franco", "francesco", "luca", "luigi", "massimo", "christine", "christina", "christin",
    "waldemar", "walter", "wilhelm",
    # Common first names plus high-frequency surnames used to prevent person
    # records from becoming company/root bridge tokens in supplier data.
    "cindy", "gustav", "ilona", "jana", "jill", "jonas", "kerstin", "nike", "philipp", "robin",
    "harald", "hermann", "herrmann", "hoermann", "kathrin", "leah", "mari",
    "martina", "melanie", "ruediger", "rüdiger", "sarah", "stefanie",
    "thum", "zimmer", "mueller", "muller", "schmidt", "fischer", "weber",
    "meyer", "schulz", "becker", "hoffmann", "schaefer", "richter",
    "baur", "bauer", "braun", "herrmann", "kraemer", "kramer", "krause",
    "lehmann", "koenig", "schneider", "wagner",
}

PERSON_TITLE_TOKENS: Set[str] = {
    "dr", "doctor", "prof", "professor", "ra", "mr", "mrs", "ms", "miss", "mx",
    "herr", "frau", "sir", "madam", "monsieur", "madame", "mme", "m",
    "dipl", "kfm", "diplkfm", "ing", "phd", "architekt", "rechtsanwalt",
}

LOCATION_ROOT_TOKENS: Set[str] = {
    # Location words can support a match through mapped city/country fields,
    # but they must never become root/family bridge tokens.
    "american", "amsterdam", "beijing", "berlin", "cologne", "darmstadt", "deutsche", "deutscher",
    "deutsches", "deutschland", "frankfurt", "german", "germany",
    "asia", "brussels", "canada", "china", "chinese", "duesseldorf", "dusseldorf",
    "europe", "european", "france", "french",
    "hamburg", "hong", "india", "jena", "jiangsu", "koeln", "koln", "kong", "london", "los", "angeles",
    "mannheim", "munich",
    "muenchen", "outremont", "paris", "singapore", "solna", "strasbourg",
    "hoechst",
    "hangzhou", "nanjing", "pacific", "shanghai", "tianjin", "toronto", "usa", "vancouver", "vienna", "wien", "wuhan",
    "europa", "qingdao", "shenyang",
    "changsha", "chengdu", "chongqing", "dalian", "fuzhou", "guiyang",
    "harbin", "hefei", "kunming", "lanzhou", "nanchang", "ningbo",
    "qinghai", "shijiazhuang", "shenzen", "suzhou", "wenzhou", "xian",
    "zhengzhou", "zhuhai",
    "aachen", "bielefeld", "bochum", "bonn", "dortmund", "edinburgh",
    "erfurt", "essen", "freiburg", "heidelberg", "karlsruhe", "kiel",
    "magdeburg", "mainz", "stuttgart", "tuebingen", "tubingen", "tübingen",
    "wiesbaden",
}

HOSPITALITY_TERMS: Set[str] = {
    "hotel", "hotels", "restaurant", "restaurants", "parkhotel", "gasthof", "guesthouse",
    "guest", "house", "inn", "resort", "motel", "cafe", "bar", "pub", "bistro",
    "hyatt", "marriott", "hilton", "radisson", "ibis", "novotel", "sheraton", "westin",
    "subway", "mcdonalds", "mcdonald", "burger", "king", "kfc", "popeyes", "pizza", "hut", "dominos",
}

GENERIC_ROOT_TOKENS: Set[str] = {
    # Generic words can support a match but must not be the main root/family bridge.
    "access", "open", "service", "services", "consulting", "consultants", "trading", "trade", "traders", "solution", "solutions",
    "avisor",
    "loesungen", "losungen", "systems", "technology", "technologies", "technologie", "technologien", "tech", "chemicals", "chemical", "chemistry", "engineering",
    "engineers", "logistics", "transport", "transportation", "shipping", "freight", "forwarding", "chain",
    "group", "global", "international", "industry", "industries", "industrial", "enterprise", "enterprises",
    "industria",
    "company", "corporation", "corp", "co", "office", "center", "centre", "general", "management", "national",
    "holding", "holdings", "partners", "associates", "agency", "agencies", "business", "commercial",
    "supplies", "supply", "supplier", "distribution", "distributors", "distributor", "wholesale", "retail",
    "import", "export", "imports", "exports", "manufacturing", "manufacturer", "manufacturers",
    "production", "point", "strategy", "strategies", "sales", "material", "materials", "performance",
    "standards", "analytical", "clinical", "diagnostic", "diagnostics", "instruments", "software",
    "advanced", "applied", "alpha", "arbeit", "arbeitgeberverband", "arbeitsmedizin", "arbeitsschutz",
    "alpha", "beta", "gamma", "green", "red", "blue", "akademie", "academy", "association", "autohaus",
    "bio", "biotech", "biotechnology", "biochem", "bioscience", "biosciences", "brand", "bund", "bundes", "bundesagentur", "bundesakademie", "bundesamt",
    "bundesanstalt", "bundesanzeiger", "bundesausschreibungsblatt", "bundesdruckerei",
    "bundesfachschule", "bundesgerichtshof", "bundesindustrieverband", "bundesinstitut",
    "chemi", "chemia", "chemial", "chemica", "chemicalia", "chemicon", "chemiefac",
    "chemieliva", "chemienord", "chemiepartner", "chemiewerk", "chemiewerke", "chemik",
    "chemie", "coaching", "college", "congress", "data", "development", "drug", "drugs", "energy", "events",
    "gas", "gebr", "gebrueder", "gebruder", "geo",
    "finanzamt", "marketing", "media", "publishing", "society", "technical", "training", "verlag",
    "medical", "health", "healthcare", "kreditverr", "kreditverrkto", "kto", "net", "optical",
    "pharm", "pharma", "pharmaceutical", "pharmaceuticals",
    "laboratory", "laboratories", "lab", "labs", "life", "research", "science", "scientific",
    "hospital", "hospitals", "clinic", "clinics", "sanitaetshaus", "sanitatshaus", "tourism", "tourismus",
    "university", "universities", "institut", "institute", "institutes",
    "construction", "builders", "building", "works", "workshop", "maintenance", "facility", "facilities",
    "support", "supports", "factory", "molecular",
    "food", "foods",
    "gastro",
    "hotel", "hotels", "restaurant", "restaurants", "parkhotel", "cafe", "bar", "resort", "motel",
    # Precision-review additions: these are common brand-like or industry words
    # in the real validation file and must not bridge suppliers by themselves.
    "sigma", "gemini", "jasmin", "atlas", "packaging", "automation", "partnership",
    "trucks", "electronics", "express", "community", "network", "cargo", "airport",
    "terminal", "association", "brand", "testing", "bv", "gmbh", "ltd", "limited",
    "inc", "co", "company",
    "private", "partner", "industrie", "industrien", "product", "products",
    "biological", "biologics", "medizin", "medizinische", "medicine", "medicines",
    "biologicals",
    "excellence", "exzellenz", "executive", "executives", "ingredients", "ingredient",
    "industrieller", "speciality", "specialty", "intermediate", "intermediates", "organics", "organic",
    "coating", "coatings", "compound", "compounds", "platform", "platforms",
    "networks", "analytics", "analyse", "analysis", "innovation", "innovations",
    "property", "properties", "asset", "assets", "surface", "surfaces",
    "composite", "composites", "nutrition", "nutritional", "diagnostic",
    "diagnostics", "electro", "elektronik", "communication", "communications",
    "chem",
    "consortium", "consortia", "regulatory", "regulation", "regulations", "reach",
    "task", "force", "taskforce", "taskforces", "working", "workinggroup", "committee", "council", "councils",
    "foundation", "fondation", "stiftung", "verein", "gastronomie",
    "messtechnik", "techn", "technische", "technik", "universitaet",
    "universitat", "universitätsklinikum", "universitaetsklinikum",
    "hochschule", "klinikum", "ingenieurbuero", "ingenieurburo",
    "verbindlichkeiten", "arbeitszeitberatung",
}

# Single-token roots in this set are too risky to create supplier brand/group
# identity clusters by themselves. They can still support stronger evidence
# such as tax, domain, exact address, or trusted phrase-level aliases.
SUPPLIER_IDENTITY_RISKY_SINGLE_TOKENS: Set[str] = {
    "access", "adm", "alfa", "alpha", "atlas", "berry", "delta", "dollar",
    "dominion", "dsm", "express", "gemini", "green", "jasmin", "meta",
    "next", "private", "partner", "red", "sigma", "virgin", "zimmer",
    "alphabet", "american", "bank", "banca", "banco", "china", "commercial",
    "deutsche", "gold", "golden", "india", "liberty", "metro", "national",
    "nippon", "orange", "phoenix", "popular", "shell", "silver", "total",
    "united", "white", "world", "insight", "insights", "springer",
}

AMBIGUOUS_REVIEW_CORES: Set[str] = {
    # These words occur as real supplier names but are broad enough that a
    # token-only match should be LLM/manual review, never deterministic 85/98.
    "apple", "insight", "insights", "springer",
}

PROTECTED_COMPOUND_IDENTITY_PHRASES: Set[str] = {
    # Do not collapse these to one parent token and bridge them to unrelated
    # families without explicit support.
    "air liquide", "air products", "axel springer", "bio springer",
    "eastman kodak", "sigma aldrich", "springer nature",
}

SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS: Set[str] = {
    # Explicitly common corporate brands that are otherwise common words.
    "3b", "3bl", "4titude", "abbott", "accenture", "airtec",
    "ajinomoto", "cognis", "cognizant", "computershare", "dhl",
    "eastman", "edi", "elrig", "fedex", "merck", "microsoft",
    "millipore", "oracle", "sap", "siemens",
}

TRUSTED_SUPPLIER_IDENTITY_CORES: Set[str] = {
    # Curated global supplier cores that should not be silently missed. These
    # still pass through generic/person/location guardrails and broad-group
    # scoring caps.
    "3b", "3bl", "4titude", "abbott", "accenture", "airtec",
    "ajinomoto", "cognis", "cognizant", "computershare", "dhl",
    "eastman", "edi", "elrig", "fedex", "merck", "microsoft",
    "millipore", "oracle", "sap", "siemens",
}

BROAD_GLOBAL_SUPPLIER_CORES: Set[str] = {
    # These are legitimate global supplier groups, but the validation data has
    # many divisions, legal entities, and unrelated address/country contexts.
    # Without tax/domain/address support, they should land in Review/<90 rather
    # than look like exact duplicate clusters.
    "abbott", "airtec", "ajinomoto", "cognis", "cognizant", "computershare",
    "eastman", "merck", "millipore", "siemens",
}

REGULATORY_REVIEW_TOKENS: Set[str] = {
    "reach", "task", "force", "taskforce", "consortium", "working", "group",
    "association", "foundation", "regulatory", "regulation", "committee",
    "council",
}

COMPANY_HINT_WORDS: Set[str] = {
    "gmbh", "ag", "kg", "inc", "ltd", "limited", "llc", "corp", "corporation", "sas", "sa",
    "srl", "spa", "bv", "ab", "as", "aps", "oy", "nv", "plc", "llp", "lp", "company", "co", "group", "holding", "holdings",
    "bank", "university", "hospital", "clinic", "institute", "systems", "technologies", "technology",
    "autohaus", "chemicals", "engineering", "logistics", "transport", "waagenservice", "service", "services",
    "coaching", "consulting", "training",
}

KNOWN_FAMILY_TOKEN_GROUPS: List[FrozenSet[str]] = [
    frozenset({"merck", "millipore", "emd", "sigma", "aldrich", "supelco"}),
    frozenset({"akzo", "nouryon"}),
    frozenset({"rain", "ruetgers", "rutgers"}),
]

KNOWN_DISTINCTIVE_FAMILY_ROOTS: Set[str] = {
    # Explicit reviewer-approved family/root token. Generic words like media,
    # publishing, verlag, and business remain non-bridge tokens.
    "weka",
}

KNOWN_ADDRESS_FAMILY_BRIDGE_GROUPS: List[FrozenSet[str]] = [
    # TURNUS shares the exact Kissing address with WEKA MEDIA entities and was
    # explicitly identified as a review-level WEKA family/address bridge.
    frozenset({"weka", "turnus"}),
]

KNOWN_RELATED_NAME_PAIRS: List[FrozenSet[str]] = [
    # Explicit reviewer-provided acquisition/rebrand/alias candidate. This is
    # only used with address/domain/tax support; it does not create broad blocks.
    frozenset({"service express", "top gun technology"}),
]

# Column name patterns for worldwide tax/legal registration identifiers.
# We do not validate every country's checksum in this engine. Instead, we normalize and use these IDs as matching signals.
TAX_COLUMN_PATTERNS: Set[str] = {
    "tax", "taxid", "tax_id", "tax number", "taxnumber", "tax no", "taxno",
    "vat", "vatid", "vat_id", "vat number", "vatnumber", "vat no", "vatno", "vat reg", "vatreg",
    "gst", "gstin", "gst no", "gst number", "pan", "tan", "cin",
    "ein", "tin", "itin", "ssn", "fein",
    "abn", "acn", "arbn", "tfn",
    "bn", "brn", "uen", "nzbn", "crn", "company registration", "registration", "reg no", "regno",
    "business registration", "business number", "company number", "corporate number",
    "siren", "siret", "tva", "ust", "ustid", "ust-id", "uid", "mwst",
    "nif", "nie", "cif", "niss", "nrt", "nifc",
    "rfc", "ruc", "rut", "nit", "cuit", "cuil", "cnpj", "cpf",
    "piva", "partita iva", "codice fiscale", "fiscal code", "iva",
    "btw", "kvk", "rsin",
    "npwp", "nib", "npwp no", "nppkp",
    "nip", "regon", "krs", "ico", "ičo", "dic", "dič", "ic dph", "ič dph",
    "orgnr", "org no", "org number", "cvr", "y-tunnus", "ytunnus",
    "trn", "qst", "hst", "pst", "sales tax",
}

DEFAULT_JSON_TAX_KEYS: List[str] = [
    "vatNumber", "taxNumber", "vatId", "taxId", "vatID", "taxID", "vat", "tax",
    "gstNumber", "gstin", "pan", "tan", "ein", "tin", "abn", "acn", "bn",
    "siren", "siret", "nif", "cif", "rfc", "ruc", "rut", "nit", "cuit", "cuil",
    "cnpj", "cpf", "trn", "registrationNumber", "businessRegistrationNumber",
    "companyRegistrationNumber", "legalRegistrationNumber", "taxRegistrationNumber",
]

DEFAULT_JSON_SECONDARY_NAME_KEYS: List[str] = [
    "familyName", "family_name", "parentName", "parent_name", "groupName",
    "group_name", "tradeName", "trade_name", "dba", "doingBusinessAs",
    "alternateName", "alternate_name", "legalName", "legal_name",
]

SUPPORT_FIELD_STRENGTHS: Set[str] = {
    "same_entity_id",
    "same_entity_name",
    "family_or_parent",
    "domain",
    "review_only",
}

DEFAULT_SUPPORT_FIELD_STRENGTHS: Dict[str, str] = {
    "family_name": "family_or_parent",
    "canonical_name": "family_or_parent",
    "parent_name": "family_or_parent",
    "normalized_supplier_name": "review_only",
    "json_secondary_name": "family_or_parent",
    "website": "domain",
    "domain": "domain",
    "email_domain": "domain",
    "orovendorid": "review_only",
    "companyentityid": "review_only",
    "tax_id": "same_entity_id",
}


def parse_support_field_strengths(raw: str = "") -> Dict[str, str]:
    """Parse optional support-field strength overrides.

    Accepted formats:
    - JSON object: {"OROVendorId": "same_entity_id"}
    - comma list: OROVendorId:same_entity_id,CompanyEntityId:review_only
    """
    out = dict(DEFAULT_SUPPORT_FIELD_STRENGTHS)
    if not raw:
        return out
    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        items = parsed.items()
    else:
        pairs = []
        for part in raw.split(","):
            if ":" not in part:
                continue
            k, v = part.split(":", 1)
            pairs.append((k, v))
        items = pairs
    for key, value in items:
        field_key = str(key or "").strip()
        strength = str(value or "").strip()
        if not field_key or strength not in SUPPORT_FIELD_STRENGTHS:
            continue
        out[field_key] = strength
        out[field_key.lower()] = strength
    return out

@dataclass
class ClusteringConfig:
    auto_cluster_threshold: float = 0.90
    review_threshold: float = 0.50
    name_prefix_length: int = 4
    max_candidates_per_block: int = 10000
    max_total_candidate_pairs: int = 1000000
    max_weak_block_size: int = 300
    max_tax_block_size: int = 5000
    max_tax_loose_block_size: int = 1000
    max_domain_block_size: int = 2000
    max_address_block_size: int = 1000
    exact_tax_star_threshold: int = 100
    max_exact_tax_only_block_size: int = 10
    max_exact_tax_distinct_roots: int = 3
    max_weak_review_cluster_size: int = 50
    max_low_confidence_cluster_size: int = 6
    max_known_family_cluster_size: int = 25
    max_known_brand_family_block_size: int = 300
    rare_token_max_document_fraction: float = 0.03
    rare_token_min_document_frequency: int = 1
    fuzzy_name_threshold_strong: float = 0.92
    fuzzy_name_threshold_medium: float = 0.85
    fuzzy_name_threshold_weak: float = 0.70
    address_similarity_threshold: float = 0.85
    max_companies_per_address: int = 10
    output_dir: str = "./output"
    ai_review_enabled: bool = False
    ai_min_match_pct: float = 45.0
    ai_max_match_pct: float = 80.0
    ai_provider: str = "openai_compatible"  # openai_compatible, claude
    ai_api_key: str = ""
    ai_base_url: str = "https://api.openai.com/v1"
    ai_model: str = "gpt-5.5"
    openai_model: str = "gpt-5.5"
    ai_timeout_seconds: int = 45
    ai_max_calls: int = 50
    ai_cache_enabled: bool = True
    ai_cache_path: str = ".cache/llm_review_cache.json"
    llm_can_auto_cluster: bool = False
    ai_uncertain_cluster_enabled: bool = True
    ai_uncertain_match_pct: float = 68.0
    # Production-safe default: unresolved 70-score LLM candidates must not be
    # silently returned to non-technical users. They are built for backend LLM
    # review, then either resolved by decisions or removed from the clean final
    # output while the run is marked incomplete.
    allow_unresolved_llm_candidates_in_final_output: bool = False
    unresolved_llm_candidate_mode: str = "exception"  # exception or blank
    expose_unresolved_llm_candidates: bool = False
    llm_group_decisions: List[Dict[str, Any]] = field(default_factory=list)
    llm_enabled: bool = False
    llm_execution_mode: str = "disabled"  # disabled, mock, live, batch
    llm_send_scope: str = "all_review_candidates"
    max_llm_groups_per_job: int = 0
    max_rows_per_llm_group: int = 60
    max_tokens_per_llm_group: int = 6000
    max_total_llm_cost_per_job: float = 0.0
    llm_timeout_seconds: int = 60
    llm_retry_count: int = 2
    override_llm_can_modify_98: bool = False
    openai_input_cost_per_1m_tokens: float = 0.0
    openai_output_cost_per_1m_tokens: float = 0.0
    allow_unknown_llm_cost: bool = False
    embeddings_enabled: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_threshold: float = 0.85
    allow_parent_family_tax_conflicts: bool = True
    enable_family_bridge: bool = True
    enable_acronym_bridge: bool = True
    known_brand_families_file: str = "data/known_brand_families.csv"
    known_brand_family_default_confidence: float = 76.0
    legal_keywords_file: str = "data/legal_keywords.csv"
    generic_non_bridge_file: str = "data/generic_non_bridge_keywords.csv"
    support_field_strengths: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SUPPORT_FIELD_STRENGTHS))

    @classmethod
    def from_env(cls) -> "ClusteringConfig":
        provider = os.getenv("LLM_PROVIDER") or os.getenv("AI_PROVIDER", "openai_compatible")
        provider_normalized = provider.lower()
        gemini_provider = provider_normalized in {"gemini", "google", "google_gemini"}
        model = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or os.getenv("AI_MODEL")
        if not model:
            model = "gemini-2.5-flash" if gemini_provider else "gpt-5.5"
        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or ""
        )
        base_url = os.getenv("AI_BASE_URL")
        if not base_url:
            base_url = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta") if gemini_provider else "https://api.openai.com/v1"
        return cls(
            auto_cluster_threshold=float(os.getenv("AUTO_CLUSTER_THRESHOLD", "0.90")),
            review_threshold=float(os.getenv("REVIEW_THRESHOLD", "0.50")),
            ai_review_enabled=os.getenv("AI_REVIEW_ENABLED", "false").lower() == "true",
            ai_provider=provider,
            ai_api_key=api_key,
            ai_base_url=base_url,
            ai_model=model,
            openai_model=model,
            llm_enabled=os.getenv("LLM_ENABLED", os.getenv("AI_REVIEW_ENABLED", "false")).lower() == "true",
            llm_execution_mode=_normalize_llm_execution_mode(os.getenv("LLM_EXECUTION_MODE", "disabled")),
            llm_send_scope=os.getenv("LLM_SEND_SCOPE", "all_review_candidates"),
            max_llm_groups_per_job=int(os.getenv("MAX_LLM_GROUPS_PER_JOB", "0") or "0"),
            max_rows_per_llm_group=int(os.getenv("MAX_ROWS_PER_LLM_GROUP", "60") or "60"),
            max_tokens_per_llm_group=int(os.getenv("MAX_TOKENS_PER_LLM_GROUP", "6000") or "6000"),
            max_total_llm_cost_per_job=float(os.getenv("MAX_TOTAL_LLM_COST_PER_JOB", "0") or "0"),
            llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", os.getenv("AI_TIMEOUT_SECONDS", "60")) or "60"),
            llm_retry_count=int(os.getenv("LLM_RETRY_COUNT", "2") or "2"),
            override_llm_can_modify_98=os.getenv("OVERRIDE_LLM_CAN_MODIFY_98", "false").lower() == "true",
            openai_input_cost_per_1m_tokens=float(os.getenv("OPENAI_INPUT_COST_PER_1M_TOKENS", "0") or "0"),
            openai_output_cost_per_1m_tokens=float(os.getenv("OPENAI_OUTPUT_COST_PER_1M_TOKENS", "0") or "0"),
            allow_unknown_llm_cost=os.getenv("ALLOW_UNKNOWN_LLM_COST", "false").lower() == "true",
            embeddings_enabled=os.getenv("EMBEDDINGS_ENABLED", "false").lower() == "true",
            output_dir=os.getenv("OUTPUT_DIR", "./output"),
            allow_parent_family_tax_conflicts=os.getenv("ALLOW_PARENT_FAMILY_TAX_CONFLICTS", "true").lower() == "true",
            max_total_candidate_pairs=int(os.getenv("MAX_TOTAL_CANDIDATE_PAIRS", "1000000")),
            max_weak_block_size=int(os.getenv("MAX_WEAK_BLOCK_SIZE", "300")),
            exact_tax_star_threshold=int(os.getenv("EXACT_TAX_STAR_THRESHOLD", "100")),
            max_exact_tax_only_block_size=int(os.getenv("MAX_EXACT_TAX_ONLY_BLOCK_SIZE", "10")),
            max_exact_tax_distinct_roots=int(os.getenv("MAX_EXACT_TAX_DISTINCT_ROOTS", "3")),
            max_weak_review_cluster_size=int(os.getenv("MAX_WEAK_REVIEW_CLUSTER_SIZE", "50")),
            max_low_confidence_cluster_size=int(os.getenv("MAX_LOW_CONFIDENCE_CLUSTER_SIZE", "6")),
            max_known_family_cluster_size=int(os.getenv("MAX_KNOWN_FAMILY_CLUSTER_SIZE", "25")),
            max_known_brand_family_block_size=int(os.getenv("MAX_KNOWN_BRAND_FAMILY_BLOCK_SIZE", "300")),
            rare_token_max_document_fraction=float(os.getenv("RARE_TOKEN_MAX_DOCUMENT_FRACTION", "0.03")),
            rare_token_min_document_frequency=int(os.getenv("RARE_TOKEN_MIN_DOCUMENT_FREQUENCY", "1")),
            ai_uncertain_cluster_enabled=os.getenv("AI_UNCERTAIN_CLUSTER_ENABLED", "true").lower() == "true",
            ai_uncertain_match_pct=float(os.getenv("AI_UNCERTAIN_MATCH_PCT", "68.0")),
            allow_unresolved_llm_candidates_in_final_output=os.getenv("ALLOW_UNRESOLVED_LLM_CANDIDATES_IN_FINAL_OUTPUT", "false").lower() == "true",
            unresolved_llm_candidate_mode=os.getenv("UNRESOLVED_LLM_CANDIDATE_MODE", "exception").lower(),
            expose_unresolved_llm_candidates=os.getenv("EXPOSE_UNRESOLVED_LLM_CANDIDATES", "false").lower() == "true",
            ai_max_calls=int(os.getenv("AI_MAX_CALLS", "50")),
            ai_cache_enabled=os.getenv("AI_CACHE_ENABLED", "true").lower() == "true",
            ai_cache_path=os.getenv("AI_CACHE_PATH", ".cache/llm_review_cache.json"),
            llm_can_auto_cluster=os.getenv("LLM_CAN_AUTO_CLUSTER", "false").lower() == "true",
            known_brand_families_file=os.getenv("KNOWN_BRAND_FAMILIES_FILE", "data/known_brand_families.csv"),
            known_brand_family_default_confidence=float(os.getenv("KNOWN_BRAND_FAMILY_DEFAULT_CONFIDENCE", "76.0")),
            legal_keywords_file=os.getenv("LEGAL_KEYWORDS_FILE", "data/legal_keywords.csv"),
            generic_non_bridge_file=os.getenv("GENERIC_NON_BRIDGE_FILE", "data/generic_non_bridge_keywords.csv"),
            support_field_strengths=parse_support_field_strengths(os.getenv("SUPPORT_FIELD_STRENGTHS", "")),
        )


def _normalize_llm_execution_mode(value: str) -> str:
    """Normalize external LLM mode names to internal execution modes."""
    mode = str(value or "disabled").strip().lower()
    if mode == "sync":
        return "live"
    if mode in {"off", "none"}:
        return "disabled"
    return mode
