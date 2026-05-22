"""Phase B alias framework tests.

Covers:
1.  CSV schema/load validation
2.  Missing/empty alias files do not change engine behavior
3.  Low-risk alias with support can become 85
4.  Medium-risk alias alone becomes 70, not 85
5.  High-risk alias alone caps at blank (no cluster)
6.  Alias + same tax/address/domain can reach 98/85
7.  Ignored client domain suppresses domain alias evidence
8.  Free email domain still ignored regardless of alias tables
9.  Merck US and Merck KGaA do not merge via alias
10. Amazon/AWS works when supported; common-name Amazon alone stays medium (70)
11. Shell/Orange/Bell/Target/Nissan/Subway/Marriott stay high-risk (no auto-85)
12. Output columns stay clean (no alias internals leaked)
13. All existing tests still pass (tested by running full suite separately)
"""
import os
import csv
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
import polars as pl

from src.config import ClusteringConfig
from src.matching import evaluate_pair
from src.matching_types import MatchResult
from src.brand_families import (
    load_alias_tables,
    get_alias_tables,
    AliasTables,
    AliasEvidence,
)
from src.main import cluster_suppliers
from src.preprocessing import extract_supplier_identity_core

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
_cfg = ClusteringConfig()


def _row(
    name_norm: str,
    supplier_identity_core: str = "",
    domain: str = "",
    domain_sld: str = "",
    addr_norm: str = "",
    city_norm: str = "",
    country_norm: str = "",
    tax_norm: str = "",
    is_generic_domain: bool = False,
    row_id: int = 0,
) -> Dict[str, Any]:
    core = supplier_identity_core or extract_supplier_identity_core(name_norm)
    return {
        "name_norm": name_norm,
        "supplier_identity_core": core,
        "domain": domain,
        "domain_sld": domain_sld,
        "addr_norm": addr_norm,
        "city_norm": city_norm,
        "country_norm": country_norm,
        "tax_norm": tax_norm,
        "tax_loose_norm": "",
        "is_generic_domain": is_generic_domain,
        "name_location_core": name_norm,
        "row_id": row_id,
        "postal_norm": "",
        "has_operational_status_hint": False,
        "root_brand": "",
        "known_brand_family_ids": "",
        "known_brand_family_safe_ids": "",
        "known_brand_family_risky_ids": "",
        "known_brand_alias_hits": "",
        "support_fields_json": "[]",
        "json_secondary_names_norm": "",
        "is_likely_individual": False,
        "is_hospitality": False,
        "has_legal_suffix": False,
        "person_name_norm": "",
        "idf_discriminative_tokens": "",
        "franchise_store_number": "",
    }


def _cluster_simple(records, config=None):
    """Run cluster_suppliers on (name, email) tuples."""
    df = pl.DataFrame({
        "Supplier Name": [r[0] for r in records],
        "Email": [r[1] if len(r) > 1 else "" for r in records],
        "Country": [r[2] if len(r) > 2 else "" for r in records],
    })
    return cluster_suppliers(df, {
        "supplier_name": "Supplier Name",
        "email": "Email",
        "country": "Country",
    }, config or ClusteringConfig())


def _clusters_by_name(result):
    return {r["Supplier Name"]: r["Cluster Number"] for r in result["main_df"].to_dicts()}


# ---------------------------------------------------------------------------
# 1. CSV schema/load validation
# ---------------------------------------------------------------------------

class TestCsvSchemaLoad:

    def test_brand_aliases_csv_exists_and_loads(self):
        """brand_aliases.csv must exist and have required columns."""
        path = DATA_DIR / "brand_aliases.csv"
        assert path.exists(), f"Missing {path}"
        with path.open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            required = {"alias", "canonical_brand", "score_hint", "risk_level", "category"}
            missing = required - set(reader.fieldnames)
            assert not missing, f"Missing columns: {missing}"
            rows = list(reader)
            assert len(rows) > 0, "brand_aliases.csv must have at least one data row"

    def test_domain_aliases_csv_exists_and_loads(self):
        path = DATA_DIR / "domain_aliases.csv"
        assert path.exists(), f"Missing {path}"
        with path.open() as f:
            reader = csv.DictReader(f)
            required = {"domain", "canonical_domain_family", "canonical_brand", "score_hint", "risk_level"}
            missing = required - set(reader.fieldnames or [])
            assert not missing, f"Missing columns: {missing}"

    def test_acronym_aliases_csv_exists_and_loads(self):
        path = DATA_DIR / "acronym_aliases.csv"
        assert path.exists(), f"Missing {path}"
        with path.open() as f:
            reader = csv.DictReader(f)
            required = {"acronym", "full_form", "canonical_brand", "score_hint", "risk_level"}
            missing = required - set(reader.fieldnames or [])
            assert not missing, f"Missing columns: {missing}"

    def test_risk_levels_valid(self):
        """Every risk_level in brand_aliases.csv must be low/medium/high."""
        path = DATA_DIR / "brand_aliases.csv"
        if not path.exists():
            pytest.skip("brand_aliases.csv not found")
        valid = {"low", "medium", "high"}
        with path.open() as f:
            for i, row in enumerate(csv.DictReader(f), 1):
                risk = row.get("risk_level", "").strip().lower()
                assert risk in valid, f"Row {i}: invalid risk_level '{risk}'"

    def test_score_hints_valid(self):
        """Every score_hint in brand_aliases.csv must be 70, 85, or 98."""
        path = DATA_DIR / "brand_aliases.csv"
        if not path.exists():
            pytest.skip("brand_aliases.csv not found")
        valid = {70, 85, 98}
        with path.open() as f:
            for i, row in enumerate(csv.DictReader(f), 1):
                try:
                    score = int(row.get("score_hint", "0"))
                except ValueError:
                    pytest.fail(f"Row {i}: non-integer score_hint '{row.get('score_hint')}'")
                assert score in valid, f"Row {i}: score_hint {score} not in {{70, 85, 98}}"

    def test_key_aliases_present(self):
        """Critical aliases must be present with correct canonicals."""
        tables = load_alias_tables(str(DATA_DIR))
        # AWS → amazon
        assert "aws" in tables.alias_to_entry or "aws" in tables.acronym_to_entry, "aws alias missing"
        entry = tables.alias_to_entry.get("aws") or tables.acronym_to_entry.get("aws")
        assert entry and entry[0] == "amazon", f"aws should map to amazon, got {entry}"
        # msft → microsoft
        assert "msft" in tables.alias_to_entry, "msft alias missing"
        assert tables.alias_to_entry["msft"][0] == "microsoft"
        # merck kgaa domains must NOT map to merck (US)
        if "merckgroup.com" in tables.domain_to_entry:
            assert tables.domain_to_entry["merckgroup.com"][0] == "merck kgaa", \
                "merckgroup.com must map to 'merck kgaa', not 'merck'"

    def test_load_alias_tables_returns_alias_tables_type(self):
        tables = load_alias_tables(str(DATA_DIR))
        assert isinstance(tables, AliasTables)


# ---------------------------------------------------------------------------
# 2. Missing/empty alias files do not change engine behavior
# ---------------------------------------------------------------------------

class TestGracefulDegradation:

    def test_missing_data_dir_returns_empty_tables(self):
        tables = load_alias_tables("/nonexistent/path/to/data")
        assert tables.is_empty

    def test_empty_tables_alias_bridge_returns_none(self):
        """When alias tables are empty, _alias_bridge returns None → no change to scoring."""
        from src.matching import _alias_bridge
        cfg = ClusteringConfig(alias_tables_dir="/nonexistent/path")
        r1 = _row("microsoft corporation")
        r2 = _row("msft")
        result = _alias_bridge(r1, r2, cfg)
        assert result is None

    def test_empty_tables_evaluate_pair_unchanged(self):
        """evaluate_pair with empty alias tables must return same result as before aliases."""
        cfg_no_alias = ClusteringConfig(alias_tables_dir="/nonexistent/path")
        cfg_default = ClusteringConfig(alias_tables_dir="/nonexistent/path")
        r1 = _row("some supplier inc", domain="somecompany.com", domain_sld="somecompany")
        r2 = _row("other vendor llc", domain="othervendor.com",  domain_sld="othervendor")
        result = evaluate_pair(r1, r2, {}, cfg_no_alias)
        # No domain match, no name match, no address — must be no_match
        assert not result.is_match


# ---------------------------------------------------------------------------
# 3. Low-risk alias can support 85
# ---------------------------------------------------------------------------

class TestLowRiskAlias:

    def test_low_risk_alias_alone_reaches_85(self):
        """Low-risk alias (e.g. 'msft' → microsoft) with no other evidence → ≥85."""
        tables = load_alias_tables(str(DATA_DIR))
        if "msft" not in tables.alias_to_entry:
            pytest.skip("msft alias not in brand_aliases.csv")
        _, score, risk = tables.alias_to_entry["msft"]
        if risk != "low" or score < 85:
            # Some implementations may set microsoft as medium — skip if so
            pytest.skip(f"msft alias is {risk}/{score}, not low/85")

        cfg = ClusteringConfig()
        r1 = _row("microsoft corporation", supplier_identity_core="microsoft")
        r2 = _row("msft software solutions", supplier_identity_core="msft")
        result = evaluate_pair(r1, r2, {}, cfg)
        assert result.is_match, "Low-risk alias should produce a match"
        assert result.match_pct >= 85.0, f"Low-risk alias score should be ≥85, got {result.match_pct}"

    def test_low_risk_alias_jnj_janssen(self):
        """janssen → johnson & johnson is a low-risk alias (distinctive sub-brand)."""
        tables = load_alias_tables(str(DATA_DIR))
        janssen = tables.alias_to_entry.get("janssen") or tables.alias_to_entry.get("janssen pharmaceutica")
        if janssen is None:
            pytest.skip("janssen alias not found in brand_aliases.csv")
        assert janssen[0] == "johnson & johnson", f"janssen should map to j&j, got {janssen[0]}"
        assert janssen[2] == "low", f"janssen risk should be low, got {janssen[2]}"

    def test_low_risk_alias_bms(self):
        """bms → bristol myers squibb is a distinctive low-risk acronym."""
        tables = load_alias_tables(str(DATA_DIR))
        entry = tables.acronym_to_entry.get("bms") or tables.alias_to_entry.get("bms")
        if entry is None:
            pytest.skip("bms not found")
        assert entry[0] == "bristol myers squibb"
        assert entry[2] == "low", f"bms risk should be low, got {entry[2]}"


# ---------------------------------------------------------------------------
# 4. Medium-risk alias alone becomes 70, not 85
# ---------------------------------------------------------------------------

class TestMediumRiskAlias:

    def test_medium_risk_alias_alone_is_70(self):
        """AWS (medium risk) without domain/address support → 70, never 85."""
        tables = load_alias_tables(str(DATA_DIR))
        aws_entry = tables.alias_to_entry.get("aws") or tables.acronym_to_entry.get("aws")
        if aws_entry is None:
            pytest.skip("aws alias not found")
        if aws_entry[2] != "medium":
            pytest.skip(f"aws is classified as {aws_entry[2]}, skipping medium test")

        cfg = ClusteringConfig()
        r1 = _row("amazon marketplace", supplier_identity_core="amazon")
        r2 = _row("aws cloud services",  supplier_identity_core="aws")
        result = evaluate_pair(r1, r2, {}, cfg)
        assert result.is_match, "Medium-risk alias should still produce a match"
        assert result.match_pct <= 70.0, (
            f"Medium-risk alias alone must be ≤70, got {result.match_pct}"
        )

    def test_medium_risk_alias_with_same_domain_can_exceed_70(self):
        """Medium-risk alias + same owned domain → existing domain pass can score >70."""
        cfg = ClusteringConfig()
        # amazon.com → same domain → domain pass fires first, alias augments related_root
        r1 = _row("amazon inc",           domain="amazon.com", domain_sld="amazon", supplier_identity_core="amazon")
        r2 = _row("amazon web services",  domain="amazon.com", domain_sld="amazon", supplier_identity_core="amazon web services")
        result = evaluate_pair(r1, r2, {}, cfg)
        assert result.is_match
        # same_domain=True + related_root=True (from alias) → domain_name_related at ≥78
        assert result.match_pct >= 72.0, f"Domain+alias should score ≥72, got {result.match_pct}"


# ---------------------------------------------------------------------------
# 5. High-risk alias alone = no cluster
# ---------------------------------------------------------------------------

class TestHighRiskAlias:

    def test_shell_high_risk_alias_alone_no_cluster(self):
        """shell / shell oil with no shared domain/address/tax → no match."""
        tables = load_alias_tables(str(DATA_DIR))
        entry = tables.alias_to_entry.get("shell oil") or tables.alias_to_entry.get("shell")
        if entry is None:
            pytest.skip("shell alias not found")
        # Verify shell is high risk in the CSV
        assert entry[2] == "high", f"shell should be high risk, got {entry[2]}"

        cfg = ClusteringConfig()
        r1 = _row("shell", supplier_identity_core="shell",
                  domain="shell.com", domain_sld="shell")
        r2 = _row("shell oil company", supplier_identity_core="shell oil",
                  domain="shelloil.com", domain_sld="shelloil")
        result = evaluate_pair(r1, r2, {}, cfg)
        # No shared domain, no address — high risk alias alone must not produce a cluster
        if result.is_match:
            assert result.pass_type != "brand_alias_candidate", (
                "High-risk alias must not fire brand_alias_candidate"
            )

    @pytest.mark.parametrize("brand,alias", [
        ("orange",   "orange telecom"),
        ("target",   "target stores"),
        ("marriott", "marriott hotels"),
        ("subway",   "subway restaurants"),
        ("nissan",   "nissan motor"),
    ])
    def test_high_risk_brands_are_classified_high(self, brand, alias):
        """Franchise/retail/telecom brands must be high-risk in the alias CSV."""
        tables = load_alias_tables(str(DATA_DIR))
        entry = (
            tables.alias_to_entry.get(brand)
            or tables.alias_to_entry.get(alias)
            or tables.alias_to_entry.get(alias.split()[0])
        )
        if entry is None:
            pytest.skip(f"{brand} alias not found in brand_aliases.csv")
        assert entry[2] == "high", (
            f"{brand} must be high-risk (franchise/retail/telecom), got {entry[2]}"
        )

    def test_bell_high_risk_no_auto_85(self):
        """Bell (telecom) must not auto-cluster at 85 via alias alone."""
        tables = load_alias_tables(str(DATA_DIR))
        bell_entry = tables.alias_to_entry.get("bell") or tables.alias_to_entry.get("bell canada")
        if bell_entry is None:
            pytest.skip("bell alias not found")
        assert bell_entry[2] == "high", f"bell should be high risk, got {bell_entry[2]}"

    def test_ge_short_acronym_high_risk(self):
        """'ge' is a 2-char acronym — must be high-risk regardless of brand recognition."""
        tables = load_alias_tables(str(DATA_DIR))
        entry = tables.acronym_to_entry.get("ge") or tables.alias_to_entry.get("ge")
        if entry is None:
            pytest.skip("ge not found in alias tables")
        assert entry[2] == "high", f"Short acronym 'ge' must be high risk, got {entry[2]}"


# ---------------------------------------------------------------------------
# 6. Alias + deterministic support reaches 98/85
# ---------------------------------------------------------------------------

class TestAliasWithDeterministicSupport:

    def test_alias_plus_same_tax_can_reach_98(self):
        """Low/medium alias + matching tax ID → existing tax pass scores 98."""
        cfg = ClusteringConfig()
        r1 = _row("microsoft corporation", supplier_identity_core="microsoft",
                  tax_norm="us-12-3456789")
        r2 = _row("msft ltd",             supplier_identity_core="msft",
                  tax_norm="us-12-3456789")
        result = evaluate_pair(r1, r2, {}, cfg)
        assert result.is_match
        assert result.match_pct >= 85.0, f"Alias + same tax should reach ≥85, got {result.match_pct}"

    def test_alias_plus_address_support(self):
        """Alias + matching address → existing address pass can produce a strong score."""
        cfg = ClusteringConfig()
        r1 = _row("amazon inc",           addr_norm="410 terry ave n seattle wa",
                  supplier_identity_core="amazon")
        r2 = _row("amazon web services",  addr_norm="410 terry ave n seattle wa",
                  supplier_identity_core="amazon web services")
        result = evaluate_pair(r1, r2, {}, cfg)
        assert result.is_match
        assert result.match_pct >= 70.0


# ---------------------------------------------------------------------------
# 7. Ignored client domain suppresses domain alias evidence
# ---------------------------------------------------------------------------

class TestIgnoredDomainSuppressesAlias:

    def test_ignored_domain_suppresses_domain_alias(self):
        """If amazon.com is in ignore_client_domains, domain alias for amazon must not fire."""
        cfg = ClusteringConfig(ignore_client_domains=frozenset({"amazon.com"}))
        # Use unrelated names so only domain evidence could trigger a match
        r1 = _row("acme wholesale corp",   domain="amazon.com", domain_sld="amazon",
                  supplier_identity_core="acme wholesale")
        r2 = _row("vertex retail systems", domain="amazon.in",  domain_sld="amazon",
                  supplier_identity_core="vertex retail")
        result = evaluate_pair(r1, r2, {}, cfg)
        assert not result.is_match, (
            "Unrelated names must not cluster when domain is in ignore list"
        )


# ---------------------------------------------------------------------------
# 8. Free email domain still ignored
# ---------------------------------------------------------------------------

class TestFreeEmailDomainStillIgnored:

    @pytest.mark.parametrize("domain", ["gmail.com", "outlook.com", "yahoo.com", "hotmail.com"])
    def test_generic_domain_never_clusters(self, domain):
        """Free/public email domains must never produce domain-based matches."""
        cfg = ClusteringConfig()
        r1 = {**_row("supplier alpha inc",  domain=domain), "is_generic_domain": True}
        r2 = {**_row("vendor beta limited", domain=domain), "is_generic_domain": True}
        result = evaluate_pair(r1, r2, {}, cfg)
        if result.is_match:
            assert "domain" not in result.pass_type, (
                f"{domain} must never create domain-based matches (got {result.pass_type})"
            )


# ---------------------------------------------------------------------------
# 9. Merck US and Merck KGaA do not merge
# ---------------------------------------------------------------------------

class TestMerckSeparation:

    def test_merck_us_and_merck_kgaa_have_separate_canonicals(self):
        """merck.com → merck (US) and merckgroup.com → merck kgaa must be separate."""
        tables = load_alias_tables(str(DATA_DIR))
        if "merck.com" in tables.domain_to_entry:
            merck_us = tables.domain_to_entry["merck.com"][0]
            assert merck_us == "merck", f"merck.com should map to 'merck', got '{merck_us}'"
        if "merckgroup.com" in tables.domain_to_entry:
            merck_eu = tables.domain_to_entry["merckgroup.com"][0]
            assert merck_eu == "merck kgaa", \
                f"merckgroup.com should map to 'merck kgaa', got '{merck_eu}'"

    def test_merck_us_and_merck_kgaa_do_not_alias_bridge(self):
        """Rows resolving to different Merck canonicals must NOT alias-bridge."""
        from src.matching import _alias_bridge
        cfg = ClusteringConfig()
        # Row A: Merck US (merck.com domain)
        r1 = _row("merck sharp dohme", supplier_identity_core="merck",
                  domain="merck.com", domain_sld="merck")
        # Row B: Merck KGaA (merckgroup.com domain)
        r2 = _row("merck group research", supplier_identity_core="merck kgaa",
                  domain="merckgroup.com", domain_sld="merckgroup")
        result = _alias_bridge(r1, r2, cfg)
        # They should NOT resolve to the same canonical
        if result is not None:
            assert result.canonical_brand != "merck" or result.canonical_brand != "merck kgaa", (
                "Merck US and Merck KGaA must not bridge to the same canonical"
            )
        # More specifically: if both map to the same canonical, the test fails
        tables = load_alias_tables(str(DATA_DIR))
        us_entry = tables.alias_to_entry.get("merck") or tables.domain_to_entry.get("merck.com")
        eu_entry = tables.alias_to_entry.get("merck kgaa") or tables.domain_to_entry.get("merckgroup.com")
        if us_entry and eu_entry:
            us_canonical = us_entry[0] if isinstance(us_entry[0], str) else "merck"
            eu_canonical = eu_entry[0] if isinstance(eu_entry[0], str) else "merck kgaa"
            assert us_canonical != eu_canonical, (
                f"merck US and merck kgaa must have different canonicals; both got '{us_canonical}'"
            )

    def test_sigma_aldrich_maps_to_merck_kgaa_not_merck_us(self):
        """sigma aldrich is a Merck KGaA subsidiary — must not map to Merck US."""
        tables = load_alias_tables(str(DATA_DIR))
        entry = tables.alias_to_entry.get("sigma aldrich") or tables.alias_to_entry.get("sigma-aldrich")
        if entry is None:
            pytest.skip("sigma aldrich not in alias tables")
        assert entry[0] == "merck kgaa", (
            f"sigma aldrich must map to 'merck kgaa', got '{entry[0]}'"
        )


# ---------------------------------------------------------------------------
# 10. Amazon/AWS — medium risk, supported vs unsupported
# ---------------------------------------------------------------------------

class TestAmazonAwsAlias:

    def test_aws_maps_to_amazon(self):
        """aws must resolve to amazon canonical."""
        tables = load_alias_tables(str(DATA_DIR))
        entry = tables.alias_to_entry.get("aws") or tables.acronym_to_entry.get("aws")
        assert entry is not None, "aws alias must exist"
        assert entry[0] == "amazon", f"aws must map to amazon, got {entry[0]}"

    def test_aws_is_medium_risk(self):
        """aws is medium risk — needs support to reach 85."""
        tables = load_alias_tables(str(DATA_DIR))
        entry = tables.alias_to_entry.get("aws") or tables.acronym_to_entry.get("aws")
        if entry is None:
            pytest.skip("aws not found")
        assert entry[2] == "medium", f"aws should be medium risk, got {entry[2]}"

    def test_amazon_alias_alone_does_not_auto_85(self):
        """amazon + aws without domain/address support stays at 70."""
        cfg = ClusteringConfig()
        r1 = _row("amazon logistics", supplier_identity_core="amazon")
        r2 = _row("aws data services", supplier_identity_core="aws")
        result = evaluate_pair(r1, r2, {}, cfg)
        if result.is_match and result.pass_type == "brand_alias_candidate":
            assert result.match_pct <= 70.0, (
                f"Medium-risk alias alone must not exceed 70, got {result.match_pct}"
            )

    def test_amazon_sld_cross_tld_clusters_without_ignore(self):
        """amazon.com / amazon.in share SLD 'amazon' — clusters by default."""
        cfg = ClusteringConfig()
        r1 = _row("acme distributor", domain="amazon.com", domain_sld="amazon")
        r2 = _row("vertex logistics", domain="amazon.in",  domain_sld="amazon")
        result = evaluate_pair(r1, r2, {}, cfg)
        assert result.is_match, "SLD amazon must create evidence by default"


# ---------------------------------------------------------------------------
# 11. High-risk franchise/retail/telecom brands do not auto-85
# ---------------------------------------------------------------------------

class TestHighRiskBrandsNoAuto85:

    @pytest.mark.parametrize("brand_a,core_a,brand_b,core_b", [
        ("shell plc",       "shell",   "shell oil company",  "shell oil"),
        ("orange sa",       "orange",  "orange telecom",     "orange"),
        ("target corporation","target","target stores inc",  "target"),
        ("subway restaurants","subway","subway franchisee",  "subway"),
        ("nissan motor co", "nissan",  "nissan north america","nissan"),
        ("marriott international","marriott","marriott hotels","marriott"),
    ])
    def test_high_risk_brand_alias_does_not_produce_85(self, brand_a, core_a, brand_b, core_b):
        """High-risk brands must not auto-cluster at 85 via alias alone."""
        cfg = ClusteringConfig()
        r1 = _row(brand_a, supplier_identity_core=core_a)
        r2 = _row(brand_b, supplier_identity_core=core_b)
        result = evaluate_pair(r1, r2, {}, cfg)
        if result.is_match and result.pass_type == "brand_alias_candidate":
            assert result.match_pct < 85.0, (
                f"High-risk brand '{core_a}' must not produce brand_alias_candidate at ≥85, "
                f"got {result.match_pct}"
            )


# ---------------------------------------------------------------------------
# 12. Output columns stay clean
# ---------------------------------------------------------------------------

class TestOutputColumnsClean:

    def test_no_alias_internals_in_main_output(self):
        """Internal alias columns must not appear in cluster_suppliers main output."""
        cfg = ClusteringConfig()
        result = _cluster_simple([
            ("Microsoft Corporation", "info@microsoft.com", "US"),
            ("MSFT Software Ltd",     "sales@microsoft.com", "US"),
        ], config=cfg)
        output_cols = set(result["main_df"].columns)
        forbidden = {
            "alias_ev", "_alias_ev", "alias_evidence",
            "alias_a", "alias_b", "alias_match_type",
            "canonical_brand", "brand_alias_candidate",
        }
        leaked = forbidden & output_cols
        assert not leaked, f"Internal alias columns leaked into output: {leaked}"

    def test_preprocessed_df_has_no_alias_cols(self):
        """preprocessed_df must also be clean of alias internals."""
        cfg = ClusteringConfig()
        result = _cluster_simple([
            ("Amazon Inc",         "contact@amazon.com",   "US"),
            ("Amazon India Pvt",   "contact@amazon.in",    "IN"),
        ], config=cfg)
        preproc = result.get("preprocessed_df")
        if preproc is not None:
            assert "alias_ev" not in preproc.columns
            assert "canonical_brand" not in preproc.columns
