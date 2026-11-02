"""Deterministic tests for the data layer and tools. No LLM needed: run with `pytest`."""

import json
from datetime import date

import pytest

from emporio_agent.data_loader import load_store_data, normalize
from emporio_agent.tools import StoreTools, max_installments, tool_schemas

TODAY = date(2026, 11, 2)


@pytest.fixture(scope="module")
def tools():
    return StoreTools(load_store_data(), today=TODAY)


# ---------- data loader

def test_normalize_removes_accents_and_punctuation():
    assert normalize("Violões Elétricos!") == "violoes eletricos"


def test_only_active_promotions_are_applied(tools):
    promo_ids = {p["product_id"] for p in tools.list_active_promotions()["promotions"]}
    assert promo_ids == {90, 94, 121, 127}


def test_pix_discount_not_cumulative_with_promotion(tools):
    p = tools.data.products
    assert p.loc[90, "promo_price_brl"] == pytest.approx(2399 * 0.82, abs=0.01)
    assert p.loc[90, "pix_price_brl"] != p.loc[90, "pix_price_brl"]  # NaN -> no PIX discount
    assert p.loc[95, "pix_price_brl"] == pytest.approx(2199 * 0.95, abs=0.01)


# ---------- get_product

def test_price_lookup_takamine_gd20(tools):
    r = tools.get_product("Takamine GD20")
    assert r["found"] and r["product_id"] == 95
    assert r["price_brl"] == 2199.0
    assert "promotion" not in r  # its promotion (id 4) is inactive
    assert r["pix_price_brl"] == pytest.approx(2089.05)


def test_product_lookup_tolerates_hyphen_and_case(tools):
    assert tools.get_product("takamine gd-20")["product_id"] == 95


def test_ambiguous_product_returns_options(tools):
    r = tools.get_product("Taylor")
    assert r["multiple_matches"] and len(r["products"]) == 2


def test_out_of_stock_product_suggests_available_alternatives(tools):
    r = tools.get_product(96)  # Giannini GF-3D, stock 0
    assert r["available"] is False and r["status"] == "esgotado no momento"
    assert r["alternatives"] and all(a["available"] for a in r["alternatives"])


def test_discontinued_product_is_flagged(tools):
    r = tools.get_product(113)
    assert r["available"] is False and "descontinuado" in r["status"]


def test_unknown_product(tools):
    assert tools.get_product("Stradivarius 1715")["found"] is False


# ---------- search_products

def test_acoustic_guitars_up_to_1000(tools):
    r = tools.search_products(category="violão", max_price=1000)
    assert r["total_found"] > 0
    for p in r["products"]:
        assert p["category"] == "Violões" and p["final_price_brl"] <= 1000 and p["available"]
    ids = {p["product_id"] for p in r["products"]}
    assert 96 not in ids and 113 not in ids  # out of stock / discontinued


def test_search_by_keyword(tools):
    r = tools.search_products(query="nylon elétrico", category="Violões")
    assert r["total_found"] > 0 and not r["partial_match"]


def test_invalid_category_returns_valid_list(tools):
    r = tools.search_products(category="amplificador")
    assert "error" in r and "Violões" in r["valid_categories"]


# ---------- get_order_status (identity check)

def test_order_with_correct_email(tools):
    r = tools.get_order_status(2, "Thiago.Barbosa@jmail.com")
    assert r["found"] and r["status"] == "entregue"
    assert r["days_since_delivery"] == (TODAY - date(2025, 11, 15)).days


def test_order_with_phone_without_area_code(tools):
    assert tools.get_order_status(8, "99654-7821")["found"]


def test_wrong_identity_and_missing_order_look_the_same(tools):
    wrong = tools.get_order_status(2, "someone@else.com")
    missing = tools.get_order_status(999, "thiago.barbosa@jmail.com")
    assert wrong == missing and wrong["found"] is False


def test_stale_pending_order_is_escalated(tools):
    r = tools.get_order_status(13, "fernanda.ribeiro@coldmail.com")
    assert r["needs_human_followup"] is True


# ---------- payments

@pytest.mark.parametrize("amount,expected", [(2199, 12), (450, 5), (159.9, 3), (99, 1)])
def test_max_installments(amount, expected):
    assert max_installments(amount) == expected


# ---------- dispatcher

def test_execute_returns_json_and_never_raises(tools):
    ok = json.loads(tools.execute("get_product", '{"product": "Takamine GD20"}'))
    assert ok["product_id"] == 95
    err = json.loads(tools.execute("get_product", '{"wrong_arg": 1}'))
    assert "error" in err
    assert "error" in json.loads(tools.execute("does_not_exist", "{}"))


def test_schemas_cover_all_tools(tools):
    names = {s["function"]["name"] for s in tool_schemas(tools.data.categories["name"].tolist())}
    assert names == set(tools.registry())
