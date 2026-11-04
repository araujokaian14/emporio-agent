"""Deterministic tools the LLM can call (function calling).

Rule of thumb: anything that must be exact (price, stock, order status, installments)
is computed here in Python, never by the LLM. The model only decides WHICH tool to call
and turns the JSON result into a friendly answer.
"""

from __future__ import annotations

import difflib
import json
from datetime import date
from typing import Any, Callable

import pandas as pd

from .config import today as store_today
from .data_loader import StoreData, compact, money, normalize

# --------------------------------------------------------------------------- constants

STATUS_LABELS = {
    "active": "disponível",
    "discontinued": "descontinuado (não faz mais parte do catálogo)",
    "coming_soon": "em breve (ainda não disponível para venda)",
}

ORDER_STATUS_LABELS = {
    "pending": "aguardando confirmação de pagamento",
    "confirmed": "pagamento confirmado, em separação",
    "shipped": "enviado",
    "delivered": "entregue",
    "cancelled": "cancelado",
}

# Category synonyms customers actually type -> category_id(s)
CATEGORY_ALIASES: dict[str, list[int]] = {
    "guitarra": [1], "guitarras": [1],
    "baixo": [2], "baixos": [2],
    "bateria": [3], "baterias": [3], "percussao": [3],
    "teclado": [4], "teclados": [4], "piano": [4], "pianos": [4], "sintetizador": [4],
    "violao": [5], "violoes": [5],
    "sopro": [6, 7], "flauta": [6], "saxofone": [6], "sax": [6], "clarinete": [6],
    "trompete": [7], "trombone": [7], "tuba": [7],
    "violino": [8], "viola": [8], "violoncelo": [8], "cello": [8],
    "ukulele": [9], "ukuleles": [9],
}

STOPWORDS = {"de", "da", "do", "das", "dos", "para", "com", "e", "o", "a", "os", "as",
             "um", "uma", "ate", "em", "no", "na", "por", "r"}

# Installment rules from policy section 3.1.
# ASSUMPTION: "sem valor mínimo (exceto abaixo de R$ 50,00)" = installments of 2x-3x
# must be at least R$ 50. The summary table in section 3 (min R$ 100 for any credit)
# contradicts 3.1; we follow the more detailed rule in 3.1 and document it.
MAX_INSTALLMENTS = 12


def min_installment_value(n: int) -> float:
    if n <= 3:
        return 50.0
    if n <= 6:
        return 80.0
    return 100.0


def max_installments(amount: float) -> int:
    best = 1
    for n in range(2, MAX_INSTALLMENTS + 1):
        if amount / n >= min_installment_value(n):
            best = n
    return best


def _payment_label(method: str | None) -> str | None:
    if not method:
        return None
    if method.startswith("credit_"):
        return f"cartão de crédito em {method.split('_', 1)[1]}"
    return {"pix": "PIX", "debit": "cartão de débito", "boleto": "boleto bancário"}.get(method, method)


def _money(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), 2)


# --------------------------------------------------------------------------- tools

class StoreTools:
    def __init__(self, data: StoreData, today: date | None = None, policies: Any = None):
        """`policies` is an optional PolicyIndex; when given, `search_policies` becomes a tool."""
        self.data = data
        self._today = today
        self.policies = policies

    @property
    def today(self) -> date:
        return self._today or store_today()

    # ---------- helpers

    def _product_summary(self, row: pd.Series) -> dict:
        summary = {
            "product_id": int(row["product_id"]),
            "name": row["name"],
            "category": row["category_name"],
            "price_brl": _money(row["price_brl"]),
            "final_price_brl": _money(row["final_price_brl"]),
            "stock_quantity": int(row["stock_quantity"]),
            "status": STATUS_LABELS.get(row["status"], row["status"]),
            "available": bool(row["is_available"]),
        }
        if pd.notna(row["discount_percent"]):
            summary["promotion"] = f"{row['promotion_name']} (-{int(row['discount_percent'])}%)"
        if row["status"] == "active" and row["stock_quantity"] == 0:
            summary["status"] = "esgotado no momento"
        return summary

    def _resolve_category(self, category: str) -> list[int] | None:
        key = normalize(category)
        cats = self.data.categories
        for _, c in cats.iterrows():
            if normalize(c["name"]) == key:
                return [int(c["category_id"])]
        if key in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[key]
        for word in key.split():
            if word in CATEGORY_ALIASES:
                return CATEGORY_ALIASES[word]
        names = {normalize(c["name"]): int(c["category_id"]) for _, c in cats.iterrows()}
        close = difflib.get_close_matches(key, list(names), n=1, cutoff=0.6)
        return [names[close[0]]] if close else None

    def _alternatives(self, row: pd.Series, k: int = 3) -> list[dict]:
        p = self.data.products
        pool = p[(p["category_id"] == row["category_id"]) & p["is_available"] & (p["product_id"] != row["product_id"])]
        price = row["price_brl"]
        pool = pool[(pool["price_brl"] >= price * 0.6) & (pool["price_brl"] <= price * 1.5)]
        # Prefer the same string type (nylon vs steel) first, then the closest price.
        strings = row["specs"].get("strings")
        same_type = pool["specs"].apply(lambda s: s.get("strings") == strings)
        pool = pool.assign(diff=(pool["price_brl"] - price).abs(), other_type=~same_type)
        pool = pool.sort_values(["other_type", "diff"]).head(k)
        return [self._product_summary(r) for _, r in pool.iterrows()]

    # ---------- tool 1

    def search_products(self, query: str | None = None, category: str | None = None,
                        min_price: float | None = None, max_price: float | None = None,
                        only_available: bool = True, limit: int = 10) -> dict:
        df = self.data.products
        if category:
            cat_ids = self._resolve_category(category)
            if not cat_ids:
                return {"error": f"Categoria '{category}' não encontrada.",
                        "valid_categories": self.data.categories["name"].tolist()}
            df = df[df["category_id"].isin(cat_ids)]
        if min_price is not None:
            df = df[df["final_price_brl"] >= float(min_price)]
        if max_price is not None:
            df = df[df["final_price_brl"] <= float(max_price)]
        tokens = [t for t in normalize(query).split() if t not in STOPWORDS] if query else []
        hidden: list[dict] = []
        if only_available:
            # Products that match the query exactly but are out of stock / discontinued would
            # otherwise vanish, and the model would wrongly say "we don't have that model".
            if tokens:
                unavailable = df[~df["is_available"]]
                full = unavailable["search_text"].apply(lambda text: all(t in text for t in tokens))
                hidden = [self._product_summary(r) for _, r in unavailable[full].iterrows()]
            df = df[df["is_available"]]

        partial = False
        if tokens:
            hits = df["search_text"].apply(lambda text: sum(t in text for t in tokens))
            exact = df[hits == len(tokens)]
            if exact.empty:  # fall back to best partial matches
                partial = True
                df = df.assign(_hits=hits)[hits > 0].sort_values("_hits", ascending=False)
            else:
                df = exact

        df = df.sort_values("final_price_brl") if not partial else df
        limit = max(1, min(int(limit), 20))
        result = {
            "total_found": int(len(df)),
            "partial_match": partial,
            "products": [self._product_summary(r) for _, r in df.head(limit).iterrows()],
        }
        if hidden:
            result["unavailable_matches"] = hidden
            result["note"] = ("Os produtos em unavailable_matches EXISTEM no catálogo, mas estão indisponíveis. "
                              "Informe isso ao cliente e ofereça alternativas disponíveis.")
        return result

    # ---------- tool 2

    def get_product(self, product: str | int) -> dict:
        p = self.data.products
        row = None
        if str(product).strip().isdigit() and int(product) in p.index:
            row = p.loc[int(product)]
        else:
            tokens = normalize(product).split()
            key = compact(product)
            matches = p[p["name_norm"].apply(lambda n: all(t in n for t in tokens))] if tokens else p.iloc[0:0]
            if matches.empty and key:
                matches = p[p["name_compact"].str.contains(key, regex=False)]
            if len(matches) == 1:
                row = matches.iloc[0]
            elif len(matches) > 1:
                return {"multiple_matches": True,
                        "message": "Mais de um produto corresponde. Pergunte ao cliente qual deles.",
                        "products": [self._product_summary(r) for _, r in matches.head(8).iterrows()]}
            else:
                close = difflib.get_close_matches(normalize(product), p["name_norm"].tolist(), n=3, cutoff=0.4)
                return {"found": False,
                        "message": "Produto não encontrado no catálogo.",
                        "did_you_mean": [p[p["name_norm"] == c].iloc[0]["name"] for c in close]}

        result = self._product_summary(row)
        result.update({
            "found": True,
            "description": row["description"],
            "specs": row["specs"],
        })
        if pd.notna(row["discount_percent"]):
            result["promotion"] = {
                "name": row["promotion_name"],
                "discount_percent": int(row["discount_percent"]),
                "original_price_brl": _money(row["price_brl"]),
                "promo_price_brl": _money(row["promo_price_brl"]),
                "note": "Desconto PIX não é cumulativo com promoção.",
            }
        else:
            result["pix_price_brl"] = _money(row["pix_price_brl"])

        n = max_installments(row["final_price_brl"])
        result["payment"] = {
            "max_installments_no_interest": n,
            "installment_value_brl": money(row["final_price_brl"], 1 / n),
        }
        if not row["is_available"]:
            result["alternatives"] = self._alternatives(row)
        return result

    # ---------- tool 3

    def get_order_status(self, order_id: int | str, email_or_phone: str) -> dict:
        # Same generic message for "not found" and "wrong identity" so nobody can
        # discover whether an order number exists (LGPD / privacy).
        denied = {"found": False,
                  "message": "Não foi possível localizar um pedido com esses dados. "
                             "Peça ao cliente para conferir o número do pedido e o e-mail/telefone do cadastro."}
        try:
            order_id = int(str(order_id).strip().lstrip("#"))
        except ValueError:
            return denied
        orders, customers = self.data.orders, self.data.customers
        if order_id not in orders.index:
            return denied
        order = orders.loc[order_id]
        customer = customers.loc[order["customer_id"]]

        ident = str(email_or_phone or "").strip().lower()
        if "@" in ident:
            ok = ident == customer["email_norm"]
        else:
            digits = "".join(ch for ch in ident if ch.isdigit())
            ok = len(digits) >= 8 and customer["phone_digits"].endswith(digits)
        if not ok:
            return denied

        items = self.data.order_items[self.data.order_items["order_id"] == order_id]
        products = self.data.products
        result = {
            "found": True,
            "order_id": order_id,
            "customer_first_name": customer["name"].split()[0],
            "customer_city": customer["city"],
            "order_date": str(order["order_date"]),
            "status": ORDER_STATUS_LABELS.get(order["status"], order["status"]),
            "items": [{"name": products.loc[i.product_id, "name"], "quantity": int(i.quantity)}
                      for i in items.itertuples()],
            "total_brl": _money(order["total_brl"]),
            "payment_method": _payment_label(order["payment_method"]),
            "tracking_code": order["tracking_code"],
            "estimated_delivery": str(order["estimated_delivery"]) if pd.notna(order["estimated_delivery"]) else None,
            "notes": order["notes"],
            "today": str(self.today),
        }
        # ASSUMPTION: there is no real delivery date in the data, so for delivered orders
        # estimated_delivery is used as the receipt date (needed for the 7/30-day policies).
        if order["status"] == "delivered" and pd.notna(order["estimated_delivery"]):
            received = order["estimated_delivery"]
            days_received = (self.today - received).days
            days_bought = (self.today - order["order_date"]).days
            result["delivery_date_assumed"] = str(received)
            result["delivery_date_note"] = ("Data de recebimento = previsão de entrega (o sistema não registra a "
                                            "data real). A data do pedido é a data da compra, não da entrega.")
            result["days_since_delivery"] = days_received
            # Deadlines from the policy manual, computed here so the model never does date math:
            # 4.1 arrependimento (7 days after receipt), 4.3 preference exchange (7 days),
            # 4.2 defect exchange (30 days after purchase), 8.1 legal warranty (90 days after receipt).
            result["policy_deadlines"] = {
                "arrependimento_7_dias_apos_recebimento": days_received <= 7,
                "troca_por_preferencia_7_dias": days_received <= 7,
                "troca_por_defeito_30_dias_apos_compra": days_bought <= 30,
                "garantia_legal_90_dias_apos_recebimento": days_received <= 90,
            }
        if order["status"] in ("pending", "confirmed"):
            days_open = (self.today - order["order_date"]).days
            result["days_since_order"] = days_open
            if days_open > 15:
                result["needs_human_followup"] = True
                result["followup_reason"] = (f"Pedido parado há {days_open} dias sem avançar de status. "
                                             "Encaminhar para a equipe (retorno em até 24h úteis).")
        return result

    # ---------- tool 4

    def list_active_promotions(self) -> dict:
        p = self.data.products
        promo = p[p["discount_percent"].notna()].sort_values("discount_percent", ascending=False)
        return {
            "count": int(len(promo)),
            "promotions": [{
                "product_id": int(r["product_id"]),
                "name": r["name"],
                "promotion": r["promotion_name"],
                "discount_percent": int(r["discount_percent"]),
                "original_price_brl": _money(r["price_brl"]),
                "promo_price_brl": _money(r["promo_price_brl"]),
                "stock_quantity": int(r["stock_quantity"]),
                "available": bool(r["is_available"]),
            } for _, r in promo.iterrows()],
        }

    # ---------- dispatcher

    def registry(self) -> dict[str, Callable[..., dict]]:
        tools = {
            "search_products": self.search_products,
            "get_product": self.get_product,
            "get_order_status": self.get_order_status,
            "list_active_promotions": self.list_active_promotions,
        }
        if self.policies is not None:
            tools["search_policies"] = self.policies.search_policies
        return tools

    def schemas(self) -> list[dict]:
        return tool_schemas(self.data.categories["name"].tolist(), include_policies=self.policies is not None)

    def execute(self, name: str, arguments: str | dict | None) -> str:
        """Run a tool by name and always return a JSON string (errors included)."""
        fn = self.registry().get(name)
        if fn is None:
            return json.dumps({"error": f"Ferramenta desconhecida: {name}"}, ensure_ascii=False)
        try:
            args = json.loads(arguments) if isinstance(arguments, str) and arguments else (arguments or {})
            args = {k: v for k, v in args.items() if v is not None}
            return json.dumps(fn(**args), ensure_ascii=False, default=str)
        except Exception as exc:  # the model must see the error and recover, not crash the chat
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)


# --------------------------------------------------------------------------- schemas (OpenAI format)

POLICY_SCHEMA = {"type": "function", "function": {
    "name": "search_policies",
    "description": "Busca trechos do manual de políticas da loja: trocas, devoluções, arrependimento, garantia, "
                   "frete e prazos de entrega, rastreamento, formas de pagamento e parcelamento, regras de promoção, "
                   "privacidade (LGPD). Use sempre que precisar citar uma regra.",
    "parameters": {"type": "object", "properties": {
        "question": {"type": "string", "description": "A dúvida do cliente, reescrita de forma objetiva."},
    }, "required": ["question"]},
}}


def tool_schemas(category_names: list[str], include_policies: bool = False) -> list[dict]:
    schemas = [
        {"type": "function", "function": {
            "name": "search_products",
            "description": "Busca produtos no catálogo com filtros. Use para perguntas como "
                           "'quais violões até R$1000?' ou 'tem ukulele elétrico?'. Retorna preço e estoque reais.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Palavras-chave (marca, modelo, característica). Ex.: 'nylon', 'Yamaha', 'elétrico'."},
                "category": {"type": "string", "enum": category_names, "description": "Categoria do produto."},
                "min_price": {"type": "number", "description": "Preço mínimo em R$."},
                "max_price": {"type": "number", "description": "Preço máximo em R$."},
                "only_available": {"type": "boolean", "description": "Se true (padrão), só produtos com estoque."},
                "limit": {"type": "integer", "description": "Máximo de resultados (padrão 10)."},
            }},
        }},
        {"type": "function", "function": {
            "name": "get_product",
            "description": "Detalhes de UM produto: preço, preço PIX, promoção ativa, estoque, parcelamento, "
                           "especificações e alternativas se estiver indisponível. Use sempre que o cliente perguntar "
                           "preço ou disponibilidade de um modelo específico.",
            "parameters": {"type": "object", "properties": {
                "product": {"type": "string", "description": "Nome do produto (ex.: 'Takamine GD20') ou ID."},
            }, "required": ["product"]},
        }},
        {"type": "function", "function": {
            "name": "get_order_status",
            "description": "Consulta status, itens, rastreio e previsão de entrega de um pedido. "
                           "Exige o número do pedido E o e-mail ou telefone do cadastro. Se faltar algum, peça ao cliente antes.",
            "parameters": {"type": "object", "properties": {
                "order_id": {"type": "integer", "description": "Número do pedido."},
                "email_or_phone": {"type": "string", "description": "E-mail ou telefone do cliente."},
            }, "required": ["order_id", "email_or_phone"]},
        }},
        {"type": "function", "function": {
            "name": "list_active_promotions",
            "description": "Lista as promoções ATIVAS agora, com preço original, % de desconto e preço promocional.",
            "parameters": {"type": "object", "properties": {}},
        }},
    ]
    return schemas + ([POLICY_SCHEMA] if include_policies else [])
