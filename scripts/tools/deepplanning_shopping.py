"""
DeepPlanning Shopping Tools — ported from DeepPlanning benchmark.

13 domain-specific shopping tools that operate on per-case product databases.
Each tool is instantiated per-task with the case's database directory path.

Tools:
  1. search_products — BM25 semantic search on products
  2. filter_by_brand — Filter products by brand name(s)
  3. filter_by_color — Filter products by color(s)
  4. filter_by_size — Filter products by size(s)
  5. filter_by_applicable_coupons — Filter products by coupon applicability
  6. filter_by_range — Filter products by numeric field range
  7. sort_products — Sort products by a dimension
  8. get_product_details — Get full product info by IDs
  9. calculate_transport_time — Estimate delivery time
  10. get_user_info — Get user profile
  11. add_product_to_cart — Add product to cart
  12. delete_product_from_cart — Remove product from cart
  13. get_cart_info — Get current cart state
  14. add_coupon_to_cart — Add coupon to cart (level 3)
  15. delete_coupon_from_cart — Remove coupon from cart (level 3)
"""

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import Tool

logger = logging.getLogger(__name__)

# BM25 score threshold
BM25_SCORE_THRESHOLD = 2

# ===========================================================================
# Transport time calculation — region-based distance model (from original)
# ===========================================================================

# Province aliases for normalization (pinyin, abbreviations, Chinese characters)
PROVINCE_ALIASES = {
    'beijing': ['beijing', 'bj', '北京'],
    'shanghai': ['shanghai', 'sh', '上海'],
    'tianjin': ['tianjin', 'tj', '天津'],
    'chongqing': ['chongqing', 'cq', '重庆'],
    'hebei': ['hebei', 'ji', '河北'],
    'shanxi': ['shanxi', 'jin', '山西'],
    'liaoning': ['liaoning', 'liao', '辽宁'],
    'jilin': ['jilin', 'ji_ln', '吉林'],
    'heilongjiang': ['heilongjiang', 'hei', '黑龙江'],
    'jiangsu': ['jiangsu', 'su', '江苏'],
    'zhejiang': ['zhejiang', 'zhe', '浙江'],
    'anhui': ['anhui', 'wan', '安徽'],
    'fujian': ['fujian', 'min', '福建'],
    'jiangxi': ['jiangxi', 'gan', '江西'],
    'shandong': ['shandong', 'lu', '山东'],
    'henan': ['henan', 'yu', '河南'],
    'hubei': ['hubei', 'e', '湖北'],
    'hunan': ['hunan', 'xiang', '湖南'],
    'guangdong': ['guangdong', 'yue', 'gd', '广东'],
    'hainan': ['hainan', 'qiong', '海南'],
    'sichuan': ['sichuan', 'chuan', 'shu', '四川'],
    'guizhou': ['guizhou', 'qian', 'gui_gz', '贵州'],
    'yunnan': ['yunnan', 'yun', 'dian', '云南'],
    'shaanxi': ['shaanxi', 'shan', 'qin', '陕西'],
    'gansu': ['gansu', 'gan_gs', '甘肃'],
    'qinghai': ['qinghai', 'qing', '青海'],
    'inner mongolia': ['inner mongolia', 'neimenggu', 'meng', '内蒙古'],
    'guangxi': ['guangxi', 'gui', '广西'],
    'tibet': ['tibet', 'xizang', 'zang', '西藏'],
    'ningxia': ['ningxia', 'ning', '宁夏'],
    'xinjiang': ['xinjiang', 'xin', '新疆'],
    'hongkong': ['hongkong', 'hk', 'xianggang', '香港'],
    'macau': ['macau', 'mo', 'aomen', '澳门'],
    'taiwan': ['taiwan', 'tw', '台湾'],
}

# Reverse mapping: alias -> standard province name
PROVINCE_NORMALIZATION_MAP = {
    alias: std_name for std_name, aliases in PROVINCE_ALIASES.items() for alias in aliases
}

# Region code mapping for each province
REGION_MAP = {
    'beijing': 'NC', 'tianjin': 'NC', 'hebei': 'NC', 'shanxi': 'NC', 'inner mongolia': 'NC',
    'liaoning': 'NE', 'jilin': 'NE', 'heilongjiang': 'NE',
    'shanghai': 'EC', 'jiangsu': 'EC', 'zhejiang': 'EC', 'anhui': 'EC', 'fujian': 'EC', 'jiangxi': 'EC', 'shandong': 'EC',
    'henan': 'CC', 'hubei': 'CC', 'hunan': 'CC',
    'guangdong': 'SC', 'guangxi': 'SC', 'hainan': 'SC', 'hongkong': 'SC', 'macau': 'SC', 'taiwan': 'SC',
    'sichuan': 'SW', 'chongqing': 'SW', 'guizhou': 'SW', 'yunnan': 'SW', 'tibet': 'SW',
    'shaanxi': 'NW', 'gansu': 'NW', 'qinghai': 'NW', 'ningxia': 'NW', 'xinjiang': 'NW',
}

# Region-to-region base delivery days
BASE_REGION_TIME = {
    'NC': {'NC': 1, 'NE': 2, 'EC': 2, 'CC': 2, 'SC': 3, 'SW': 3, 'NW': 3},
    'NE': {'NC': 2, 'NE': 1, 'EC': 3, 'CC': 3, 'SC': 4, 'SW': 4, 'NW': 4},
    'EC': {'NC': 2, 'NE': 3, 'EC': 1, 'CC': 2, 'SC': 2, 'SW': 3, 'NW': 4},
    'CC': {'NC': 2, 'NE': 3, 'EC': 2, 'CC': 1, 'SC': 2, 'SW': 2, 'NW': 3},
    'SC': {'NC': 3, 'NE': 4, 'EC': 2, 'CC': 2, 'SC': 1, 'SW': 3, 'NW': 4},
    'SW': {'NC': 3, 'NE': 4, 'EC': 3, 'CC': 2, 'SC': 3, 'SW': 1, 'NW': 3},
    'NW': {'NC': 3, 'NE': 4, 'EC': 4, 'CC': 3, 'SC': 4, 'SW': 3, 'NW': 1},
}

# Remote provinces that get +2 day penalty
REMOTE_PROVINCES = {'tibet', 'xinjiang', 'qinghai', 'inner mongolia'}

# Provider-specific delivery day modifiers
PROVIDER_MODIFIERS = {
    'sf express': -2, 'sf': -2,
    'jd logistics': -1, 'jd': -1,
    'yto express': 1, 'yto': 0,
    'zto express': 1, 'zto': 0,
    'sto express': 1, 'sto': 0,
    'yunda express': 1, 'yunda': 0,
    'cainiao': 1,
    'china post': 2,
    'ems': 0,
    'deppon express': 0, 'deppon': 0,
    'default': 0,
}

# Valid coupon names (from original benchmark)
VALID_COUPONS = [
    "Cross-store: ¥30 off every ¥300",
    "Cross-store: ¥60 off every ¥500",
    "Cross-store: ¥120 off every ¥900",
    "Cross-store: ¥200 off every ¥1,200",
    "Cross-store: ¥300 off every ¥1,500",
    "Same-brand: ¥25 off every ¥200",
    "Same-brand: ¥60 off every ¥400",
    "Same-brand: ¥180 off every ¥1,000",
    "Same-brand: ¥300 off every ¥1,200",
    "VIP: ¥200 off every ¥1,000",
]


def _normalize_province(address_str: str) -> Optional[str]:
    """Normalize address string to standard province name."""
    if not address_str:
        return None
    processed = (
        address_str.lower()
        .replace(' ', '')
        .replace('province', '')
        .replace('city', '')
    )
    # Exact alias match
    if processed in PROVINCE_NORMALIZATION_MAP:
        return PROVINCE_NORMALIZATION_MAP[processed]
    # Substring match
    for alias, std_name in PROVINCE_NORMALIZATION_MAP.items():
        if alias in processed:
            return std_name
    return None


def _normalize_provider(provider: str) -> str:
    """Normalize a shipping-provider name to a PROVIDER_MODIFIERS key.

    PROVIDER_MODIFIERS is keyed by the space form ('sf express'), which is how
    the product data spells it ("SF Express"), but the tool's own schema tells
    the agent to pass the underscore form ('sf_express'). Without this, the
    lookup missed and every provider silently fell back to `default` (0), so
    the SF/JD speed-up never applied for an agent that followed the docs.
    """
    if not provider:
        return 'default'
    normalized = re.sub(r'[\s_-]+', ' ', str(provider).strip().lower())
    return normalized or 'default'


def _parse_coupon(coupon_name: str):
    """Parse coupon name → (discount, threshold). Returns (None, None) on failure."""
    import re
    match = re.search(r'¥([\d,]+)\s+off\s+every\s+¥([\d,]+)', coupon_name)
    if not match:
        return None, None
    try:
        discount = float(match.group(1).replace(',', ''))
        threshold = float(match.group(2).replace(',', ''))
        return discount, threshold
    except ValueError:
        return None, None


def _load_products_jsonl(path: str) -> List[Dict]:
    """Load products from a JSONL file."""
    products = []
    if not os.path.exists(path):
        return products
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                products.append(json.loads(line))
    return products


def _load_json(path: str) -> Any:
    """Load a JSON file."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Any) -> None:
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _get_nested_value(obj: dict, key_path: str) -> Any:
    """Get nested dict value by dot-separated key path. e.g. 'sales_volume.monthly'."""
    keys = key_path.split(".")
    current = obj
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current


def _fmt(result: Any) -> str:
    """Format result as JSON string."""
    return json.dumps(result, ensure_ascii=False, indent=2)


# ===========================================================================
# Shared database context — instantiated per-case
# ===========================================================================

class ShoppingDatabase:
    """Holds the per-case product database, user info, and cart state."""

    def __init__(self, database_dir: str, cart_path: Optional[str] = None):
        self.database_dir = database_dir
        self.products: List[Dict] = _load_products_jsonl(
            os.path.join(database_dir, "products.jsonl")
        )
        self.product_index: Dict[str, Dict] = {
            p["product_id"]: p for p in self.products
        }
        self.user_info: Dict = _load_json(
            os.path.join(database_dir, "user_info.json")
        )
        # Cart is run-mutable state. Default to database_dir/cart.json for
        # backward compat, but callers should pass a per-(case, rollout)
        # cart_path so concurrent rollouts don't race on a shared file (and the
        # source dataset dir stays pristine).
        self.cart_path = cart_path or os.path.join(database_dir, "cart.json")
        # Initialize empty cart
        self._reset_cart()

        # BM25 index (lazy)
        self._bm25 = None
        self._bm25_ready = False

    def _reset_cart(self) -> None:
        """Reset cart to empty state."""
        cart = {"items": [], "used_coupons": [], "summary": {"total_items_count": 0, "total_price": 0}}
        _save_json(self.cart_path, cart)

    def load_cart(self) -> Dict:
        return _load_json(self.cart_path)

    def save_cart(self, cart: Dict) -> None:
        _save_json(self.cart_path, cart)

    def get_bm25(self):
        """Build BM25 index on first access."""
        if self._bm25_ready:
            return self._bm25
        self._bm25_ready = True
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed. search_products will not work.")
            return None

        corpus = []
        for p in self.products:
            text = " ".join([
                p.get("brand", ""), p.get("color", ""), p.get("size", ""),
                p.get("thickness", ""), p.get("elasticity", ""),
                p.get("version_type", ""), p.get("collar_type", ""),
                p.get("suitable_season", ""), p.get("target_demographic", ""),
                p.get("name", ""),
            ])
            corpus.append(text)
        if corpus:
            tokenized = [doc.lower().split() for doc in corpus]
            self._bm25 = BM25Okapi(tokenized)
        return self._bm25

    def filter_products(self, product_ids: Optional[List[str]] = None) -> List[Dict]:
        """Return products matching the given IDs, or all if None."""
        if product_ids is None:
            return list(self.products)
        return [self.product_index[pid] for pid in product_ids if pid in self.product_index]


# ===========================================================================
# Tool classes — all share a ShoppingDatabase reference
# ===========================================================================

class _ShoppingToolBase(Tool):
    """Base for all shopping tools. Holds a reference to the shared database."""
    output_type = "string"

    def __init__(self, db: ShoppingDatabase):
        self._db = db
        super().__init__()

    def forward(self, **kwargs) -> str:
        raise NotImplementedError


class SearchProductsTool(_ShoppingToolBase):
    name = "search_products"
    description = (
        "Handles broad, open-ended natural language queries. Performs semantic BM25 search "
        "on key product information (name, brand, category, tags) to retrieve relevant products."
    )
    inputs = {
        "query": {"type": "string", "description": "Natural language query, e.g. 'Nike running shoes'."},
        "limit": {"type": "integer", "description": "Max results to return (default 20).", "nullable": True},
    }

    def forward(self, query: str, limit: int = 20) -> str:
        bm25 = self._db.get_bm25()
        if not bm25:
            return _fmt({"error": "BM25 index not available."})
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        results = []
        for i, score in enumerate(scores):
            if score > BM25_SCORE_THRESHOLD:
                results.append({
                    "product_id": self._db.products[i]["product_id"],
                    "name": self._db.products[i]["name"],
                    "score": score,
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return _fmt({"product_ids": [r["product_id"] for r in results[:limit]]})


class FilterByBrandTool(_ShoppingToolBase):
    name = "filter_by_brand"
    description = "Filters product IDs by one or more brand names (OR logic)."
    inputs = {
        "product_ids": {"type": "array", "description": "Product IDs to filter (optional, filters all if omitted).", "nullable": True},
        "brand_names": {"type": "array", "description": "Brand names to match, e.g. ['Nike', 'ZARA']."},
    }

    def forward(self, brand_names: list, product_ids: list = None) -> str:
        products = self._db.filter_products(product_ids)
        brand_set = {b.lower() for b in brand_names}
        matched = [p["product_id"] for p in products if p.get("brand", "").lower() in brand_set]
        return _fmt({"filtered_products_ids": matched})


class FilterByColorTool(_ShoppingToolBase):
    name = "filter_by_color"
    description = "Filters product IDs by one or more colors (OR logic)."
    inputs = {
        "product_ids": {"type": "array", "description": "Product IDs to filter (optional).", "nullable": True},
        "colors": {"type": "array", "description": "Colors to match, e.g. ['White', 'Black']."},
    }

    def forward(self, colors: list, product_ids: list = None) -> str:
        products = self._db.filter_products(product_ids)
        color_set = {c.lower() for c in colors}
        matched = [p["product_id"] for p in products if p.get("color", "").lower() in color_set]
        return _fmt({"filtered_products_ids": matched})


class FilterBySizeTool(_ShoppingToolBase):
    name = "filter_by_size"
    description = "Filters product IDs by one or more sizes (OR logic)."
    inputs = {
        "product_ids": {"type": "array", "description": "Product IDs to filter (optional).", "nullable": True},
        "sizes": {"type": "array", "description": "Sizes to match, e.g. ['M', 'L', '42']."},
    }

    def forward(self, sizes: list, product_ids: list = None) -> str:
        products = self._db.filter_products(product_ids)
        size_set = {s.lower() for s in sizes}
        matched = [p["product_id"] for p in products if p.get("size", "").lower() in size_set]
        return _fmt({"filtered_products_ids": matched})


class FilterByApplicableCouponsTool(_ShoppingToolBase):
    name = "filter_by_applicable_coupons"
    description = (
        "Filters product IDs by applicable coupons. Returns products whose "
        "applicable_coupons list contains ALL the provided coupon names."
    )
    inputs = {
        "product_ids": {"type": "array", "description": "Product IDs to filter (optional).", "nullable": True},
        "coupon_names": {"type": "array", "description": "Coupon names to match."},
    }

    def forward(self, coupon_names: list, product_ids: list = None) -> str:
        if not coupon_names:
            return _fmt({"filtered_products_ids": []})
        # Validate coupon names
        for cn in coupon_names:
            if cn not in VALID_COUPONS:
                return _fmt({"error": f"Invalid coupon name: '{cn}'. Valid coupons are: {', '.join(VALID_COUPONS)}"})
        products = self._db.filter_products(product_ids)
        coupon_set = set(coupon_names)
        matched = []
        for p in products:
            product_coupons = set(p.get("applicable_coupons", []))
            if coupon_set.issubset(product_coupons):
                matched.append(p["product_id"])
        return _fmt({"filtered_products_ids": matched})


class FilterByRangeTool(_ShoppingToolBase):
    name = "filter_by_range"
    description = "Filters product IDs by a numeric field, operator, and threshold."
    inputs = {
        "product_ids": {"type": "array", "description": "Product IDs to filter (optional).", "nullable": True},
        "condition_key": {"type": "string", "description": "Numeric field: price, stock_quantity, sales_volume.monthly, sales_volume.total, rating.average_score, rating.total_reviews."},
        "operator": {"type": "string", "description": "Comparison operator: >, >=, <, <=, ==."},
        "value": {"type": "number", "description": "Threshold value."},
    }

    def forward(self, condition_key: str, operator: str, value: float, product_ids: list = None) -> str:
        products = self._db.filter_products(product_ids)
        ops = {
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
        }
        if operator not in ops:
            return _fmt({"error": f"Invalid operator: {operator}"})
        op_fn = ops[operator]
        matched = []
        for p in products:
            val = _get_nested_value(p, condition_key)
            if val is not None:
                try:
                    if op_fn(float(val), float(value)):
                        matched.append(p["product_id"])
                except (ValueError, TypeError):
                    pass
        return _fmt({"filtered_products_ids": matched})


class SortProductsTool(_ShoppingToolBase):
    name = "sort_products"
    description = "Sorts product IDs by a specified dimension and order."
    inputs = {
        "product_ids": {"type": "array", "description": "Product IDs to sort (optional).", "nullable": True},
        "sort_by": {"type": "string", "description": "Sort dimension: price, sales_volume.monthly, sales_volume.total, rating.average_score, stock_quantity, name."},
        "order": {"type": "string", "description": "Sort order: asc or desc (default desc).", "nullable": True},
    }

    def forward(self, sort_by: str, product_ids: list = None, order: str = "desc") -> str:
        sort_by = sort_by.lower()
        products = self._db.filter_products(product_ids)
        reverse = (order != "asc")

        def sort_key(p):
            val = _get_nested_value(p, sort_by)
            if val is None:
                return float("-inf") if reverse else float("inf")
            if sort_by == "name":
                return str(val)
            try:
                return float(val)
            except (ValueError, TypeError):
                return float("-inf") if reverse else float("inf")

        if sort_by == "name":
            products.sort(key=lambda p: str(_get_nested_value(p, sort_by) or ""), reverse=reverse)
        else:
            products.sort(key=sort_key, reverse=reverse)
        return _fmt({"sorted_product_ids": [p["product_id"] for p in products]})


class GetProductDetailsTool(_ShoppingToolBase):
    name = "get_product_details"
    description = "Retrieves complete detailed information for a list of product IDs."
    inputs = {
        "product_ids": {"type": "array", "description": "Product IDs to fetch details for."},
    }

    def forward(self, product_ids: list) -> str:
        results = [
            self._db.product_index[pid]
            for pid in product_ids
            if pid in self._db.product_index
        ]
        return _fmt({"products": results})


class CalculateTransportTimeTool(_ShoppingToolBase):
    name = "calculate_transport_time"
    description = "Calculates the estimated delivery time in days for a product. It uses the product's shipping origin and the user's destination address. The final result is an integer number of days."
    inputs = {
        "product_id": {"type": "string", "description": "The unique identifier of the product to find its shipping origin."},
        "destination_address": {"type": "string", "description": "The user's destination address, which must be the province name written in pinyin, such as 'guangdong', 'beijing', 'shanghai', etc. Only pinyin is accepted for this input parameter."},
        "provider": {"type": "string", "description": "Optional. The shipping provider name to adjust delivery time. Supported providers include 'sf_express', 'jd_logistics', 'yto_express', 'zto_express', 'sto_express', 'yunda_express', 'cainiao', 'china_post', 'ems', 'deppon_express', 'default'. If not provided, uses the product's default provider.", "nullable": True},
    }

    def forward(self, product_id: str, destination_address: str, provider: str = None) -> str:
        product = self._db.product_index.get(product_id)
        if not product:
            return _fmt({"error": f"Product with ID '{product_id}' not found."})

        # Get shipping origin from product's shipping_info
        shipping_info = product.get('shipping_info', {})
        origin_address = shipping_info.get('origin')
        # Use explicit provider if given, otherwise fall back to product's default.
        # Normalize so the schema-documented 'sf_express' and the product-data
        # "SF Express" both resolve to the same PROVIDER_MODIFIERS entry.
        provider = _normalize_provider(
            provider if provider else shipping_info.get('provider', 'default')
        )

        if not origin_address:
            return _fmt({"error": f"Shipping origin not found for product '{product_id}'."})

        # Normalize provinces
        origin_province = _normalize_province(origin_address)
        destination_province = _normalize_province(destination_address)

        if not origin_province:
            return _fmt({"error": f"Could not determine a valid province from origin address: '{origin_address}'."})
        if not destination_province:
            return _fmt({"error": f"Could not determine a valid province from destination address: '{destination_address}'. Please provide a valid Chinese province name."})

        # Region-based distance lookup
        origin_region = REGION_MAP.get(origin_province)
        dest_region = REGION_MAP.get(destination_province)

        if not origin_region or not dest_region:
            return _fmt({"error": "Could not map provinces to geographical regions."})

        base_days = BASE_REGION_TIME[origin_region][dest_region]

        # Remote province penalty (+2 days)
        if origin_province in REMOTE_PROVINCES or destination_province in REMOTE_PROVINCES:
            base_days += 2

        # Provider modifier
        modifier = PROVIDER_MODIFIERS.get(provider, PROVIDER_MODIFIERS['default'])
        final_days = max(1, base_days + modifier)

        return _fmt({
            "product_id": product_id,
            "origin": origin_address,
            "destination": destination_address,
            "estimated_delivery_days": final_days,
        })


class GetUserInfoTool(_ShoppingToolBase):
    name = "get_user_info"
    description = "Retrieves user info including address, body measurements, coupons, and profile details."
    inputs = {
        "user_id": {"type": "string", "description": "User ID (optional, returns default user if omitted).", "nullable": True},
    }

    def forward(self, user_id: str = None) -> str:
        return _fmt(self._db.user_info)


class AddProductToCartTool(_ShoppingToolBase):
    name = "add_product_to_cart"
    description = (
        "Adds a product to the shopping cart. Validates product existence and stock. "
        "If already in cart, increases quantity. Returns updated cart."
    )
    inputs = {
        "product_id": {"type": "string", "description": "Product ID to add."},
        "quantity": {"type": "integer", "description": "Quantity to add (default 1).", "nullable": True},
    }

    def forward(self, product_id: str, quantity: int = 1) -> str:
        product = self._db.product_index.get(product_id)
        if not product:
            return _fmt({"error": f"Product '{product_id}' not found in catalog"})
        if quantity <= 0:
            return _fmt({"error": "Quantity must be a positive integer"})

        stock = int(product.get("stock_quantity", 0))
        cart = self._db.load_cart()
        items = cart.get("items", [])

        # Find existing item
        existing = None
        for item in items:
            if item.get("product_id") == product_id:
                existing = item
                break

        current_qty = int(existing.get("quantity", 0)) if existing else 0
        if current_qty + quantity > stock:
            return _fmt({"error": f"Insufficient stock. Available: {stock}, In cart: {current_qty}, Requested: {quantity}"})

        if existing:
            existing["quantity"] = current_qty + quantity
            existing["price"] = product.get("price", 0)
        else:
            items.append({
                "product_id": product_id,
                "name": product.get("name", ""),
                "quantity": quantity,
                "price": product.get("price", 0),
            })

        cart["items"] = items
        self._update_cart_summary(cart)
        self._db.save_cart(cart)
        return _fmt(cart)

    def _update_cart_summary(self, cart: dict) -> None:
        items = cart.get("items", [])
        total_items = sum(int(i.get("quantity", 0)) for i in items)
        total_price = sum(float(i.get("price", 0)) * int(i.get("quantity", 0)) for i in items)
        cart["summary"] = {"total_items_count": total_items, "total_price": round(total_price, 2)}


class DeleteProductFromCartTool(_ShoppingToolBase):
    name = "delete_product_from_cart"
    description = (
        "Removes a product from the cart or reduces its quantity. "
        "If quantity to remove >= cart quantity, item is completely removed."
    )
    inputs = {
        "product_id": {"type": "string", "description": "Product ID to remove."},
        "quantity": {"type": "integer", "description": "Quantity to remove (default 1).", "nullable": True},
    }

    def forward(self, product_id: str, quantity: int = 1) -> str:
        if product_id not in self._db.product_index:
            return _fmt({"error": f"Product '{product_id}' not found in catalog"})

        cart = self._db.load_cart()
        items = cart.get("items", [])

        existing = None
        for item in items:
            if item.get("product_id") == product_id:
                existing = item
                break

        if not existing:
            return _fmt({"error": f"Product '{product_id}' is not in the cart"})

        current_qty = int(existing.get("quantity", 0))
        if quantity >= current_qty:
            items = [i for i in items if i.get("product_id") != product_id]
        else:
            existing["quantity"] = current_qty - quantity

        cart["items"] = items
        self._update_cart_summary(cart)
        self._db.save_cart(cart)
        return _fmt(cart)

    def _update_cart_summary(self, cart: dict) -> None:
        items = cart.get("items", [])
        total_items = sum(int(i.get("quantity", 0)) for i in items)
        total_price = sum(float(i.get("price", 0)) * int(i.get("quantity", 0)) for i in items)
        cart["summary"] = {"total_items_count": total_items, "total_price": round(total_price, 2)}


class GetCartInfoTool(_ShoppingToolBase):
    name = "get_cart_info"
    description = "Retrieves current shopping cart information including all items and summary statistics."
    inputs = {}

    def forward(self) -> str:
        cart = self._db.load_cart()
        return _fmt(cart)


class AddCouponToCartTool(_ShoppingToolBase):
    name = "add_coupon_to_cart"
    description = (
        "Adds a coupon to the shopping cart. Validates coupon existence, user ownership, "
        "eligibility based on cart total, and VIP status. Returns updated cart."
    )
    inputs = {
        "coupon_name": {"type": "string", "description": "Coupon name to add."},
        "quantity": {"type": "integer", "description": "Number of coupons to add (default 1).", "nullable": True},
    }

    def forward(self, coupon_name: str, quantity: int = 1) -> str:
        if not isinstance(quantity, (int, float)) or quantity <= 0:
            return _fmt({"error": "quantity must be a positive number"})
        quantity = int(quantity)

        # 1. Validate coupon exists in the valid set
        if coupon_name not in VALID_COUPONS:
            return _fmt({"error": f"Coupon not found: '{coupon_name}'. Valid coupons are: {', '.join(VALID_COUPONS)}"})

        # 2. Validate user ownership — coupons stored as {name: qty} dict
        user_coupons = self._db.user_info.get("coupons", {})
        user_owned_quantity = user_coupons.get(coupon_name, 0)

        cart = self._db.load_cart()
        used_coupons = cart.get("used_coupons", [])

        # Calculate currently used quantity
        currently_used = 0
        for coupon in used_coupons:
            if coupon.get("coupon_name") == coupon_name:
                currently_used += int(coupon.get("quantity", 0))

        total_needed = currently_used + quantity
        if total_needed > user_owned_quantity:
            return _fmt({"error": (
                f"Insufficient coupon quantity: User owns {user_owned_quantity} of '{coupon_name}', "
                f"cart already uses {currently_used}, cannot add {quantity} more"
            )})

        # 3. Check VIP status for VIP coupons
        if coupon_name.startswith("VIP:"):
            is_vip = self._db.user_info.get("is_vip", False)
            if not is_vip:
                return _fmt({"error": f"VIP coupon '{coupon_name}' requires VIP status, but user is not a VIP"})

        # 4. Update or add coupon to used_coupons
        coupon_found = False
        for coupon in used_coupons:
            if coupon.get("coupon_name") == coupon_name:
                coupon["quantity"] = total_needed
                coupon_found = True
                break
        if not coupon_found:
            used_coupons.append({"coupon_name": coupon_name, "quantity": quantity})

        # 5. Validate coupon combination — cart total must cover sum of all thresholds
        base_total = self._calculate_base_total(cart)
        is_valid, error_msg = self._validate_coupon_combination(base_total, used_coupons)
        if not is_valid:
            # Rollback
            if coupon_found:
                for coupon in used_coupons:
                    if coupon.get("coupon_name") == coupon_name:
                        coupon["quantity"] = currently_used
                        break
            else:
                used_coupons.pop()
            cart["used_coupons"] = used_coupons
            return _fmt({"error": error_msg})

        # 6. Update summary and persist
        cart["used_coupons"] = [dict(c) for c in used_coupons]
        self._update_summary(cart)
        self._db.save_cart(cart)
        return _fmt(cart)

    @staticmethod
    def _calculate_base_total(cart: dict) -> float:
        """Calculate cart base total (without coupon discounts)."""
        items = cart.get("items", [])
        total = 0.0
        for item in items:
            price = float(item.get("price") or item.get("items_price", 0.0))
            qty = int(item.get("quantity", 0))
            total += price * qty
        return round(total, 2)

    @staticmethod
    def _calculate_total_discount(used_coupons: list) -> float:
        """Calculate total discount from all used coupons."""
        total_discount = 0.0
        for coupon in used_coupons:
            c_name = coupon.get("coupon_name", "")
            c_qty = int(coupon.get("quantity", 0))
            discount, _ = _parse_coupon(c_name)
            if discount is not None:
                total_discount += discount * c_qty
        return round(total_discount, 2)

    @staticmethod
    def _validate_coupon_combination(base_total: float, used_coupons: list):
        """Check if cart total can cover the sum of all coupon thresholds."""
        coupon_usage = {}
        for coupon in used_coupons:
            c_name = coupon.get("coupon_name", "")
            c_qty = int(coupon.get("quantity", 0))
            coupon_usage[c_name] = coupon_usage.get(c_name, 0) + c_qty

        total_threshold = 0.0
        for c_name, total_qty in coupon_usage.items():
            discount, threshold = _parse_coupon(c_name)
            if discount is None or threshold is None:
                return False, f"Invalid coupon format: {c_name}"
            total_threshold += threshold * total_qty

        if base_total < total_threshold:
            return False, (
                f"Cart total {base_total} is insufficient for this combination of coupons"
                f" (requires at least {total_threshold})"
            )
        return True, ""

    def _update_summary(self, cart: dict) -> None:
        """Update cart summary — total_price is the FINAL discounted price (matching original)."""
        items = cart.get("items", [])
        total_items_count = sum(int(item.get("quantity", 0)) for item in items)
        base_total = self._calculate_base_total(cart)
        used_coupons = cart.get("used_coupons", [])
        total_discount = self._calculate_total_discount(used_coupons)
        final_price = max(0.0, base_total - total_discount)

        if "summary" not in cart:
            cart["summary"] = {}
        cart["summary"]["total_items_count"] = total_items_count
        cart["summary"]["total_price"] = round(final_price, 2)


class DeleteCouponFromCartTool(_ShoppingToolBase):
    name = "delete_coupon_from_cart"
    description = (
        "Removes a coupon from the cart or reduces its quantity. "
        "If quantity >= used quantity, coupon is completely removed. "
        "Recalculates cart total with remaining coupons."
    )
    inputs = {
        "coupon_name": {"type": "string", "description": "Coupon name to remove."},
        "quantity": {"type": "integer", "description": "Number of coupons to remove (default 1).", "nullable": True},
    }

    def forward(self, coupon_name: str, quantity: int = 1) -> str:
        if not isinstance(quantity, (int, float)) or quantity <= 0:
            return _fmt({"error": "quantity must be a positive number"})
        quantity = int(quantity)

        if coupon_name not in VALID_COUPONS:
            return _fmt({"error": f"Invalid coupon name: '{coupon_name}'. Valid coupons are: {', '.join(VALID_COUPONS)}"})

        cart = self._db.load_cart()
        used_coupons = cart.get("used_coupons", [])

        # Find the coupon — support both formats:
        #   {"coupon_name": "...", "quantity": N}  and  {"CouponName": N}
        coupon_found = False
        coupon_index = -1
        current_quantity = 0
        for idx, coupon in enumerate(used_coupons):
            if isinstance(coupon, dict):
                if len(coupon) == 1:
                    existing_name = list(coupon.keys())[0]
                    if existing_name == coupon_name:
                        coupon_found = True
                        coupon_index = idx
                        current_quantity = int(coupon.get(coupon_name, 0))
                        break
                elif "coupon_name" in coupon:
                    if coupon.get("coupon_name") == coupon_name:
                        coupon_found = True
                        coupon_index = idx
                        current_quantity = int(coupon.get("quantity", 0))
                        break

        if not coupon_found:
            return _fmt({"error": f"Coupon not in cart: '{coupon_name}'"})

        if current_quantity < quantity:
            return _fmt({"error": f"Insufficient coupon quantity in cart: Cart has {current_quantity} of '{coupon_name}', cannot remove {quantity}"})

        new_quantity = current_quantity - quantity
        if new_quantity == 0:
            used_coupons.pop(coupon_index)
        else:
            coupon = used_coupons[coupon_index]
            if len(coupon) == 1:
                coupon[list(coupon.keys())[0]] = new_quantity
            elif "coupon_name" in coupon:
                coupon["quantity"] = new_quantity

        cart["used_coupons"] = used_coupons
        # Remove zero-quantity coupons
        cart["used_coupons"] = [c for c in cart["used_coupons"] if isinstance(c, dict) and (
            (len(c) == 1 and int(list(c.values())[0]) > 0) or
            ("coupon_name" in c and int(c.get("quantity", 0)) > 0)
        )]

        # Recalculate summary with remaining coupons (total_price = final discounted price)
        items = cart.get("items", [])
        total_items = sum(int(i.get("quantity", 0)) for i in items)
        base_total = sum(float(i.get("price") or i.get("items_price", 0.0)) * int(i.get("quantity", 0)) for i in items)
        total_discount = 0.0
        for c in cart["used_coupons"]:
            if isinstance(c, dict):
                c_name = c.get("coupon_name", "") if "coupon_name" in c else list(c.keys())[0] if len(c) == 1 else ""
                c_qty = int(c.get("quantity", c.get(c_name, 0))) if "coupon_name" in c else int(list(c.values())[0]) if len(c) == 1 else 0
                discount, _ = _parse_coupon(c_name)
                if discount is not None:
                    total_discount += discount * c_qty
        final_price = max(0.0, round(base_total, 2) - round(total_discount, 2))

        if "summary" not in cart:
            cart["summary"] = {}
        cart["summary"]["total_items_count"] = total_items
        cart["summary"]["total_price"] = round(final_price, 2)

        self._db.save_cart(cart)
        return _fmt(cart)


# ===========================================================================
# Factory function to create all shopping tools for a given case directory
# ===========================================================================

def create_shopping_tools(database_dir: str, cart_path: Optional[str] = None) -> Dict[str, Tool]:
    """Create all 15 shopping tools bound to a specific case database.

    Args:
        database_dir: Path to the case directory containing products.jsonl,
                     user_info.json, validation_cases.json.
        cart_path: Optional explicit path for the run-mutable cart.json. When
                     omitted, defaults to database_dir/cart.json (legacy, shared).
                     Pass a per-(case, rollout) path to avoid concurrent races.

    Returns:
        Dict mapping tool name -> Tool instance.
    """
    db = ShoppingDatabase(database_dir, cart_path=cart_path)
    tools = {}
    tool_classes = [
        SearchProductsTool, FilterByBrandTool, FilterByColorTool,
        FilterBySizeTool, FilterByApplicableCouponsTool, FilterByRangeTool,
        SortProductsTool, GetProductDetailsTool, CalculateTransportTimeTool,
        GetUserInfoTool, AddProductToCartTool, DeleteProductFromCartTool,
        GetCartInfoTool, AddCouponToCartTool, DeleteCouponFromCartTool,
    ]
    for cls in tool_classes:
        tool = cls(db)
        tools[tool.name] = tool
    return tools
