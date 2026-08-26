"""DeepPlanning shopping evaluation helpers."""

from __future__ import annotations

import json
import os
from typing import Any, Dict


def evaluate_shopping_prediction(*, db_dir: str, cart_path: str = None) -> Dict[str, Any]:
    """Compare the final cart against DeepPlanning shopping ground truth.

    The cart is read from `cart_path` (the per-(case, rollout) isolated cart);
    when omitted it falls back to db_dir/cart.json (legacy shared location).
    Ground truth always comes from db_dir/validation_cases.json.
    """
    if not db_dir:
        return {"score": 0.0, "case_score": 0.0, "match_rate": 0.0, "error": "No database dir"}

    cart_path = cart_path or os.path.join(db_dir, "cart.json")
    validation_path = os.path.join(db_dir, "validation_cases.json")

    if not os.path.isfile(cart_path) or not os.path.isfile(validation_path):
        return {
            "score": 0.0,
            "case_score": 0.0,
            "match_rate": 0.0,
            "error": "Missing cart.json or validation_cases.json",
        }

    try:
        with open(cart_path, "r", encoding="utf-8") as handle:
            cart = json.load(handle)
        with open(validation_path, "r", encoding="utf-8") as handle:
            validation = json.load(handle)
    except Exception as exc:
        return {"score": 0.0, "case_score": 0.0, "match_rate": 0.0, "error": str(exc)}

    cart_items = cart.get("items", [])
    cart_product_ids = {item.get("product_id") for item in cart_items if item.get("product_id")}

    gt_products = validation.get("ground_truth_products", [])
    gt_product_ids = {product.get("product_id") for product in gt_products if product.get("product_id")}

    matched_product_ids = cart_product_ids & gt_product_ids

    gt_coupons = validation.get("ground_truth_coupons", {})
    cart_coupons = cart.get("used_coupons", [])
    matched_coupons = 0
    for coupon in cart_coupons:
        coupon_name = coupon.get("coupon_name", "")
        coupon_qty = int(coupon.get("quantity", 0))
        if coupon_name in gt_coupons and coupon_qty == gt_coupons[coupon_name]:
            matched_coupons += 1

    expected_count = len(gt_product_ids) + len(gt_coupons)
    matched_count = len(matched_product_ids) + matched_coupons
    match_rate = matched_count / expected_count if expected_count > 0 else 0.0
    case_score = 1.0 if matched_count == expected_count else 0.0

    return {
        "score": match_rate,
        "case_score": case_score,
        "match_rate": match_rate,
        "matched_products": len(matched_product_ids),
        "expected_products": len(gt_product_ids),
        "matched_coupons": matched_coupons,
        "expected_coupons": len(gt_coupons),
        "extra_in_cart": len(cart_product_ids - gt_product_ids),
    }
