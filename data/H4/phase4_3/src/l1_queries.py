"""L1 selection namespace; frozen H4A grammar and RNG function are unchanged."""
import h4_l0_queries as frozen

schema = frozen.schema
validate = frozen.validate
matches = frozen.matches
catalogue = frozen.catalogue
grammar_text = frozen.grammar_text
_rng = frozen._rng


def select_only_no_reveal(value, *, arm, seed, round_index, labeled_ids, bundle, carry=0):
    if arm != "L1" or type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("Unknown L1 arm/seed")
    if type(round_index) is not int or not 1 <= round_index <= 20:
        raise ValueError("Round must be1..20")
    if type(carry) is not int or carry < 0:
        raise ValueError("Carry must be nonnegative")
    labeled = list(labeled_ids)
    if len(labeled) != len(set(labeled)) or not set(labeled) <= set(bundle["pool_ids"]):
        raise ValueError("L1 selection may use only distinct already-labelled POOL identities")
    capacity = 75 + carry
    validate(value, bundle=bundle, capacity=capacity)
    remaining = set(bundle["pool_ids"]) - set(labeled)
    selected, audit = [], []
    for index, query in enumerate(value["queries"]):
        candidates = sorted(rid for rid in remaining if matches(query, bundle["by_id"][rid]))
        count = min(query["n"], len(candidates))
        chosen = _rng(arm, seed, round_index, index).sample(candidates, count)
        selected.extend(chosen)
        remaining.difference_update(chosen)
        audit.append({"query_index": index, "requested_n": query["n"],
            "matched_unlabeled_n": len(candidates), "selected_n": count, "shortfall_n": query["n"] - count})
    if len(selected) != len(set(selected)) or len(selected) > capacity:
        raise AssertionError("L1 duplicate/quota selection error")
    return {"selected_ids": selected, "carry_in": carry, "capacity": capacity,
        "selected_n": len(selected), "carry_out": capacity - len(selected), "queries": audit,
        "automatic_topup": False, "labels_revealed": False, "grammar_version": frozen.GRAMMAR_VERSION,
        "selection_condition": "L1"}


def final_fill_only_no_reveal(*, arm, seed, labeled_ids, bundle, total_new_revealed):
    if arm != "L1" or type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("Unknown L1 arm/seed")
    if type(total_new_revealed) is not int or not 0 <= total_new_revealed <= 1500:
        raise ValueError("Invalid final revealed budget")
    labeled = list(labeled_ids)
    if len(labeled) != len(set(labeled)) or not set(labeled) <= set(bundle["pool_ids"]):
        raise ValueError("Final L1 fill may use POOL only")
    if len(labeled) != 500 + total_new_revealed:
        raise ValueError("Final label ledger differs from S0+reveals")
    count = 1500 - total_new_revealed
    remaining = sorted(set(bundle["pool_ids"]) - set(labeled))
    if len(remaining) < count:
        raise ValueError("Insufficient remaining POOL for L1 fill")
    chosen = _rng(arm, seed, 21, 0, "final_uniform_fill").sample(remaining, count)
    return {"selected_ids": chosen, "selected_n": count, "labels_revealed": False, "after_round": 20,
        "extra_notes_update": False, "method": "uniform without replacement on remaining POOL; fixed L1/seed RNG"}
