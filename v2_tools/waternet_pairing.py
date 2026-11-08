"""Strict pair validation for the WaterNet protocol. Pure logic (reads no files); delegates to the project-wide validate_pairs used by the WWE-UIE protocol."""
from v2_tools.wwe_pairing import validate_pairs  # noqa: F401  (same rules: basename equality, manifest filename, distinct paths, no UIEB / val_B path, no duplicates)


def checked_loader(orig_load_pairs, expected):
    """Wrap train_v2.load_pairs: refuse every split not in `expected` BEFORE the original loader is touched, validate the FULL list, then apply the optional limit."""
    def load(key, limit=None):
        if key not in expected:
            raise AssertionError(f"split {key!r} is not permitted in the WaterNet protocol")
        full = orig_load_pairs(key)
        validate_pairs(full, expected[key], key)
        return full[:limit] if limit else full
    return load
