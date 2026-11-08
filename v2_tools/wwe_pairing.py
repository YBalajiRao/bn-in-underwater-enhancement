"""Strict pair validation for manifest pairs (dicts with input_path / gt_path[/filename]). Pure logic: reads no files."""
import posixpath


def _base(p):
    return posixpath.basename(str(p).replace("\\", "/"))


def validate_pairs(pairs, expected_n, split):
    if len(pairs) != expected_n:
        raise AssertionError(f"{split}: {len(pairs)} pairs, expected {expected_n}")
    ins, gts = set(), set()
    for i, p in enumerate(pairs):
        bi, bg = _base(p["input_path"]), _base(p["gt_path"])
        if bi != bg:
            raise AssertionError(f"{split}[{i}]: input/GT basename mismatch {bi!r} != {bg!r}")
        if "filename" in p and p["filename"] != bi:
            raise AssertionError(f"{split}[{i}]: manifest filename {p['filename']!r} != basename {bi!r}")
        if str(p["input_path"]) == str(p["gt_path"]):
            raise AssertionError(f"{split}[{i}]: input path equals GT path")
        low = (str(p["input_path"]) + str(p["gt_path"])).lower()
        if "uieb" in low or "val_b" in low:
            raise AssertionError(f"{split}[{i}]: path refers to a restricted set")
        ins.add(str(p["input_path"])); gts.add(str(p["gt_path"]))
    if len(ins) != expected_n or len(gts) != expected_n:
        raise AssertionError(f"{split}: duplicate input or GT paths ({len(ins)} inputs, {len(gts)} GTs)")
    return True


def checked_loader(orig_load_pairs, expected):
    """Wrap train_v2.load_pairs: validate the FULL list of a split first, then apply the optional limit."""
    def load(key, limit=None):
        if key not in expected:  # refuse BEFORE touching the original loader, so no other manifest is ever read
            raise AssertionError(f"split {key!r} is not permitted in the WWE-UIE protocol")
        full = orig_load_pairs(key)
        validate_pairs(full, expected[key], key)
        return full[:limit] if limit else full
    return load
