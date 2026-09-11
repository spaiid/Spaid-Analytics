import hashlib, json

def stable_uid(*parts) -> str:
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(blob.encode(), digest_size=16).hexdigest()
