"""Generate the Python -> TypeScript token-hash parity fixture.

Writes ``packages/core-ts/tests/fixtures/token-hash-vectors.json``: scrypt
token hashes minted by core-py ``hull_core.auth.tokens`` (plus a few
non-default parameter sets built with ``hashlib.scrypt`` directly), the
wrong-token negatives, and the malformed encodings core-py rejects with
``TokenHashError`` (with the exact message). core-ts must verify / reject
every entry identically.

Run from the repo root::

    uv run python scripts/gen_token_hash_vectors.py

Each run mints fresh random salts, so the file changes on every run; only
regenerate when the vector set itself changes. The tokens are test values,
never real credentials.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from hull_core.auth.tokens import TokenHashError, hash_token, is_valid_encoding, verify_token

OUT = Path(__file__).resolve().parents[1] / "packages" / "core-ts" / "tests" / "fixtures" / "token-hash-vectors.json"

# Test-only tokens: ASCII, empty, embedded whitespace, Vietnamese + emoji
# (non-BMP), and a long one. Escapes keep this file ASCII.
DEFAULT_TOKENS = {
    "ascii": "s3cret-token",
    "empty": "",
    "inner-spaces": "two words here",
    "unicode": "m\u1eadt-kh\u1ea9u-Vi\u1ec7t-\U0001f511",
    "long": "x" * 1024,
}

# Non-default parameters through the same encoding (fixed salts).
CUSTOM = [
    {"name": "n16384-r8-p1", "token": "custom-a", "n": 16384, "r": 8, "p": 1, "dklen": 32, "salt": "00" * 16},
    {"name": "n1024-r4-p2", "token": "custom-b", "n": 1024, "r": 4, "p": 2, "dklen": 32, "salt": "a5" * 16},
    {"name": "dklen16", "token": "custom-c", "n": 2048, "r": 8, "p": 1, "dklen": 16, "salt": "0f" * 8},
    {"name": "dklen64", "token": "custom-d", "n": 2048, "r": 8, "p": 1, "dklen": 64, "salt": "f0" * 24},
]

# Encodings core-py must reject with TokenHashError (asserted below).
VALID_TAIL = "00112233445566778899aabbccddeeff"
MALFORMED = [
    "",
    "not-a-hash",
    "scrypt$bad$bits",
    "scrypt$x$8$1$" + VALID_TAIL + "$" + VALID_TAIL,
    "scrypt$32768$y$1$" + VALID_TAIL + "$" + VALID_TAIL,
    "scrypt$32768$8$z$" + VALID_TAIL + "$" + VALID_TAIL,
    "bcrypt$32768$8$1$" + VALID_TAIL + "$" + VALID_TAIL,
    "scrypt$32768$8$1$$" + VALID_TAIL,
    "scrypt$32768$8$1$" + VALID_TAIL + "$",
    "scrypt$32768$8$1$zz$" + VALID_TAIL,
    "scrypt$32768$8$1$abc$" + VALID_TAIL,
    "scrypt$32768$8$1$" + VALID_TAIL + "$" + VALID_TAIL + "$extra",
    'it\'s a "quoted" token',
    "tab\there\nnewline",
    "only'single",
    "na\u00efve-\u1ec7-\U0001f511-value-longer-than-24",
]


def scrypt_encode(token: str, n: int, r: int, p: int, dklen: int, salt_hex: str) -> str:
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.scrypt(token.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=dklen, maxmem=64 * 1024 * 1024)
    return "$".join(["scrypt", str(n), str(r), str(p), salt.hex(), dk.hex()])


def build() -> dict[str, object]:
    valid: list[dict[str, object]] = []
    for name, token in DEFAULT_TOKENS.items():
        encoded = hash_token(token)
        valid.append({"name": name, "token": token, "encoded": encoded})
    for case in CUSTOM:
        encoded = scrypt_encode(
            str(case["token"]), int(case["n"]), int(case["r"]), int(case["p"]), int(case["dklen"]), str(case["salt"])
        )
        valid.append({"name": case["name"], "token": case["token"], "encoded": encoded})
    # bytes.fromhex accepts uppercase hex; so must core-ts.
    first = str(valid[0]["encoded"]).split("$")
    upper = "$".join([*first[:4], first[4].upper(), first[5].upper()])
    valid.append({"name": "uppercase-hex", "token": valid[0]["token"], "encoded": upper})

    for case in valid:
        token, encoded = str(case["token"]), str(case["encoded"])
        assert verify_token(token, encoded), case["name"]
        assert is_valid_encoding(encoded), case["name"]

    wrong = []
    for case in valid:
        candidate = str(case["token"]) + "-wrong"
        assert not verify_token(candidate, str(case["encoded"]))
        wrong.append({"name": case["name"], "token": candidate, "encoded": case["encoded"]})

    malformed = []
    for encoded in MALFORMED:
        try:
            verify_token("x", encoded)
        except TokenHashError as exc:
            malformed.append({"encoded": encoded, "message": str(exc)})
            assert not is_valid_encoding(encoded)
        else:  # pragma: no cover - generator guard
            raise SystemExit(f"core-py accepted a malformed vector: {encoded!r}")

    return {
        "_comment": (
            "Python-minted scrypt token hashes (hull_core.auth.tokens) that core-ts must verify; "
            "regenerate with `uv run python scripts/gen_token_hash_vectors.py`. Test tokens only."
        ),
        "valid": valid,
        "wrong_token": wrong,
        "malformed": malformed,
    }


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
