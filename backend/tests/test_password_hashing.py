"""Password hashing, after passlib was dropped.

passlib 1.7.4 cannot initialise its bcrypt backend against bcrypt 5.x: its
one-time wrap-bug detection hashes a probe secret longer than bcrypt's 72-byte
limit, and bcrypt 5 raises instead of truncating. The exception escaped
``hash_password``, so every user created with a password died with a 500 and
``/auth/login`` could never succeed. Only the API-key path worked, which kept
this invisible until the route-level isolation tests.

These tests pin the replacement: same algorithm, same stored format, no
exception path for hostile input.
"""

from __future__ import annotations

import pytest

from app.core.errors import ValidationError
from app.core.security import hash_password, verify_password

# Pinned literally on purpose: this is the contract callers see, not an
# internal setting to track.
BCRYPT_MAX_SECRET_BYTES = 72
BCRYPT_COST = "12"


def test_a_password_verifies_against_its_own_hash():
    stored = hash_password("correct horse")
    assert verify_password("correct horse", stored) is True


def test_a_wrong_password_does_not():
    stored = hash_password("correct horse")
    assert verify_password("correct hors", stored) is False


def test_hashes_are_salted_and_still_all_verify():
    first = hash_password("same secret")
    second = hash_password("same secret")
    assert first != second
    assert verify_password("same secret", first)
    assert verify_password("same secret", second)


def test_the_stored_format_matches_what_passlib_wrote():
    """Rows already in a database keep working: bcrypt's $2b$ ident and the
    cost passlib used."""
    stored = hash_password("legacy reader")
    assert stored.startswith(f"$2b${BCRYPT_COST}$")

    salt = f"$2b${BCRYPT_COST}$"
    assert len(stored) == len(salt) + 22 + 31


def test_a_user_without_a_password_cannot_be_logged_in():
    """``password_hash=""`` is what user creation stores when no password was
    given; it must never authenticate anything, including an empty request."""
    assert verify_password("", "") is False
    assert verify_password("anything", "") is False


def test_a_malformed_hash_is_rejected_not_crashed():
    for junk in ["not-a-hash", "$2b$12$" + "x" * 53, "💡💡💡", " "]:
        assert verify_password("whatever", junk) is False


def test_non_ascii_passwords_work():
    stored = hash_password("sécret-🔐-2026")
    assert verify_password("sécret-🔐-2026", stored)
    assert verify_password("sécret-🔐-2025", stored) is False


def test_the_byte_limit_is_inclusive():
    password = "a" * BCRYPT_MAX_SECRET_BYTES
    stored = hash_password(password)
    assert verify_password(password, stored) is True


def test_an_over_long_password_is_refused_instead_of_truncated():
    """Silently folding to 72 bytes would make two different passphrases the
    same credential; bcrypt 5 raises, and the API must report that as a
    validation error rather than a 500."""
    too_long = "a" * (BCRYPT_MAX_SECRET_BYTES + 1)
    with pytest.raises(ValidationError, match="72 bytes"):
        hash_password(too_long)
    # On the login side there is nothing to match against.
    assert verify_password(too_long, hash_password("a" * BCRYPT_MAX_SECRET_BYTES)) is False


def test_a_multibyte_password_is_measured_in_bytes_not_characters():
    # 36 characters, 72 bytes.
    assert hash_password("é" * 36)
    with pytest.raises(ValidationError):
        hash_password("é" * 37)
