"""Tests for API-key → user-id resolution."""

import pytest

from classify_v2.app.auth.api_key_resolver import (
    hash_api_key,
    resolve_user_id_from_api_key,
)


class TestResolveUserIdFromApiKey:
    async def test_resolves_owner_for_a_known_key(
        self, in_memory_application_database
    ):
        # GIVEN a stored key whose sha256 hash maps to a user
        givenPlaintextKey = "tk_live_abcdef123456"
        givenOwnerUserId = "firebase-uid-owner"
        await in_memory_application_database["api_keys"].insert_one(
            {
                "key_id": "k1",
                "user_id": givenOwnerUserId,
                "key_hash": hash_api_key(givenPlaintextKey),
                "revoked": False,
            }
        )
        expectedUserId = givenOwnerUserId

        # WHEN the plaintext key is resolved
        resolvedUserId = await resolve_user_id_from_api_key(
            in_memory_application_database, givenPlaintextKey
        )

        # THEN the owning user id is returned
        assert resolvedUserId == expectedUserId

    async def test_returns_none_for_an_unknown_key(
        self, in_memory_application_database
    ):
        # GIVEN no matching key stored
        givenUnknownKey = "tk_live_not_in_db"
        expectedUserId = None

        # WHEN the key is resolved
        resolvedUserId = await resolve_user_id_from_api_key(
            in_memory_application_database, givenUnknownKey
        )

        # THEN nothing is resolved
        assert resolvedUserId is expectedUserId

    async def test_ignores_revoked_keys(self, in_memory_application_database):
        # GIVEN a stored key that has been revoked
        givenPlaintextKey = "tk_live_revoked"
        await in_memory_application_database["api_keys"].insert_one(
            {
                "key_id": "k2",
                "user_id": "firebase-uid-owner",
                "key_hash": hash_api_key(givenPlaintextKey),
                "revoked": True,
            }
        )
        expectedUserId = None

        # WHEN the revoked key is resolved
        resolvedUserId = await resolve_user_id_from_api_key(
            in_memory_application_database, givenPlaintextKey
        )

        # THEN it does not resolve to the owner
        assert resolvedUserId is expectedUserId

    def test_hash_matches_creation_time_sha256(self):
        # GIVEN a plaintext key
        givenPlaintextKey = "tk_live_stable"
        # AND the sha256-hex the api_keys service computes at creation time
        import hashlib

        expectedHash = hashlib.sha256(
            givenPlaintextKey.encode("utf-8")
        ).hexdigest()

        # WHEN the resolver hashes the same key
        actualHash = hash_api_key(givenPlaintextKey)

        # THEN the two hashes match, so lookups line up with stored hashes
        assert actualHash == expectedHash
