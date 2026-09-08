"""Tests for API-key → user-id resolution in nel-v2."""

import base64
import json

import pytest
from starlette.requests import Request

from nel.app.user_config.routes import auth
from nel.app.user_config.routes.api_key_resolver import (
    hash_api_key,
    resolve_user_id_from_api_key,
)


def _request_with_headers(headers: dict[str, str]) -> Request:
    raw_headers = [
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in headers.items()
    ]
    return Request({"type": "http", "headers": raw_headers})


async def _immediate(value):
    """Await-able that returns `value` — stand-in for get_application_db."""
    return value


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


class TestGetFirebaseUidApiKeyPath:
    async def test_api_key_resolves_to_owning_user(
        self, monkeypatch, in_memory_application_database
    ):
        # GIVEN production auth (not local) and a stored key owned by a user
        monkeypatch.setattr(auth, "TARGET_ENVIRONMENT_TYPE", "development")
        givenPlaintextKey = "tk_live_owner_key"
        givenOwnerUserId = "firebase-uid-owner"
        await in_memory_application_database["api_keys"].insert_one(
            {
                "key_id": "k1",
                "user_id": givenOwnerUserId,
                "key_hash": hash_api_key(givenPlaintextKey),
                "revoked": False,
            }
        )
        monkeypatch.setattr(
            auth.ClassifierDBProvider,
            "get_application_db",
            classmethod(lambda cls: _immediate(in_memory_application_database)),
        )
        givenRequest = _request_with_headers({"x-api-key": givenPlaintextKey})
        expectedUserId = givenOwnerUserId

        # WHEN the auth dependency resolves the uid
        resolvedUserId = await auth.get_firebase_uid(givenRequest)

        # THEN it is the key's owner, so this user's model config applies
        assert resolvedUserId == expectedUserId

    async def test_unknown_api_key_falls_back_to_shared_uid(
        self, monkeypatch, in_memory_application_database
    ):
        # GIVEN production auth and an api key with no matching owner
        monkeypatch.setattr(auth, "TARGET_ENVIRONMENT_TYPE", "development")
        monkeypatch.setattr(
            auth.ClassifierDBProvider,
            "get_application_db",
            classmethod(lambda cls: _immediate(in_memory_application_database)),
        )
        givenRequest = _request_with_headers({"x-api-key": "tk_live_unknown"})
        expectedUserId = auth._API_KEY_UID

        # WHEN the auth dependency resolves the uid
        resolvedUserId = await auth.get_firebase_uid(givenRequest)

        # THEN it falls back to the shared service uid
        assert resolvedUserId == expectedUserId

    async def test_firebase_userinfo_takes_precedence_over_api_key(
        self, monkeypatch
    ):
        # GIVEN production auth and a Firebase user-info header present
        monkeypatch.setattr(auth, "TARGET_ENVIRONMENT_TYPE", "development")
        givenFirebaseUid = "firebase-session-uid"
        userinfo = base64.b64encode(
            json.dumps({"user_id": givenFirebaseUid}).encode()
        ).decode()
        givenRequest = _request_with_headers(
            {
                "x-apigateway-api-userinfo": userinfo,
                "x-api-key": "tk_live_should_be_ignored",
            }
        )
        expectedUserId = givenFirebaseUid

        # WHEN the auth dependency resolves the uid
        resolvedUserId = await auth.get_firebase_uid(givenRequest)

        # THEN the Firebase identity wins; the api key is never consulted
        assert resolvedUserId == expectedUserId
