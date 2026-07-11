"""Tests for the classify-v2 auth dependency, focused on the API-key path."""

import base64
import json

import pytest
from starlette.requests import Request

from classify_v2.app.auth import firebase
from classify_v2.app.auth.api_key_resolver import hash_api_key


def _request_with_headers(headers: dict[str, str]) -> Request:
    raw_headers = [
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in headers.items()
    ]
    return Request({"type": "http", "headers": raw_headers})


class TestGetFirebaseUidApiKeyPath:
    async def test_api_key_resolves_to_owning_user(
        self, monkeypatch, in_memory_application_database
    ):
        # GIVEN production auth (gateway-fronted, not local)
        monkeypatch.setenv("TARGET_ENVIRONMENT_TYPE", "development")
        # AND a stored key owned by a specific user
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
            firebase.ClassifyDBProvider,
            "get_application_db",
            classmethod(lambda cls: _immediate(in_memory_application_database)),
        )
        # AND a request carrying only the api key (no Firebase user-info)
        givenRequest = _request_with_headers({"x-api-key": givenPlaintextKey})
        expectedUserId = givenOwnerUserId

        # WHEN the auth dependency resolves the uid
        resolvedUserId = await firebase.get_firebase_uid(givenRequest)

        # THEN it is the key's owner, not the shared service uid
        assert resolvedUserId == expectedUserId

    async def test_unknown_api_key_falls_back_to_shared_uid(
        self, monkeypatch, in_memory_application_database
    ):
        # GIVEN production auth and an api key with no matching owner
        monkeypatch.setenv("TARGET_ENVIRONMENT_TYPE", "development")
        monkeypatch.setattr(
            firebase.ClassifyDBProvider,
            "get_application_db",
            classmethod(lambda cls: _immediate(in_memory_application_database)),
        )
        givenRequest = _request_with_headers({"x-api-key": "tk_live_unknown"})
        expectedUserId = firebase._API_KEY_UID

        # WHEN the auth dependency resolves the uid
        resolvedUserId = await firebase.get_firebase_uid(givenRequest)

        # THEN it falls back to the shared service uid (gateway already gated)
        assert resolvedUserId == expectedUserId

    async def test_firebase_userinfo_takes_precedence_over_api_key(
        self, monkeypatch
    ):
        # GIVEN production auth and a Firebase user-info header present
        monkeypatch.setenv("TARGET_ENVIRONMENT_TYPE", "development")
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
        resolvedUserId = await firebase.get_firebase_uid(givenRequest)

        # THEN the Firebase identity wins; the api key is never consulted
        assert resolvedUserId == expectedUserId


async def _immediate(value):
    """Await-able that returns `value` — stand-in for get_application_db."""
    return value
