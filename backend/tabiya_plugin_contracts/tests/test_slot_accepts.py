"""Tests for slot compatibility (`slot_accepts`)."""

from __future__ import annotations

from tabiya_plugin_contracts import SlotType, slot_accepts


def test_exact_match_is_accepted() -> None:
    # GIVEN identical producer and consumer slot types
    givenType = SlotType.LINKED_ENTITIES

    # WHEN we check compatibility
    accepted = slot_accepts(givenType, givenType)

    # THEN an exact match is accepted
    assert accepted is True


def test_entities_feeds_a_linked_entities_consumer() -> None:
    # GIVEN a producer emitting Entities and a consumer wanting LinkedEntities
    givenProducer = SlotType.ENTITIES
    givenConsumer = SlotType.LINKED_ENTITIES

    # WHEN we check compatibility
    accepted = slot_accepts(givenProducer, givenConsumer)

    # THEN it's accepted — this lets text→ner→results run without NEL
    assert accepted is True


def test_linked_entities_does_not_feed_an_entities_only_consumer() -> None:
    # GIVEN the reverse direction (LinkedEntities → Entities)
    givenProducer = SlotType.LINKED_ENTITIES
    givenConsumer = SlotType.ENTITIES

    # WHEN we check compatibility
    accepted = slot_accepts(givenProducer, givenConsumer)

    # THEN the subtype relationship is one-directional, so this is rejected
    assert accepted is False


def test_unrelated_types_are_rejected() -> None:
    # GIVEN unrelated slot types
    givenProducer = SlotType.RAW_TEXT
    givenConsumer = SlotType.ENTITIES

    # WHEN we check compatibility
    accepted = slot_accepts(givenProducer, givenConsumer)

    # THEN they're incompatible
    assert accepted is False
