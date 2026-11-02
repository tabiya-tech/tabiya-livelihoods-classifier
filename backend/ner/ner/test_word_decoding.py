"""Tests for word-level decoding: the model gets pre-split words and one tag per word.

The Spanish checkpoint (BSC-LT/mRoBERTa) uses a SentencePiece tokenizer that does not
mark word boundaries on raw text: every piece reported word 0, so a whole sentence came
back as a single entity. Splitting words ourselves fixes the boundaries for any tokenizer.
"""

from ner.model import first_piece_word_tags, split_words


class TestSplitWords:
    def test_punctuation_is_split_from_words(self):
        # GIVEN a job title with a location in brackets
        given_text = "Promotor de ventas (Abancay)."

        # WHEN it is split into words
        actual_words = [word.text for word in split_words(given_text)]

        # THEN each bracket and the full stop are words of their own, as in the training data
        expected_words = ["Promotor", "de", "ventas", "(", "Abancay", ")", "."]
        assert actual_words == expected_words

    def test_offsets_point_back_into_the_text(self):
        # GIVEN Spanish text with accents and punctuation
        given_text = "Técnico en agronomía, con moto propia"

        # WHEN it is split into words
        actual_words = split_words(given_text)

        # THEN every word's offsets slice exactly that word out of the text
        assert all(given_text[word.start:word.end] == word.text for word in actual_words)

    def test_inner_hyphens_and_apostrophes_stay_in_the_word(self):
        # GIVEN words joined by a hyphen and an apostrophe
        given_text = "full-time chef's"

        # WHEN it is split into words
        actual_words = [word.text for word in split_words(given_text)]

        # THEN neither is split apart
        expected_words = ["full-time", "chef's"]
        assert actual_words == expected_words

    def test_blank_text_has_no_words(self):
        # GIVEN text made only of whitespace
        given_text = "   "

        # WHEN it is split into words
        actual_words = split_words(given_text)

        # THEN there are no words
        assert actual_words == []


class TestFirstPieceWordTags:
    def test_each_word_takes_its_first_piece_tag(self):
        # GIVEN two words split into pieces, with special tokens around them
        given_piece_tags = ["O", "B-Skill", "O", "I-Skill", "O"]
        given_word_ids = [None, 0, 0, 1, None]
        given_word_count = 2

        # WHEN reading one tag per word
        actual_word_tags = first_piece_word_tags(given_piece_tags, given_word_ids, given_word_count)

        # THEN only the first piece of each word counts
        expected_word_tags = ["B-Skill", "I-Skill"]
        assert actual_word_tags == expected_word_tags

    def test_words_cut_off_by_truncation_are_outside(self):
        # GIVEN three words, of which truncation kept only the first
        given_piece_tags = ["O", "B-Skill", "O"]
        given_word_ids = [None, 0, None]
        given_word_count = 3

        # WHEN reading one tag per word
        actual_word_tags = first_piece_word_tags(given_piece_tags, given_word_ids, given_word_count)

        # THEN the words that never reached the model are tagged O
        expected_word_tags = ["B-Skill", "O", "O"]
        assert actual_word_tags == expected_word_tags
