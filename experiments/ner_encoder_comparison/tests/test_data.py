from ner_experiments.corpus.peru_corpus import ConversionReport, _read_ad
from ner_experiments.corpus.qualification_levels import is_qualification_phrase
from ner_experiments.corpus.splits import make_split


class TestQualificationLevels:
    def test_levels_of_education_are_qualifications(self):
        for phrase in ["Técnico", "titulado", "postgrado", "post - grado", "Maestría", "profesional titulado"]:
            assert is_qualification_phrase(phrase), phrase

    def test_techniques_and_skills_stay_skills(self):
        for phrase in ["técnicas de venta", "técnicas varias", "comunicación con equipos técnicos", "liderazgo"]:
            assert not is_qualification_phrase(phrase), phrase


def _peru_lines(rows: list[tuple[str, str]]) -> str:
    return "\n".join(f"{word}\tnc\t{tag}" for word, tag in rows)


class TestPeruConversion:
    def test_tags_map_to_our_three_types(self):
        # GIVEN an ad with a job title, a requirement, a field of study and a benefit
        ad_text = _peru_lines([
            ("Contador", "B-JOB"), ("Excel", "B-REQ"), ("Contabilidad", "B-CARR"), ("movilidad", "B-BEN"),
        ])

        # WHEN converting it
        sentences = _read_ad("random/1", ad_text, "ignore", True, ConversionReport())

        # THEN JOB, REQ and CARR map to our types and everything else is dropped
        assert sentences[0].tags == ["B-Occupation", "B-Skill", "B-Qualification", "O"]

    def test_qualification_levels_move_from_skill_to_qualification(self):
        report = ConversionReport()
        ad_text = _peru_lines([("Técnico", "B-REQ"), ("en", "O"), ("Contabilidad", "B-CARR")])
        sentences = _read_ad("random/1", ad_text, "ignore", True, report)
        assert sentences[0].tags == ["B-Qualification", "O", "B-Qualification"]
        assert [moved.phrase for moved in report.moved_to_qualification] == ["Técnico"]

    def test_blank_lines_separate_sentences(self):
        ad_text = _peru_lines([("Chef", "B-JOB")]) + "\n\n" + _peru_lines([("Cocina", "B-REQ")])
        sentences = _read_ad("random/1", ad_text, "ignore", False, ConversionReport())
        assert [sentence.sentence_index for sentence in sentences] == [0, 1]


class TestSplit:
    def test_every_ad_lands_in_exactly_one_split(self):
        ad_ids_by_folder = {"random": [f"random/{n}" for n in range(100)], "careers": [f"careers/{n}" for n in range(50)]}
        split = make_split(ad_ids_by_folder, 0.7, 0.1, seed=13)
        all_ids = split["train"] + split["dev"] + split["test"]
        assert sorted(all_ids) == sorted(ad_ids_by_folder["random"] + ad_ids_by_folder["careers"])
        assert len(all_ids) == len(set(all_ids))

    def test_the_same_seed_gives_the_same_split(self):
        ad_ids_by_folder = {"random": [f"random/{n}" for n in range(40)]}
        assert make_split(ad_ids_by_folder, 0.7, 0.1, 13) == make_split(ad_ids_by_folder, 0.7, 0.1, 13)
