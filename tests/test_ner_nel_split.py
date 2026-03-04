"""
Local test: Verify NER and NEL modules work independently.
Run from the classifier root: python -m tests.test_ner_nel_split
"""

import time

SAMPLE_TEXT = (
    "Head Chef. We are looking for an experienced Head Chef who can plan menus, "
    "manage kitchen staff, and ensure food safety standards. "
    "A diploma in Culinary Arts is required."
)


def main():
    print("=" * 60)
    print("TEST: NER and NEL modules (standalone)")
    print("=" * 60)

    # --- Test NER ---
    print("\n[1/3] Loading NER model...")
    t0 = time.time()
    from inference.ner import NERModel
    ner = NERModel()
    print(f"      Loaded in {time.time() - t0:.1f}s")

    print("\n[2/3] Running NER on sample text...")
    print(f'      Input: "{SAMPLE_TEXT[:80]}..."')
    t0 = time.time()
    entities = ner.extract(SAMPLE_TEXT)
    ner_time = time.time() - t0

    print(f"      Found {len(entities)} entities in {ner_time:.3f}s:\n")
    for e in entities:
        print(f"        {e['entity_type']:15s}  \"{e['surface_form']}\"  "
              f"(chars {e['span']['start']}-{e['span']['end']})")

    # --- Test NEL ---
    print("\n[3/3] Loading NEL linker and linking entities...")
    t0 = time.time()
    from inference.nel import NELLinker
    nel = NELLinker()
    print(f"      Loaded in {time.time() - t0:.1f}s")

    nel_input = [
        {"text": e["surface_form"], "entity_type": e["entity_type"]}
        for e in entities
        if e["entity_type"] in ("occupation", "skill", "qualification")
    ]

    print(f"      Linking {len(nel_input)} entities to ESCO...")
    t0 = time.time()
    linked = nel.link(nel_input, top_k=3, min_similarity=0.3)
    nel_time = time.time() - t0

    print(f"      Linked in {nel_time:.3f}s:\n")
    for item in linked:
        print(f"        \"{item['input_text']}\" ({item['entity_type']}):")
        for m in item["matches"][:3]:
            label = m.get("label", "?")
            score = m.get("similarity_score", 0)
            code = m.get("code", "")
            suffix = f"  code={code}" if code else ""
            print(f"          -> {label}  (score={score}){suffix}")
        print()

    
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  NER: {len(entities)} entities extracted in {ner_time:.3f}s")
    print(f"  NEL: {len(linked)} entities linked in {nel_time:.3f}s")
    print(f"  NER module: inference/ner.py  (reuses EntityLinker statics)")
    print(f"  NEL module: inference/nel.py  (reuses EntityLinker.create_tensors)")
    print(f"  Original linker.py: untouched")
    print("=" * 60)


if __name__ == "__main__":
    main()
