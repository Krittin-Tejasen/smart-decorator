"""The pure parts of scripts/ingest_catalog_v2.py: which row stands for a product,
what the Pinecone record carries, and what is copied versus encoded."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ingest_catalog_v2 import group_rows, pick_representative, plan_vectors, product_metadata


def row(uid, country, group="img:1", image="https://x/a.jpg", **extra):
    return {
        "unique_id": uid, "country_code": country, "group_key": group, "main_image": image,
        "main_title": "T", "sku": "1", "furniture_type": "sofa", **extra,
    }


class RepresentativeTests(unittest.TestCase):
    def test_the_english_listing_stands_for_the_product(self):
        rows = [row("de_1", "de"), row("us_1", "us"), row("gb_1", "gb"), row("sa_1", "sa")]
        self.assertEqual(pick_representative(rows)["unique_id"], "gb_1")
        self.assertEqual(pick_representative(rows[:2] + rows[3:])["unique_id"], "us_1")

    def test_without_an_english_listing_the_smallest_id_wins(self):
        self.assertEqual(pick_representative([row("b", "sa"), row("a", "de")])["unique_id"], "a")

    def test_rows_without_a_picture_cannot_stand_for_it(self):
        rows = [row("gb_1", "gb", image=None), row("de_1", "de")]
        self.assertEqual(pick_representative(rows)["unique_id"], "de_1")
        self.assertIsNone(pick_representative([row("gb_1", "gb", image="")]))


class MetadataTests(unittest.TestCase):
    def test_carries_type_subtype_group_and_listing(self):
        metadata = product_metadata(
            row("gb_1", "gb", furniture_type="lamp", thb_price=1200.5, width_cm=30.0, color_extracted="white"),
            "floor",
        )
        self.assertEqual(metadata["furniture_type"], "lamp")
        self.assertEqual(metadata["furniture_subtype"], "floor")
        self.assertEqual(metadata["group_key"], "img:1")
        self.assertEqual(metadata["country_code"], "gb")
        self.assertEqual(metadata["thb_price"], 1200.5)
        self.assertEqual(metadata["color"], "white")

    def test_no_nulls_for_pinecone(self):
        metadata = product_metadata(row("gb_1", "gb", height_cm=None, thb_price=None), None)
        self.assertNotIn("height_cm", metadata)
        self.assertNotIn("furniture_subtype", metadata)
        self.assertTrue(all(v is not None for v in metadata.values()))


class PlanTests(unittest.TestCase):
    def test_old_vectors_are_copied_and_new_products_are_encoded(self):
        groups = group_rows([
            row("a_gb", "gb", group="img:1"), row("a_de", "de", group="img:1"),
            row("b_gb", "gb", group="img:2"),
            row("c_gb", "gb", group="img:3", image=None),
        ])
        plan, no_picture = plan_vectors(groups, {"img:1": "OLD-7"}, done=set())
        self.assertEqual([(r["unique_id"], old) for r, old in plan], [("a_gb", "OLD-7"), ("b_gb", None)])
        self.assertEqual(no_picture, 1)

    def test_a_resumed_run_skips_what_is_done(self):
        groups = group_rows([row("a_gb", "gb", group="img:1"), row("b_gb", "gb", group="img:2")])
        plan, _ = plan_vectors(groups, {}, done={"a_gb"})
        self.assertEqual([r["unique_id"] for r, _ in plan], ["b_gb"])

    def test_one_record_per_product(self):
        groups = group_rows([row(f"r{i}", c, group="img:1") for i, c in enumerate(["gb", "de", "fr", "sa"])])
        plan, _ = plan_vectors(groups, {}, done=set())
        self.assertEqual(len(plan), 1)


if __name__ == "__main__":
    unittest.main()
