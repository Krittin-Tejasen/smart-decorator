import unittest

from app.services.catalog_attributes import extract_color_from_title, material_tags, materials_text


class ColorTests(unittest.TestCase):
    def test_colour_segment_after_first_comma(self):
        self.assertEqual(extract_color_from_title("BILLY bookcase, white, 80x202 cm"), "white")

    def test_first_mentioned_colour_wins_not_table_order(self):
        # "white" is first in the colour table, but the title says black first
        self.assertEqual(extract_color_from_title("KALLAX shelf unit, black/white, 77x77 cm"), "black")

    def test_wood_tone_only_when_no_colour_word(self):
        self.assertEqual(extract_color_from_title("HEMNES coffee table, oak effect, 90x55 cm"), "brown")
        self.assertEqual(extract_color_from_title("MALM bed frame, white stained oak veneer, 160x200 cm"), "white")

    def test_material_words_are_not_colours(self):
        self.assertIsNone(extract_color_from_title("BRANÄS basket, rattan, 32x34 cm"))

    def test_size_is_not_read_as_colour(self):
        self.assertIsNone(extract_color_from_title("FOO table, 120x80 cm"))


class MaterialTests(unittest.TestCase):
    def test_materials_text_reads_only_material_lines(self):
        data = [{"type": "materials", "value": "undefined Steel"}, {"type": "care", "value": "Wipe with a cloth"}]
        self.assertEqual(materials_text(data, None), "steel")

    def test_melamine_foil_is_not_plastic(self):
        self.assertEqual(material_tags("particleboard, melamine foil, paper foil, plastic edging"), "Wood")

    def test_two_materials(self):
        tags = material_tags("seat: polyester fabric, frame: solid wood, solid wood, solid wood")
        self.assertEqual(tags.split("|")[0], "Wood")
        self.assertIn("Fabric", tags)

    def test_ambiguous_short_word_ignored_in_title_fallback(self):
        # "ek" is oak in Swedish but also part of names/words elsewhere
        self.assertEqual(material_tags("", "EKET cabinet", "a cabinet"), "Other")

    def test_ambiguous_word_trusted_in_materials_text(self):
        self.assertEqual(material_tags("ek, fanér"), "Wood")


if __name__ == "__main__":
    unittest.main()
