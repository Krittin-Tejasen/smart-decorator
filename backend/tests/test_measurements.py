import unittest

from app.services.measurements import extract_dims, parse_length_cm


def m(**named):
    return [{"name": name.replace("_", " "), "value": value} for name, value in named.items()]


class ParseLengthTests(unittest.TestCase):
    def test_metric(self):
        self.assertEqual(parse_length_cm("75 cm"), 75.0)
        self.assertEqual(parse_length_cm("95 mm"), 9.5)
        self.assertEqual(parse_length_cm("1.5 m"), 150.0)
        self.assertEqual(parse_length_cm("80"), 80.0)

    def test_imperial_with_fractions(self):
        self.assertEqual(parse_length_cm('33 1/2 "'), 85.09)
        self.assertEqual(parse_length_cm('¾ "'), 1.91)
        self.assertEqual(parse_length_cm("5 ' 3 \""), 160.02)

    def test_metric_before_bracket(self):
        self.assertEqual(parse_length_cm('75 cm (29 1/2 ")'), 75.0)

    def test_arabic_and_hebrew_units(self):
        self.assertEqual(parse_length_cm("75 سم"), 75.0)
        self.assertEqual(parse_length_cm('75 ס"מ'), 75.0)

    def test_not_a_length(self):
        for value in ("13 W", "77.50 sq feet", "1.86 kg", "25000 hr", None, ""):
            self.assertIsNone(parse_length_cm(value), value)


class ExtractDimsTests(unittest.TestCase):
    def test_plain_names(self):
        dims = extract_dims(m(Width="80 cm", Height="200 cm", Depth="40 cm"))
        self.assertEqual((dims["width_cm"], dims["height_cm"], dims["depth_cm"]), (80.0, 200.0, 40.0))

    def test_rug_has_width_and_length_but_no_height(self):
        dims = extract_dims(m(Length='9 \' 10 "', Width='7 \' 10 "', Area="77.50 sq feet", Thickness='¾ "'))
        self.assertIsNotNone(dims["width_cm"])
        self.assertIsNone(dims["height_cm"])
        self.assertEqual(dims["depth_cm"], dims["length_cm"])

    def test_lamp_diameter_becomes_width(self):
        dims = extract_dims(m(Height='7 "', Luminous_flux="3 Lumen", Diameter='4 "'))
        self.assertEqual(dims["diameter_cm"], 10.16)
        self.assertEqual(dims["width_cm"], 10.16)

    def test_sofa_height_including_cushions(self):
        dims = extract_dims(m(Bed_height='20 7/8 "', Height_including_back_cushions='33 1/2 "', Backrest_height='28 "', Depth='38 5/8 "'))
        self.assertEqual(dims["height_cm"], 85.09)

    def test_plain_name_beats_qualified_one(self):
        dims = extract_dims(m(Seat_width="50 cm", Width="200 cm"))
        self.assertEqual(dims["width_cm"], 200.0)

    def test_qualified_name_is_last_resort(self):
        dims = extract_dims(m(Max_height="120 cm", Min_height="75 cm", Width="70 cm"))
        self.assertEqual(dims["height_cm"], 120.0)

    def test_german_dining_table(self):
        dims = extract_dims(m(Tischbreite="85 cm", Tischlänge="140 cm", Tischhöhe="75 cm"))
        self.assertEqual((dims["width_cm"], dims["height_cm"], dims["depth_cm"]), (85.0, 75.0, 140.0))

    def test_arabic_names(self):
        dims = extract_dims([
            {"name": "الإرتفاع", "value": "75 سم"},
            {"name": "الطول", "value": "80 سم"},
            {"name": "العرض", "value": "70 سم"},
        ])
        self.assertEqual((dims["width_cm"], dims["height_cm"], dims["depth_cm"]), (70.0, 75.0, 80.0))

    def test_other_catalog_languages(self):
        romanian = extract_dims([
            {"name": "Lățime", "value": "60 cm"}, {"name": "Înălțime", "value": "80 cm"}, {"name": "Adâncime", "value": "40 cm"},
        ])
        self.assertEqual((romanian["width_cm"], romanian["height_cm"], romanian["depth_cm"]), (60.0, 80.0, 40.0))
        basque = extract_dims([{"name": "Zabalera", "value": "60 cm"}, {"name": "Altuera", "value": "80 cm"}, {"name": "Sakonera", "value": "40 cm"}])
        self.assertEqual((basque["width_cm"], basque["height_cm"], basque["depth_cm"]), (60.0, 80.0, 40.0))

    def test_unreadable_is_kept_as_none(self):
        dims = extract_dims(m(Weight="1.86 kg"))
        self.assertTrue(all(v is None for v in dims.values()))
        self.assertTrue(all(v is None for v in extract_dims(None).values()))

    def test_leading_space_in_name(self):
        dims = extract_dims([{"name": " Width", "value": "10 cm"}])
        self.assertEqual(dims["width_cm"], 10.0)


if __name__ == "__main__":
    unittest.main()
