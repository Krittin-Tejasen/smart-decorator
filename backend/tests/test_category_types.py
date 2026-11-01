import unittest

from app.services.category_types import fill_missing_types, resolve_category_paths, type_from_category


def path(*names):
    return ["Products", *names]


class TypeFromCategoryTests(unittest.TestCase):
    def test_room_furniture(self):
        self.assertEqual(type_from_category(path("Bookcases & shelving units", "Bookcases")), "bookshelf")
        self.assertEqual(type_from_category(path("Sofas", "3 seater sofas")), "sofa")
        self.assertEqual(type_from_category(path("Lamps", "Floor lamps")), "lamp")
        self.assertEqual(type_from_category(path("Rugs", "Round rugs")), "rug")
        self.assertEqual(type_from_category(path("Curtains", "Blackout curtains")), "curtain")

    def test_specific_names_win_over_the_words_they_contain(self):
        self.assertEqual(type_from_category(path("Sofa & armchairs covers", "VIMLE covers")), "textile")
        self.assertEqual(type_from_category(path("Storage solution systems", "PAX system", "PAX sliding doors")), "part")
        self.assertEqual(type_from_category(path("Storage solution systems", "PAX system", "PAX wardrobes with doors")), "wardrobe")
        self.assertEqual(type_from_category(path("Storage solution systems", "BESTÅ storage system", "BESTÅ TV benches")), "tv_bench")
        self.assertEqual(type_from_category(path("Kitchen systems", "METOD kitchens", "METOD cover panels & deco strips")), "part")
        self.assertEqual(type_from_category(path("Bathroom ranges", "ENHET", "ENHET mirror & mirror cabinets")), "bathroom")
        self.assertEqual(type_from_category(path("Mirrors", "Vanity mirrors")), "mirror")
        self.assertEqual(type_from_category(path("Coffee & tea", "Tea pots & accessories")), "kitchenware")

    def test_things_that_are_not_furniture_are_still_named(self):
        self.assertEqual(type_from_category(path("Cookware", "Frying pans")), "kitchenware")
        self.assertEqual(type_from_category(path("Cables & chargers", "USB chargers")), "electronics")
        self.assertEqual(type_from_category(path("Baby", "Nursery furniture", "Baby highchairs")), "baby")

    def test_small_pieces_that_fit_no_room_type_are_named(self):
        self.assertEqual(type_from_category(path("Kitchen islands & trolleys")), "cart")
        self.assertEqual(type_from_category(path("Clothes storage & organisers", "Racks & stands")), "rack")
        self.assertEqual(type_from_category(path("Room dividers")), "room_divider")
        self.assertEqual(type_from_category(path("Clocks", "Wall clocks")), "decoration")
        self.assertEqual(type_from_category(path("Speakers", "Bluetooth speakers")), "electronics")

    def test_never_none(self):
        self.assertEqual(type_from_category([]), "other")
        self.assertEqual(type_from_category(["Products", "Zzz unknown"]), "other")
        self.assertEqual(type_from_category([None, "", "Products"]), "other")


class FillMissingTypesTests(unittest.TestCase):
    PATHS = {"100": path("Bookcases & shelving units", "Bookcases"), "200": path("Cookware", "Pots")}

    def test_missing_types_come_from_the_category_id_in_any_language(self):
        rows = [
            {"unique_id": "a", "category_id": "100"},   # an Arabic title, same category id
            {"unique_id": "b", "category_id": "200"},
            {"unique_id": "c", "category_id": "999"},   # an id with no English path
        ]
        filled = fill_missing_types({}, rows, self.PATHS)
        self.assertEqual(filled, {"a": "bookshelf", "b": "kitchenware", "c": "other"})

    def test_a_title_type_is_never_replaced(self):
        rows = [{"unique_id": "a", "category_id": "200"}]
        self.assertEqual(fill_missing_types({"a": "sofa"}, rows, self.PATHS), {"a": "sofa"})

    def test_a_title_other_does_not_block_the_category(self):
        # "other" from the title reader only means a small-item word matched.
        rows = [{"unique_id": "a", "category_id": "200"}]
        self.assertEqual(fill_missing_types({"a": "other"}, rows, self.PATHS), {"a": "kitchenware"})
        rows = [{"unique_id": "b", "category_id": "999"}]
        self.assertEqual(fill_missing_types({"b": "other"}, rows, self.PATHS), {"b": "other"})

    def test_a_path_is_read_from_any_country_when_no_english_site_has_one(self):
        paths = {
            "1": {"de": ["Produkte", "Küche"], "ae": ["Products", "Bar furniture", "Bar tables sets"]},
            "2": {"gb": ["Products", "Rugs"], "ae": ["Products", "Mats"]},
            "3": {"fr": ["Produits", "Zzz"]},
        }
        resolved = resolve_category_paths(paths, ("gb", "us"))
        self.assertEqual(resolved["1"], ["Products", "Bar furniture", "Bar tables sets"])
        self.assertEqual(resolved["2"], ["Products", "Rugs"])
        self.assertNotIn("3", resolved)


if __name__ == "__main__":
    unittest.main()
