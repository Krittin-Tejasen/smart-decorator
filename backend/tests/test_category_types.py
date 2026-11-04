import unittest

from app.services.category_types import (
    fill_missing_types,
    one_type_per_product,
    type_from_category,
    types_by_category_id,
)


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
        for unreadable in ([], ["Products", "Zzz unknown"], [None, "", "Products"]):
            self.assertEqual(type_from_category(unreadable), "household")
            self.assertNotEqual(type_from_category(unreadable), "other")


class FillMissingTypesTests(unittest.TestCase):
    TYPES = {"100": "bookshelf", "200": "kitchenware"}

    def test_missing_types_come_from_the_category_id_in_any_language(self):
        rows = [
            {"unique_id": "a", "category_id": "100"},   # an Arabic title, same category id
            {"unique_id": "b", "category_id": "200"},
            {"unique_id": "c", "category_id": "999"},   # an unknown id
        ]
        filled = fill_missing_types({}, rows, self.TYPES)
        self.assertEqual(filled, {"a": "bookshelf", "b": "kitchenware", "c": "household"})

    def test_a_title_type_is_never_replaced(self):
        rows = [{"unique_id": "a", "category_id": "200"}]
        self.assertEqual(fill_missing_types({"a": "sofa"}, rows, self.TYPES), {"a": "sofa"})

    def test_a_title_other_does_not_block_the_category(self):
        # "other" from the title reader only means a small-item word matched.
        rows = [{"unique_id": "a", "category_id": "200"}]
        self.assertEqual(fill_missing_types({"a": "other"}, rows, self.TYPES), {"a": "kitchenware"})
        rows = [{"unique_id": "b", "category_id": "999"}]
        self.assertEqual(fill_missing_types({"b": "other"}, rows, self.TYPES), {"b": "household"})


class OneTypePerProductTests(unittest.TestCase):
    def test_every_row_of_a_product_gets_the_majority_type(self):
        rows = [
            {"unique_id": "gb", "group_key": "img:1"},
            {"unique_id": "de", "group_key": "img:1"},
            {"unique_id": "sa", "group_key": "img:1"},
            {"unique_id": "x", "group_key": "img:2"},
        ]
        types = {"gb": "bookshelf", "de": "bookshelf", "sa": "cabinet", "x": "rug"}
        self.assertEqual(
            one_type_per_product(types, rows),
            {"gb": "bookshelf", "de": "bookshelf", "sa": "bookshelf", "x": "rug"},
        )

    def test_a_tie_goes_to_the_first_row(self):
        rows = [{"unique_id": "a", "group_key": "k"}, {"unique_id": "b", "group_key": "k"}]
        self.assertEqual(one_type_per_product({"a": "lamp", "b": "mirror"}, rows), {"a": "lamp", "b": "lamp"})


class TypesByCategoryIdTests(unittest.TestCase):
    def test_countries_vote_and_unreadable_paths_do_not(self):
        # the real id 18751: one site says "Pendant lighting", others "Pendants" or another language
        paths = {
            "18751": {
                "us": path("Lamps & light fixtures", "Ceiling lights", "Pendant lighting"),
                "qa": path("Lamps", "Ceiling lights", "Pendants"),
                "de": ["Produkte", "Beleuchtung", "Hängeleuchten"],
                "my": ["Produk", "Pencahayaan"],
            }
        }
        self.assertEqual(types_by_category_id(paths, ("gb", "us"))["18751"], "lamp")

    def test_the_majority_wins_over_a_single_odd_path(self):
        paths = {"1": {
            "gb": path("Rugs", "Round rugs"), "us": path("Rugs", "Large & medium rugs"),
            "ae": path("Rugs", "Runners & small rugs"), "sg": path("Bath textiles", "Bath mats"),
        }}
        self.assertEqual(types_by_category_id(paths, ("gb", "us", "sg"))["1"], "rug")

    def test_a_tie_goes_to_the_english_site(self):
        paths = {"1": {"gb": path("Mirrors"), "de": path("Candles and candle holders")}}
        self.assertEqual(types_by_category_id(paths, ("gb",))["1"], "mirror")

    def test_an_id_no_path_can_read_is_a_household_item_never_other(self):
        self.assertEqual(types_by_category_id({"3": {"fr": ["Produits", "Zzz"]}})["3"], "household")

    def test_laundry_furniture_in_dutch_and_spanish(self):
        paths = {"48925": {
            "nl": ["Producten", "Badkamerproducten", "Washok", "Wasmachinekasten & wasmachinerekken"],
            "cl": ["Productos", "Baño", "Lavandería y lavadero", "Muebles para lavadero"],
        }}
        self.assertEqual(types_by_category_id(paths)["48925"], "cabinet")


if __name__ == "__main__":
    unittest.main()
