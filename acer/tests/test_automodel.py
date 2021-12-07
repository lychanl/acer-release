import unittest

from algos.automodel import AutoModelComponent, AutoModel, AutoModelCyclicReferenceError


class TestAutoModel(unittest.TestCase):
    def test_call_list_build(self):
        model = AutoModel()
        comp1 = AutoModelComponent()
        comp2 = AutoModelComponent()

        m1 = object()
        m2 = object()
        m3 = object()

        comp1.register_method("m1", m1, {})
        comp2.register_method("m2", m2, {"1": "c1.m1"})
        comp2.register_method("m3", m3, {"1": "c1.m1", "2": "c2.m2"})

        model.register_component("c1", comp1)
        model.register_component("c2", comp2)

        self.assertListEqual(
            model.prepare_call_list(["c2.m2", "c2.m3"], [])[0], [
                ("c1.m1", m1, {}),
                ("c2.m2", m2, {"1": "c1.m1"}),
                ("c2.m3", m3, {"1": "c1.m1", "2": "c2.m2"}),
        ])

    def test_call_list_invalid(self):
        model = AutoModel()
        comp1 = AutoModelComponent()
        comp2 = AutoModelComponent()

        m1 = object()
        m2 = object()
        m3 = object()

        comp1.register_method("m1", m1, {})
        comp2.register_method("m2", m2, {"1": "c1.m1"})
        comp2.register_method("m3", m3, {"1": "c3.m1", "2": "c2.m2"})

        model.register_component("c1", comp1)
        model.register_component("c2", comp2)

        self.assertRaises(ValueError, lambda: model.prepare_call_list(["c2.m2", "c2.m3"], []))
        self.assertRaises(ValueError, lambda: model.prepare_call_list(["c2.m4"], []))

    def test_call_list_cyclic(self):
        model = AutoModel()
        comp1 = AutoModelComponent()
        comp2 = AutoModelComponent()

        m1 = object()
        m2 = object()
        m3 = object()

        comp1.register_method("m1", m1, {})
        comp2.register_method("m2", m2, {"3": "c2.m3"})
        comp2.register_method("m3", m3, {"1": "c1.m1", "2": "c2.m2"})

        model.register_component("c1", comp1)
        model.register_component("c2", comp2)

        self.assertRaises(AutoModelCyclicReferenceError, lambda: model.prepare_call_list(["c2.m2", "c2.m3"], []))

    def test_call_list_data(self):
        model = AutoModel()
        comp1 = AutoModelComponent()
        comp2 = AutoModelComponent()

        m1 = object()
        m2 = object()
        m3 = object()

        comp1.register_method("m1", m1, {"1": "d2"})
        comp2.register_method("m2", m2, {"1": "c1.m1"})
        comp2.register_method("m3", m3, {"1": "c1.m1", "2": "c2.m2", "3": "d3"})

        model.register_component("c1", comp1)
        model.register_component("c2", comp2)

        self.assertCountEqual(model.prepare_call_list(["c2.m2", "c2.m3"], ["d1", "d2", "d3"])[1], ["d2", "d3"])
