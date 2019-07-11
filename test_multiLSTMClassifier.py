from unittest import TestCase
from classify import MultiLSTMClassifier


class TestMultiLSTMClassifier(TestCase):
    c = MultiLSTMClassifier()
    p = c.p
    m = c.m

    def test_exploratory_analysis(self):
        self.c.p.exploratory_analysis()

    def test_print_complaint(self):
        self.p.print_complaint(4165)

    def test_clean_text(self):
        self.p.clean_text("[...test[]\{}")

    def test_remove_stop_words(self):
        self.p.remove_stop_words("this is i not you")

    def test_pipeline(self):
        self.c.pipeline()

    def test_LSTMModel(self):
        self.m.LSTMModel()

if __name__ == "__main__":
    t = TestMultiLSTMClassifier()
    t.test_exploratory_analysis()
    t.test_pipeline()
    #t.test_LSTMModel()