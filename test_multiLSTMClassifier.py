from unittest import TestCase
from classify import MultiLSTMClassifier


class TestMultiLSTMClassifier(TestCase):
    mlc = MultiLSTMClassifier()

    def test_exploratory_analysis(self):
        self.mlc.exploratory_analysis()

    def test_print_complaint(self):
        self.mlc.print_complaint(4165)

    def test_clean_text(self):
        self.mlc.clean_text("[...test[]\{}")

    def test_remove_stop_words(self):
        self.mlc.remove_stop_words("this is i not you")

    def test_pipeline(self):
        self.mlc.pipeline()

if __name__ == "__main__":
    t = TestMultiLSTMClassifier()
    # t.test_exploratory_analysis()
    t.test_pipeline()



