from vaani.generate import translate_query_for_msmarco, _rule_based_translate


def test_translate_english_economic_capital():
    q = "what is the economic capital of india"
    hin = translate_query_for_msmarco(q)
    assert "भारत की आर्थिक राजधानी" in hin or "आर्थिक राजधानी" in hin


def test_translate_english_corporation():
    q = "what is a corporation"
    hin = translate_query_for_msmarco(q)
    assert "कॉर्पोरेशन क्या है" in hin


def test_translate_hindi_passthrough():
    q = "भारत की राजधानी क्या है?"
    assert translate_query_for_msmarco(q) == q


def test_rule_based_translate_capital():
    assert "भारत की राजधानी" in _rule_based_translate("capital of india")
