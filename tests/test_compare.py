from decimal import Decimal

from dbagenteval.compare import canonical, match_any, rows_match


def test_values_that_mean_the_same_get_the_same_text():
    assert canonical(Decimal("1.20")) == canonical(1.2) == canonical("1.2000")
    assert canonical(None) == canonical("") == ""
    assert canonical(True) == canonical(1) == "1"
    assert canonical(-0.001) == "0"
    assert canonical(" Books ") == "Books"
    assert canonical("nan") == "nan"


def test_column_names_and_extra_columns_are_ignored():
    actual = [{"total_order_count": Decimal("15.00"), "store": "A"}]
    assert rows_match([{"count": 15}], actual)


def test_row_count_must_match():
    assert not rows_match([{"n": 1}], [{"n": 1}, {"n": 1}])


def test_wrong_value_fails():
    assert not rows_match([{"count": 15}], [{"count": 16}])


def test_values_must_pair_up_within_rows():
    expected = [{"category": "A", "n": 1}, {"category": "B", "n": 2}]
    swapped = [{"p": "A", "c": 2}, {"p": "B", "c": 1}]
    assert not rows_match(expected, swapped)
    assert rows_match(expected, [{"p": "B", "c": 2}, {"p": "A", "c": 1}])


def test_ordered_comparison_respects_row_order():
    expected = [{"p": "A"}, {"p": "B"}]
    reversed_rows = [{"p": "B"}, {"p": "A"}]
    assert rows_match(expected, reversed_rows)
    assert not rows_match(expected, reversed_rows, ordered=True)


def test_empty_expected_matches_only_empty_result():
    assert rows_match([], [])
    assert not rows_match([], [{"n": 1}])


def test_match_any_returns_index_of_matching_answer():
    accepted = [[{"n": 1}], [{"n": 2}]]
    assert match_any(accepted, [{"x": 2}]) == 1
    assert match_any(accepted, [{"x": 3}]) is None


def test_a_column_with_the_expected_name_is_the_one_compared():
    assert not rows_match([{"A": 1, "B": 2}], [{"A": 2, "B": 1}])
    assert not rows_match([{"delivered": 3, "cancelled": 1}], [{"delivered": 1, "cancelled": 3}])
    # The named column is wrong even though another column holds the expected value.
    assert not rows_match([{"total": 20000}], [{"total": 8000, "other": 20000}])
    assert rows_match([{"Total": 20000}], [{"total": 20000, "other": 1}])


def test_precision_controls_how_closely_numbers_must_agree():
    assert rows_match([{"rate": 0.001}], [{"rate": 0.004}])  # both 0.00 at 2 places
    assert not rows_match([{"rate": 0.001}], [{"rate": 0.004}], precision=3)
    assert rows_match([{"x": 2.67}], [{"x": 2.665}])  # half rounds up, as databases do
    assert match_any([[{"rate": 0.12}]], [{"r": 0.118}], precision=3) is None


def test_only_plainly_written_numbers_are_read_as_numbers():
    assert rows_match([{"x": "1.20"}], [{"x": Decimal("1.2")}])
    assert not rows_match([{"zip": "01234"}], [{"zip": "1234"}])
    assert not rows_match([{"code": "1E5"}], [{"code": "100000"}])
    assert canonical("1_000") == "1_000"
    assert canonical(1e-7, precision=8) == "0.0000001"


def test_many_look_alike_columns_do_not_stall_the_comparison():
    expected = [{f"e{i}": 0 for i in range(13)}]
    actual = [{f"a{i}": 0 for i in range(12)}]
    assert not rows_match(expected, actual)

    flags = 12
    expected = [{f"e{c}": int(r % (c + 2) == 0) for c in range(flags)} for r in range(24)]
    actual = [{f"a{c}": row[f"e{c}"] for c in range(flags)} for row in expected]
    assert rows_match(expected, actual)
    actual[0]["a0"], actual[1]["a0"] = actual[1]["a0"], actual[0]["a0"]
    assert not rows_match(expected, actual)


def test_an_unmatched_arrangement_of_look_alike_columns_gives_up_quickly():
    # Ten flag columns that each look right on their own, but two of them flag the same
    # row. Without a limit the search would try every ordering of the ten.
    size = 10
    expected = [{f"e{c}": int(r == c) for c in range(size)} for r in range(size)]
    actual = [{f"a{c}": int(r == c) for c in range(size)} for r in range(size)]
    for row in actual:
        row["a9"] = actual[0]["a0"] if row is actual[0] else 0
    assert not rows_match(expected, actual)
    assert rows_match([{"n": 10**5000}], [{"m": 10**5000}])  # too long to print as a number
