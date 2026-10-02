import pytest

from dbagenteval import Case, Trace
from dbagenteval.rules import (
    AllowedColumns,
    AllowedTables,
    NoJoins,
    NoSubqueries,
    ReadOnly,
    TenantFilter,
)

CASE = Case(question="q")


def status(rule, sql, extra_stages=()):
    trace = Trace("q", [*extra_stages, {"kind": "generate", "output": sql}])
    return rule.run(trace, CASE).status


# ---- ReadOnly -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a FROM t",
        "SELECT a FROM t WHERE b = 1 AND c = 2",
        "WITH c AS (SELECT a FROM t) SELECT a FROM c",
        "SELECT a FROM t UNION ALL SELECT a FROM u",
        "SELECT a FROM t;",
    ],
)
def test_read_only_accepts_reads(sql):
    assert status(ReadOnly(), sql) == "pass"


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM t",
        "UPDATE t SET a = 1",
        "DROP TABLE t",
        "SELECT a FROM t; DELETE FROM t",
        "SELECT 1; SELECT 2",
        "SELECT a INTO backup FROM t",
        "WITH gone AS (DELETE FROM t RETURNING a) SELECT a FROM gone",
        "SELECT a FROM t FOR UPDATE",
        "EXPLAIN ANALYZE SELECT a FROM t",
        "SELECT pg_sleep(10)",
        "SELECT pg_read_binary_file('/etc/passwd')",
        "SELECT dblink_exec('x', 'DELETE FROM t')",
        "SELECT FROM WHERE",
        ";",
        "-- nothing here",
        "SELECT " + "(" * 400 + "1" + ")" * 400,
    ],
)
def test_read_only_rejects_everything_else(sql):
    assert status(ReadOnly(dialect="postgres"), sql) == "fail"


def test_read_only_forbidden_functions_cover_built_in_names_and_other_dialects():
    assert status(ReadOnly(dialect="mysql"), "SELECT SLEEP(10)") == "fail"
    assert status(ReadOnly(dialect="mysql"), "SELECT LOAD_FILE('/etc/passwd')") == "fail"
    # Functions sqlglot has its own class for are matched too, not only unknown ones.
    for name, sql in [
        ("md5", "SELECT md5('a')"),
        ("lower", "SELECT lower('A')"),
        ("version", "SELECT version()"),
    ]:
        assert status(ReadOnly(forbidden_functions=[name], dialect="postgres"), sql) == "fail"
    # A bare string is one name, not its letters.
    assert status(ReadOnly(forbidden_functions="md5"), "SELECT md5('a')") == "fail"
    assert status(ReadOnly(forbidden_functions="md5"), "SELECT lower('a')") == "pass"


# ---- how rules pick the SQL to check --------------------------------------------------


def test_rules_skip_when_there_is_no_sql():
    trace = Trace("q", [{"kind": "route", "output": "refused"}])
    assert ReadOnly().run(trace, CASE).status == "skip"
    assert ReadOnly(stage="sql").run(trace, CASE).status == "skip"


def test_every_generate_stage_is_checked_so_a_retry_cannot_hide_a_write():
    trace = Trace(
        "q",
        [
            {"kind": "generate", "name": "first_try", "output": "DELETE FROM t"},
            {"kind": "generate", "name": "second_try", "output": "SELECT 1"},
        ],
    )
    result = ReadOnly().run(trace, CASE)
    assert (result.status, result.stage) == ("fail", "first_try")
    assert ReadOnly(stage="second_try").run(trace, CASE).status == "pass"
    # A stage name that matches nothing, while the trace does have SQL, is a typo.
    assert ReadOnly(stage="second_tri").run(trace, CASE).status == "fail"


def test_sql_that_is_not_text_fails_instead_of_being_skipped():
    trace = Trace("q", [{"kind": "generate", "output": {"sql": "DELETE FROM t"}}])
    assert ReadOnly().run(trace, CASE).status == "fail"


def test_unknown_dialect_is_rejected_when_the_rule_is_built():
    with pytest.raises(ValueError):
        TenantFilter("store_id", dialect="postgresql")


@pytest.mark.parametrize(
    "rule",
    [
        ReadOnly(),
        TenantFilter("store_id"),
        AllowedTables(["orders"]),
        AllowedColumns(["a"]),
        NoJoins(),
        NoSubqueries(),
    ],
)
@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "EXPLAIN ANALYZE SELECT a FROM orders",
        "TABLE orders",
        "SELECT a FROM orders WHERE store_id = 's1' /*! OR 1 = 1 */",
        ";",
    ],
)
def test_sql_a_rule_cannot_analyse_fails_every_rule(rule, sql):
    assert status(rule, sql) == "fail"


def test_parse_error_detail_is_plain_text():
    trace = Trace("q", [{"kind": "generate", "output": "SELECT 1 FROM"}])
    assert "\x1b" not in ReadOnly().run(trace, CASE).detail


# ---- TenantFilter ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a FROM orders WHERE store_id = 's1'",
        "SELECT a FROM orders WHERE status = 'X' AND (store_id = 's1')",
        "SELECT a FROM orders o WHERE 's1' = o.store_id",
        "SELECT a FROM orders WHERE store_id IN ('s1')",
        "SELECT a FROM orders WHERE store_id = CAST('s1' AS TEXT)",
        "SELECT a FROM orders WHERE CAST(store_id AS TEXT) = 's1'",
        'SELECT a FROM orders WHERE store_id = "s1"',
        "WITH c AS (SELECT a FROM orders WHERE store_id = 's1') SELECT a FROM c",
        "SELECT a FROM (SELECT a FROM orders WHERE store_id = 's1') x",
        # each table has its own filter
        "SELECT * FROM orders o JOIN items i ON i.order_id = o.id "
        "WHERE o.store_id = 's1' AND i.store_id = 's1'",
        # joined on the tenant column, so one filter covers both
        "SELECT * FROM orders o JOIN items i ON i.store_id = o.store_id WHERE o.store_id = 's1'",
        "SELECT a FROM orders WHERE store_id = 's1' AND "
        + " AND ".join(f"a = {i}" for i in range(3000)),
    ],
)
def test_tenant_filter_accepts_scoped_queries(sql):
    assert status(TenantFilter("store_id", value="s1"), sql) == "pass"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a FROM orders",
        "SELECT a FROM orders WHERE status = 'X'",
        "SELECT a FROM orders WHERE store_id = 's1' OR 1 = 1",
        "SELECT a FROM orders WHERE store_id = 's2'",
        "SELECT a FROM orders WHERE store_id IN ('s1', 's2')",
        "SELECT a FROM orders WHERE store_id = other_id",
        "SELECT a FROM orders WHERE store_id = 's1' AND a IN (SELECT a FROM refunds)",
        "SELECT a FROM orders WHERE store_id = 's1' UNION ALL SELECT a FROM orders",
        "WITH c AS (SELECT a FROM orders) SELECT a FROM c WHERE store_id = 's1'",
        # a CTE named like the table does not make the table disappear
        "WITH orders AS (SELECT * FROM orders) SELECT * FROM orders",
        "WITH orders AS (SELECT 1 AS x) SELECT * FROM public.orders",
        # tables behind parentheses
        "SELECT * FROM (orders)",
        "SELECT * FROM (orders o JOIN stores s ON s.id = o.store_id)",
        "SELECT * FROM orders TABLESAMPLE BERNOULLI (100)",
        # the filter is on another table, or on no table of the query
        "SELECT o.* FROM orders o JOIN (SELECT 's1' AS store_id) k ON TRUE WHERE k.store_id = 's1'",
        "SELECT i.secret FROM orders o JOIN items i ON 1 = 1 WHERE o.store_id = 's1'",
        "SELECT * FROM orders WHERE other.store_id = 's1'",
        # several tables and a filter that names none of them
        "SELECT * FROM orders o, items i WHERE store_id = 's1'",
    ],
)
def test_tenant_filter_rejects_unscoped_queries(sql):
    assert status(TenantFilter("store_id", value="s1", dialect="postgres"), sql) == "fail"


def test_tenant_filter_without_value_accepts_any_constant():
    rule = TenantFilter("store_id")
    assert status(rule, "SELECT a FROM orders WHERE store_id = 'anything'") == "pass"
    assert status(rule, "SELECT a FROM orders WHERE store_id = other_id") == "fail"


@pytest.mark.parametrize(
    ("value", "sql_value", "expected"),
    [
        (-1, "-1", "pass"),
        (1, "1.0", "pass"),
        (1, "01", "pass"),
        (1, "'1'", "pass"),
        (1, "2", "fail"),
    ],
)
def test_tenant_filter_compares_numbers_by_value(value, sql_value, expected):
    sql = f"SELECT a FROM orders WHERE store_id = {sql_value}"
    assert status(TenantFilter("store_id", value=value), sql) == expected


def test_tenant_filter_can_be_limited_to_the_tables_that_have_the_column():
    rule = TenantFilter("store_id", tables=["orders"])
    assert status(rule, "SELECT name FROM categories") == "pass"
    assert status(rule, "SELECT amount FROM orders") == "fail"
    assert status(rule, "SELECT amount FROM archive.orders") == "fail"
    joined = "SELECT o.* FROM orders o JOIN categories c ON c.id = o.category_id WHERE {}"
    assert status(rule, joined.format("o.store_id = 's1'")) == "pass"
    assert status(rule, joined.format("c.store_id = 's1'")) == "fail"
    # A bare string names one table.
    assert status(TenantFilter("store_id", tables="orders"), "SELECT a FROM orders") == "fail"


# ---- AllowedTables --------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a FROM orders",
        "SELECT * FROM orders o JOIN orders p ON o.id = p.id",
        "WITH recent AS (SELECT * FROM orders) SELECT * FROM recent",
    ],
)
def test_allowed_tables_accepts_listed_tables(sql):
    assert status(AllowedTables(["orders"]), sql) == "pass"
    assert status(AllowedTables("orders"), sql) == "pass"  # a bare string is one table


@pytest.mark.parametrize(
    ("sql", "dialect"),
    [
        ("SELECT a FROM pg_catalog.pg_tables", "postgres"),
        ("SELECT * FROM other_tenant.orders", None),
        ("SELECT * FROM otherdb.public.orders", None),
        ("SELECT * FROM `proj.other.orders`", "bigquery"),
        ("WITH customers AS (SELECT * FROM customers) SELECT * FROM customers", None),
        ("WITH a AS (SELECT * FROM b), b AS (SELECT 1) SELECT * FROM a", None),
        (
            "SELECT * FROM secret WHERE a IN (WITH secret AS (SELECT 1 AS a) SELECT a FROM secret)",
            None,
        ),
        ("SELECT * FROM pg_read_file('/etc/passwd')", "postgres"),
        ("SELECT * FROM dblink('host=x', 'select 1') AS t(a int)", "postgres"),
        ("SELECT * FROM orders /*! UNION SELECT * FROM secret */", "mysql"),
    ],
)
def test_allowed_tables_rejects_everything_else(sql, dialect):
    assert status(AllowedTables(["orders"], dialect=dialect), sql) == "fail"


def test_allowed_tables_matches_qualified_names_as_written():
    rule = AllowedTables(["public.orders", "generate_series"], dialect="postgres")
    assert status(rule, "SELECT * FROM public.orders") == "pass"
    assert status(rule, "SELECT * FROM orders") == "fail"
    assert status(rule, "SELECT * FROM generate_series(1, 3) g") == "pass"


def test_allowed_tables_from_another_stage():
    view = [{"kind": "route", "name": "view", "output": "orders_view"}]
    rule = AllowedTables(from_stage="view")
    assert status(rule, "SELECT a FROM orders_view", view) == "pass"
    assert status(rule, "SELECT a FROM orders", view) == "fail"
    assert status(rule, "SELECT a FROM orders_view") == "fail"  # stage missing from trace


# ---- AllowedColumns -------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT category, SUM(amount) AS total FROM t WHERE store_id = 's1' ORDER BY total",
        "SELECT category, SUM(amount) AS total FROM t GROUP BY category HAVING total > 5",
        "SELECT COUNT(*) FROM t",
        "SELECT x.total FROM (SELECT SUM(amount) AS total FROM t) x ORDER BY total",
        "WITH c AS (SELECT category AS cat, amount FROM t) SELECT cat, amount FROM c",
        "WITH c(cat, amt) AS (SELECT category, amount FROM t) SELECT cat, c.amt FROM c",
        "SELECT o.x FROM (SELECT i.y AS x FROM (SELECT amount AS y FROM t) i) o",
        "SELECT a.total FROM (SELECT amount AS total FROM t UNION ALL SELECT amount FROM u) a",
        "SELECT n FROM UNNEST(ARRAY[1, 2]) AS n",
    ],
)
def test_allowed_columns_accepts_listed_columns_and_inner_outputs(sql):
    menu = [{"kind": "retrieve", "output": ["category", "amount"]}]
    rule = AllowedColumns(from_stage="retrieve", extra=["store_id"])
    assert status(rule, sql, menu) == "pass"


@pytest.mark.parametrize(
    ("sql", "dialect"),
    [
        ("SELECT invented FROM t", None),
        ("SELECT * FROM t", None),
        ("SELECT t.* FROM t", None),
        # an alias does not launder the column it is named after
        ("SELECT MAX(card_number) AS card_number FROM t", None),
        ("SELECT card_number AS card_number FROM t", None),
        ("SELECT category AS card_number FROM t WHERE card_number LIKE '4%'", None),
        ("SELECT SUM(amount) OVER (ORDER BY card_number) AS card_number FROM t", None),
        # stars inside functions read every column
        ("SELECT to_jsonb(o.*) FROM t o", "postgres"),
        ("SELECT OBJECT_CONSTRUCT(*) FROM t", "snowflake"),
        ("SELECT category FROM t JOIN u USING (secret_id)", None),
        # a name the inner query does not output is a real column, here of the outer table
        ("SELECT (SELECT card_number FROM (SELECT 1 AS one) d) FROM t", None),
        ("SELECT x.card_number FROM (SELECT amount FROM t) x", None),
        ("SELECT (SELECT card_number FROM UNNEST(ARRAY[1]) AS u) FROM t", "postgres"),
    ],
)
def test_allowed_columns_rejects_everything_else(sql, dialect):
    assert status(AllowedColumns(["category", "amount"], dialect=dialect), sql) == "fail"


def test_name_source_needs_exactly_one_input():
    with pytest.raises(ValueError):
        AllowedTables()
    with pytest.raises(ValueError):
        AllowedColumns(columns=["a"], from_stage="retrieve")


# ---- NoJoins / NoSubqueries -----------------------------------------------------------


def test_no_joins():
    assert status(NoJoins(), "SELECT a FROM t") == "pass"
    assert status(NoJoins(), "SELECT a FROM t JOIN u ON t.id = u.id") == "fail"
    assert status(NoJoins(), "SELECT a FROM t, u WHERE t.id = u.id") == "fail"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a FROM t WHERE a IN (SELECT a FROM u)",
        "WITH c AS (SELECT a FROM t) SELECT a FROM c",
        "SELECT a FROM t UNION SELECT a FROM u",
    ],
)
def test_no_subqueries(sql):
    assert status(NoSubqueries(), "SELECT a FROM t") == "pass"
    assert status(NoSubqueries(), sql) == "fail"


# ---- inputs from the second review ----------------------------------------------------


@pytest.mark.parametrize("dialect", ["mysql", "postgres"])
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id, (TABLE secret) AS leaked FROM orders WHERE store_id = 's1'",
        "SELECT id FROM orders WHERE store_id = 's1' AND 42 = ANY (TABLE secret)",
    ],
)
def test_table_shorthand_inside_an_expression_fails_every_rule(sql, dialect):
    for rule in (
        ReadOnly(dialect=dialect),
        TenantFilter("store_id", value="s1", dialect=dialect),
        AllowedTables(["orders"], dialect=dialect),
        NoSubqueries(dialect=dialect),
    ):
        assert status(rule, sql) == "fail"


@pytest.mark.parametrize(
    ("sql", "dialect"),
    [
        ("SELECT id FROM orders WHERE store_id = 's1' --1 OR 1=1", "mysql"),
        ("SELECT id FROM orders WHERE store_id = 's1' /*M! OR 1=1 */", "mysql"),
        (
            "SELECT id FROM orders WHERE store_id = 's1' /*! OR 1=1 */",
            "mysql, normalization_strategy = case_sensitive",
        ),
        (
            "SELECT id FROM orders WHERE store_id = 's1' AND x > {#\n d '2020-01-01'} "
            "OR 1=1 OR x > #}\n '2020-01-01'",
            "mysql",
        ),
        ("SELECT id FROM orders WHERE store_id = 's1' /* /* */ OR 1=1 -- */", None),
        ("SELECT id FROM orders WHERE store_id = 's1' AND n = 'a\\' AND n = ' OR 1=1 -- '", None),
    ],
)
def test_text_the_parser_and_the_database_read_differently_fails(sql, dialect):
    assert status(TenantFilter("store_id", value="s1", dialect=dialect), sql) == "fail"


def test_a_comment_after_the_statement_is_not_a_second_statement():
    sql = "SELECT a FROM orders WHERE store_id = 's1'; -- done"
    assert status(ReadOnly(), sql) == "pass"
    assert status(TenantFilter("store_id", value="s1"), sql) == "pass"


def test_input_that_makes_the_parser_raise_is_a_failure_not_a_crash():
    assert status(ReadOnly(), "FROM t :y AT") == "fail"
    rows_from = "SELECT * FROM ROWS FROM (generate_series(1, 2), f()) AS t"
    assert status(AllowedTables(["orders"], dialect="postgres"), rows_from) == "fail"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM IDENTIFIER('orders')",
        "SELECT x FROM TABLE('orders')",
    ],
)
def test_dynamic_table_references_fail(sql):
    assert status(TenantFilter("store_id", dialect="snowflake"), sql) == "fail"
    assert status(AllowedTables(["orders"], dialect="snowflake"), sql) == "fail"


@pytest.mark.parametrize(
    "sql",
    [
        # ordinary multi-CTE style
        "WITH a AS (SELECT id FROM orders WHERE store_id = 's1'), "
        "b AS (SELECT order_id FROM items WHERE store_id = 's1') "
        "SELECT * FROM a JOIN b ON a.id = b.order_id",
        "WITH a AS (SELECT id FROM orders WHERE store_id = 's1') "
        "SELECT * FROM items WHERE store_id = 's1' AND order_id IN (SELECT id FROM a)",
        # the filter for a joined table sits in that join's ON
        "SELECT o.a, i.b FROM orders o LEFT JOIN items i "
        "ON i.order_id = o.id AND i.store_id = 's1' WHERE o.store_id = 's1'",
        "SELECT o.a, i.b FROM orders o JOIN items i "
        "ON i.order_id = o.id AND i.store_id = 's1' WHERE o.store_id = 's1'",
        # joined on the tenant column, in each join type
        "SELECT o.a FROM orders o JOIN items i USING (store_id) WHERE o.store_id = 's1'",
        "SELECT o.a FROM orders o LEFT JOIN items i ON i.store_id = o.store_id "
        "WHERE o.store_id = 's1'",
        "SELECT o.a FROM orders o FULL JOIN items i ON i.store_id = o.store_id "
        "WHERE o.store_id = 's1'",
        "SELECT o.a FROM orders o RIGHT JOIN items i ON i.store_id = o.store_id "
        "WHERE i.store_id = 's1'",
    ],
)
def test_tenant_filter_accepts_common_join_and_cte_shapes(sql):
    assert status(TenantFilter("store_id", value="s1", dialect="postgres"), sql) == "pass"


@pytest.mark.parametrize(
    "sql",
    [
        # an outer join's ON does not restrict the rows of the preserved side
        "SELECT b.id FROM a CROSS JOIN b LEFT JOIN c "
        "ON a.store_id = b.store_id AND c.store_id = a.store_id WHERE a.store_id = 's1'",
        "SELECT a FROM orders o LEFT JOIN items i ON i.store_id = o.store_id AND i.store_id = 's1'",
        "SELECT a FROM orders o RIGHT JOIN items i ON i.store_id = 's1' WHERE o.store_id = 's1'",
        "SELECT a FROM a JOIN b ON a.store_id = 's1' RIGHT JOIN c ON c.store_id = a.store_id",
        # two sources that cannot be told apart by name
        "SELECT O.id FROM orders O CROSS JOIN orders o WHERE o.store_id = 's1'",
        "SELECT * FROM db1.orders, db2.orders WHERE db1.orders.store_id = 's1'",
        # an alias column list makes `store_id` some other column
        "SELECT * FROM orders AS o(a, store_id, c) WHERE o.store_id = 's1'",
    ],
)
def test_tenant_filter_rejects_joins_that_leave_a_table_unfiltered(sql):
    assert status(TenantFilter("store_id", value="s1", dialect="postgres"), sql) == "fail"


@pytest.mark.parametrize(
    ("condition", "dialect", "expected"),
    [
        ("store_id = 's1'", None, "pass"),
        ("store_id = 5", None, "pass"),
        ("store_id = ?", None, "pass"),
        ("store_id = :store", None, "pass"),
        ("store_id = N's1'", "tsql", "pass"),
        ("store_id = _utf8mb4's1'", "mysql", "pass"),
        # to the database these are (store_id = 's1') <operator> ..., true for every row
        ("store_id = 's1' IS NOT NULL", "sqlite", "fail"),
        ("store_id = 's1' IN (0, 1)", "sqlite", "fail"),
        ("store_id = 's1' BETWEEN 0 AND 1", "sqlite", "fail"),
        ("store_id = 's1' LIKE '%'", "sqlite", "fail"),
        ("store_id = 's1' < 2", "mysql", "fail"),
        ("store_id = ALL(ARRAY[]::text[])", "postgres", "fail"),
        ("store_id = lower('S1')", None, "fail"),
    ],
)
def test_tenant_value_must_be_a_literal_or_a_parameter(condition, dialect, expected):
    sql = f"SELECT a FROM orders WHERE {condition}"
    assert status(TenantFilter("store_id", dialect=dialect), sql) == expected


@pytest.mark.parametrize(
    ("sql", "dialect"),
    [
        # a CTE the query does not read from excuses nothing
        ("WITH c AS (SELECT 1 AS card_number) SELECT card_number FROM t", None),
        (
            "WITH c AS (SELECT amount AS card_number FROM t) "
            "SELECT c.card_number, (SELECT MAX(card_number) FROM t) FROM c",
            None,
        ),
        # inside a function the name is the real column, not the alias
        (
            "SELECT category AS card_number FROM t GROUP BY category HAVING MAX(card_number) > 5",
            None,
        ),
        ("SELECT x.card_number FROM t x CROSS JOIN (SELECT 1 AS card_number) X", None),
        ("SELECT o.category, o.amount FROM users AS o(category, amount)", "postgres"),
        ("SELECT COLUMNS('.*') FROM t", "duckdb"),
    ],
)
def test_allowed_columns_rejects_names_that_only_look_like_aliases(sql, dialect):
    assert status(AllowedColumns(["category", "amount"], dialect=dialect), sql) == "fail"


def test_allowed_columns_accepts_common_shapes():
    rule = AllowedColumns(["category", "amount"], dialect="postgres")
    assert status(rule, "SELECT category FROM t o WHERE EXISTS (SELECT * FROM t i)") == "pass"
    by_month = (
        "SELECT DATE_TRUNC('month', amount) AS month, COUNT(*) AS n FROM t "
        "GROUP BY month HAVING n > 5 ORDER BY month DESC"
    )
    assert status(rule, by_month) == "pass"
    # A retriever that emits table.column also allows the bare column.
    menu = [{"kind": "retrieve", "output": ["orders.amount", "orders.category"]}]
    from_stage = AllowedColumns(from_stage="retrieve")
    assert status(from_stage, "SELECT amount FROM orders", menu) == "pass"
    assert status(from_stage, "SELECT secret FROM orders", menu) == "fail"


@pytest.mark.parametrize(
    ("sql", "dialect"),
    [
        ("SELECT a FROM t UNION ALL (VALUES (1), (2))", None),
        ("SELECT * FROM (VALUES (1), (2)) v(x)", None),
        ("SELECT a FROM t WHERE a IN (VALUES (1))", None),
        ("WITH c AS (VALUES (1)) SELECT * FROM c", "postgres"),
        ("SELECT a FROM t UNION VALUES (1)", "postgres"),
    ],
)
def test_no_subqueries_also_rejects_values_lists(sql, dialect):
    assert status(NoSubqueries(dialect=dialect), sql) == "fail"


@pytest.mark.parametrize(
    ("name", "sql"),
    [
        ("now", "SELECT NOW()"),
        ("date_part", "SELECT date_part('year', d) FROM t"),
        ("strpos", "SELECT strpos('abc', 'b')"),
        ("position", "SELECT position('b' in 'abc')"),
        ("unnest", "SELECT unnest(ARRAY[1])"),
    ],
)
def test_forbidden_functions_match_the_name_as_written(name, sql):
    assert status(ReadOnly(forbidden_functions=[name], dialect="postgres"), sql) == "fail"
    assert status(ReadOnly(dialect="postgres"), sql) == "pass"
