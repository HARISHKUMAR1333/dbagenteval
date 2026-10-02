"""SQL rules: checks on the generated query that need no LLM and no expected value.

Each rule reads the SQL from the trace's generate stages (all of them, or the one named
by `stage`), parses it with sqlglot, and passes or fails it. A rule is skipped when the
trace has no SQL, which is normal for a question the agent refused or answered without
querying.

The rules are deliberately strict: SQL they cannot analyse fails, it does not pass. They
are still a static reading of the SQL by a parser that is not your database, so they are
an evaluation aid, not a security boundary. Set `dialect` to your database, and enforce
access in the database itself (a read-only role, row-level security).
"""

from __future__ import annotations

import re
from abc import abstractmethod
from collections.abc import Iterable, Iterator
from decimal import Decimal, InvalidOperation
from typing import Any, cast

import sqlglot
from sqlglot import exp
from sqlglot.dialects.mysql import MySQL
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import Scope, traverse_scope

from dbagenteval.case import Case
from dbagenteval.checks import Check, _as_items, _short
from dbagenteval.scorecard import FAIL, PASS, SKIP, CheckResult
from dbagenteval.trace import Trace


def _node_types(*names: str) -> tuple[type, ...]:
    """sqlglot node classes by name; names vary a little between sqlglot versions."""
    return tuple(getattr(exp, name) for name in names if hasattr(exp, name))


_WRITE_NODES = _node_types(
    "Insert", "Update", "Delete", "Merge", "Drop", "Create", "Alter", "AlterTable",
    "TruncateTable", "Copy", "Grant", "Set", "Command", "Into", "Lock",
)  # fmt: skip
_NOT_FLAT_NODES = _node_types(
    "SetOperation", "Union", "Except", "Intersect", "With", "Subquery", "Values"
)
_STRING_NODES = _node_types("National")  # N'text'
_PARAMETER_NODES = _node_types("Placeholder", "Parameter")

# A starter set of functions that read files, sleep, change state or reach other
# servers (Postgres and MySQL). Extend it for your database.
DEFAULT_FORBIDDEN_FUNCTIONS = frozenset(
    {
        "pg_sleep", "pg_sleep_for", "pg_sleep_until", "sleep", "benchmark",
        "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file", "load_file",
        "lo_import", "lo_export", "lo_unlink",
        "dblink", "dblink_exec", "dblink_connect",
        "nextval", "setval", "set_config",
        "pg_terminate_backend", "pg_cancel_backend", "pg_advisory_lock",
    }
)  # fmt: skip

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_TABLE_STATEMENT = re.compile(r"^\s*\(?\s*TABLE\s", re.IGNORECASE)
_CALL = re.compile(r"\s*([A-Za-z_][\w.]*)\s*\(")
_NESTED_COMMENT = re.compile(r"/\*(?:(?!\*/)[\s\S])*?/\*")
_DASHES_WITHOUT_SPACE = re.compile(r"--(?=\S)")


def _names(value: str | Iterable[str]) -> list[str]:
    """Lower-cased names; a bare string is one name, not its characters."""
    return [n.lower() for n in ([value] if isinstance(value, str) else value)]


class SqlRule(Check):
    """Base class for rules over the generated SQL. Implement `problem`."""

    def __init__(self, stage: str | None = None, dialect: str | None = None) -> None:
        # Resolving the dialect here makes a typo fail now, not once per case.
        self._runs_comments = dialect is not None and isinstance(
            sqlglot.Dialect.get_or_raise(dialect), MySQL
        )
        self.stage = stage
        self.dialect = dialect

    def run(self, trace: Trace, case: Case) -> CheckResult:
        generate = [s for s in trace.stages if s.kind == "generate"]
        if self.stage is None:
            stages = generate  # a retried query is checked on every attempt
        else:
            named = trace.stage(self.stage)
            if named is None:
                if generate:  # there is SQL, just not under this name: likely a typo
                    found = [s.name for s in generate]
                    return self.result(
                        self.stage,
                        FAIL,
                        f"stage {self.stage!r} is not in the trace (found {found})",
                    )
                return self.result(self.stage, SKIP, "no SQL to check")
            stages = [named]

        checked = None
        for stage in stages:
            sql = stage.output
            if sql is None or (isinstance(sql, str) and not sql.strip()):
                continue
            if not isinstance(sql, str):
                return self.result(
                    stage.name, FAIL, f"the stage output must be the SQL text, got {_short(sql)!r}"
                )
            problem = self._check(sql, trace)
            if problem:
                return self.result(stage.name, FAIL, problem)
            checked = stage
        if checked is None:
            return self.result(self.stage or "generate", SKIP, "no SQL to check")
        return self.result(checked.name, PASS)

    def _check(self, sql: str, trace: Trace) -> str | None:
        hidden = self._hidden_sql(sql)
        if hidden:
            return hidden
        try:
            parsed = sqlglot.parse(sql, read=self.dialect)
            # Newer sqlglot types parse() with a wider base class than older releases
            # have, and turns a trailing `; -- note` into a Semicolon statement.
            statements = cast(
                "list[exp.Expression]",
                [s for s in parsed if s is not None and type(s).__name__ != "Semicolon"],
            )
            if not statements:
                return "no SQL statement found"
            for statement in statements:
                if not isinstance(statement, exp.Query):
                    return f"not a SELECT statement ({type(statement).__name__})"
                unreadable = _unreadable_source(statement, self.dialect)
                if unreadable:
                    return unreadable
            return self.text_problem(sql) or self.problem(statements, trace)
        except RecursionError:
            return "SQL is too deeply nested to analyse"
        except SqlglotError as e:
            return f"SQL does not parse: {_short(_ANSI.sub('', str(e)))}"
        except Exception as e:  # noqa: BLE001 — the parser can raise anything on odd input
            return f"SQL could not be analysed ({type(e).__name__}: {_short(e)})"

    def _hidden_sql(self, sql: str) -> str | None:
        """Text the parser reads as a comment or string but a database may execute."""
        if _TABLE_STATEMENT.match(sql):
            return "TABLE statements are not supported; write SELECT * FROM ..."
        if "{#" in sql:
            return "contains '{#', which is read as a comment here but not by databases"
        if (self.dialect is None or self._runs_comments) and ("/*!" in sql or "/*M!" in sql):
            return "contains a MySQL executable comment (/*! ... */), which hides SQL"
        if self._runs_comments and _DASHES_WITHOUT_SPACE.search(sql):
            return "contains '--' with no space after it, which MySQL does not read as a comment"
        if self.dialect is None:
            # Databases disagree on these; without a dialect the reading is a guess.
            if _NESTED_COMMENT.search(sql):
                return "contains a nested block comment; set `dialect` so it is read correctly"
            if "\\'" in sql or '\\"' in sql:
                return "contains a backslash before a quote; set `dialect` so it is read correctly"
        return None

    def text_problem(self, sql: str) -> str | None:
        """A problem found in the SQL text itself, before the parsed form is checked."""
        return None

    @abstractmethod
    def problem(self, statements: list[exp.Expression], trace: Trace) -> str | None:
        """What is wrong with the SQL, or None when it is fine. Every statement is a
        query; anything else has already failed."""


class ReadOnly(SqlRule):
    """Exactly one statement, and it only reads: no writes, DDL, SELECT INTO, row locks
    or forbidden functions.

    `forbidden_functions` defaults to a starter set for Postgres and MySQL. It is a
    deny-list, so it cannot be complete.
    """

    name = "read_only"

    def __init__(
        self,
        forbidden_functions: str | Iterable[str] = DEFAULT_FORBIDDEN_FUNCTIONS,
        stage: str | None = None,
        dialect: str | None = None,
    ) -> None:
        super().__init__(stage, dialect)
        self.forbidden_functions = set(_names(forbidden_functions))

    def text_problem(self, sql: str) -> str | None:
        # Read the names as written: the parser renames some functions (NOW() becomes
        # CURRENT_TIMESTAMP), which would hide them from a list of names.
        tokens = sqlglot.Dialect.get_or_raise(self.dialect).tokenize(sql)
        called = {
            token.text.lower()
            for token, following in zip(tokens, tokens[1:], strict=False)
            if following.token_type == sqlglot.TokenType.L_PAREN
        }
        forbidden = called & self.forbidden_functions
        return f"calls forbidden function {sorted(forbidden)[0]}()" if forbidden else None

    def problem(self, statements: list[exp.Expression], trace: Trace) -> str | None:
        if len(statements) != 1:
            return f"expected exactly one statement, found {len(statements)}"
        tree = statements[0]
        write = tree.find(*_WRITE_NODES)
        if write is not None:
            return f"contains a write, DDL or locking operation ({type(write).__name__})"
        for func in tree.find_all(exp.Func):  # also catches calls written without ( )
            forbidden = _function_names(func, self.dialect) & self.forbidden_functions
            if forbidden:
                return f"calls forbidden function {sorted(forbidden)[0]}()"
        return None


class TenantFilter(SqlRule):
    """Every table the SQL reads is filtered to one tenant.

    For each SELECT, each table it reads needs a condition `column = value` (or
    `column IN (...)`) among the top-level AND conditions of the WHERE clause, so
    `tenant_id = 'a' OR 1 = 1` does not count. The value must be a literal or a bind
    parameter. In a SELECT that reads several tables the condition must name its table
    (`o.tenant_id = 'a'`).

    Two join forms are also accepted: a table joined on the tenant column
    (`ON i.tenant_id = o.tenant_id`) is covered by the other table's filter, and a
    filter in a join's ON covers the table that join adds (`LEFT JOIN items i ON ... AND
    i.tenant_id = 'a'`).

    Pass `value` to require one specific tenant. Pass `tables` to name the tables that
    carry the tenant column, when some tables (lookups, child tables reached through a
    foreign key) have none; only those tables are then checked.
    """

    name = "tenant_filter"

    def __init__(
        self,
        column: str,
        value: Any = None,
        tables: str | Iterable[str] | None = None,
        stage: str | None = None,
        dialect: str | None = None,
    ) -> None:
        super().__init__(stage, dialect)
        self.column = column.lower()
        self.value = None if value is None else str(value)
        self.tables = None if tables is None else set(_names(tables))

    def problem(self, statements: list[exp.Expression], trace: Trace) -> str | None:
        for tree in statements:
            for select, tables, scope in _selects(tree):
                if not any(self._applies(t) for t in tables):
                    continue
                if select is None:
                    return f"cannot tell which query reads {_dotted(tables[0])!r}"
                problem = self._unfiltered(select, tables, scope)
                if problem:
                    return problem
        return None

    def _applies(self, table: exp.Table) -> bool:
        if self.tables is None:
            return True
        # A bare name covers that table in every schema, so `orders` also guards
        # `archive.orders`.
        return _dotted(table) in self.tables or table.name.lower() in self.tables

    def _unfiltered(
        self, select: exp.Select, tables: list[exp.Table], scope: Scope | None
    ) -> str | None:
        """Why some table of this SELECT is not tenant-filtered, or None."""
        sources = _source_names(tables, scope)
        clash = _source_problem(tables, sources)
        if clash:
            return clash
        needed = {t.alias_or_name.lower() for t in tables if self._applies(t)}

        where = select.args.get("where")
        conditions = list(_conjuncts(where.this)) if where is not None else []
        # `strong`: every result row has this table's tenant column equal to a constant.
        strong: set[str] = set()
        unqualified = False
        for condition in conditions:
            column = self._tenant_condition(condition)
            if column is None:
                continue
            if column.table:
                strong.add(column.table.lower())
            elif len(sources) == 1:
                strong.add(sources[0])  # the only source there is
            else:
                unqualified = True
        links = [pair for pair in map(self._tenant_link, conditions) if pair]

        joins = [self._join_facts(join, sources) for join in select.args.get("joins") or []]
        # An inner join's ON acts like WHERE, unless a later RIGHT or FULL join can bring
        # back rows it dropped.
        if not any(side in ("RIGHT", "FULL") for side, *_ in joins):
            for side, _own, constants, pairs in joins:
                if not side:
                    strong |= constants
                    links += pairs
        grew = True
        while grew:  # a table joined on the tenant column shares the other's filter
            grew = False
            for left, right in links:
                if (left in strong) != (right in strong):
                    strong |= {left, right}
                    grew = True

        # `covered`: wherever the table contributes a row, that row is the tenant's. An
        # outer join's ON only restricts the side that is not preserved.
        covered = set(strong)
        for side, own, constants, pairs in joins:
            if side == "LEFT" and own in constants:
                covered.add(own)
            elif side == "RIGHT":
                covered |= constants - {own}
            for left, right in pairs:
                if side == "RIGHT" or (side in ("LEFT", "FULL") and own in (left, right)):
                    if left in strong:
                        covered.add(right)
                    if right in strong:
                        covered.add(left)

        missing = sorted(needed - covered)
        if not missing:
            return None
        wanted = self.column + (f" = {self.value!r}" if self.value is not None else "")
        hint = "; name the table in the filter, as in alias.column = value" if unqualified else ""
        return f"no {wanted} filter on {missing}{hint}"

    def _join_facts(
        self, join: exp.Join, sources: list[str]
    ) -> tuple[str, str, set[str], list[tuple[str, str]]]:
        """(side, the table this join adds, tables its ON pins to a constant, tables its
        ON equates on the tenant column)."""
        own = join.this.alias_or_name.lower() if join.this is not None else ""
        on = join.args.get("on")
        conditions = list(_conjuncts(on)) if on is not None else []
        constants = set()
        for condition in conditions:
            column = self._tenant_condition(condition)
            if column is not None and column.table:
                constants.add(column.table.lower())
        pairs = [pair for pair in map(self._tenant_link, conditions) if pair]
        using = {identifier.name.lower() for identifier in join.args.get("using") or []}
        if self.column in using and len(sources) == 2 and own in sources:
            pairs.append((own, next(s for s in sources if s != own)))
        return (join.side or "").upper(), own, constants, pairs

    def _tenant_condition(self, node: exp.Expression) -> exp.Column | None:
        """The tenant column, when `node` pins it to a constant."""
        if isinstance(node, exp.EQ):
            for column_side, value_side in (
                (node.this, node.expression),
                (node.expression, node.this),
            ):
                column = self._tenant_column(column_side)
                if column is not None and self._is_value(value_side):
                    return column
        elif isinstance(node, exp.In) and not node.args.get("query"):
            column = self._tenant_column(node.this)
            values = node.expressions
            if column is not None and values and all(self._is_value(v) for v in values):
                return column
        return None

    def _tenant_link(self, node: exp.Expression) -> tuple[str, str] | None:
        """The two table names, when `node` is `a.tenant = b.tenant`."""
        if not isinstance(node, exp.EQ):
            return None
        left, right = self._tenant_column(node.this), self._tenant_column(node.expression)
        if left is None or right is None or not left.table or not right.table:
            return None
        return left.table.lower(), right.table.lower()

    def _tenant_column(self, node: exp.Expression) -> exp.Column | None:
        node = _unwrap(node)
        if isinstance(node, exp.Column) and node.name.lower() == self.column:
            return node
        return None

    def _is_value(self, node: exp.Expression) -> bool:
        """A literal or bind parameter, equal to `value` when one was given. Anything
        else is refused: `col = 'a' IS NOT NULL` is one comparison to the parser and
        `(col = 'a') IS NOT NULL` to the database."""
        node = _unwrap(node)
        negative = isinstance(node, exp.Neg)
        if negative:
            node = _unwrap(node.this)
        if isinstance(node, _PARAMETER_NODES):
            return self.value is None and not negative
        if isinstance(node, _STRING_NODES):
            return not negative and self.value in (None, node.name)
        if isinstance(node, exp.Literal):
            text = ("-" if negative else "") + node.name
            if self.value is None or text == self.value:
                return True
            if node.is_string:
                return False
            try:  # 1, 1.0 and 01 are the same number
                return Decimal(text) == Decimal(self.value)
            except InvalidOperation:
                return False
        # SQLite and MySQL read a double-quoted "abc" as a string when no such column
        # exists, and agents write it; sqlglot sees an identifier.
        if isinstance(node, exp.Column) and not node.table and not negative:
            quoted = bool(node.this.args.get("quoted"))
            return quoted and self.value is not None and node.name == self.value
        return False


class AllowedTables(SqlRule):
    """The SQL reads only allowed tables: a fixed set, or the output of another stage
    (for example the view a routing stage picked).

    Names are compared as written: `orders` allows `orders` but not `archive.orders`;
    list `archive.orders` to allow that. A function used as a table (`dblink(...)`,
    `generate_series(...)`) is a source too, allowed by its function name.
    """

    name = "allowed_tables"

    def __init__(
        self,
        tables: str | Iterable[str] | None = None,
        from_stage: str | None = None,
        stage: str | None = None,
        dialect: str | None = None,
    ) -> None:
        super().__init__(stage, dialect)
        self.source = _NameSource("tables", tables, from_stage)

    def problem(self, statements: list[exp.Expression], trace: Trace) -> str | None:
        allowed = self.source.resolve(trace)
        if isinstance(allowed, str):
            return allowed
        extra: set[str] = set()
        for tree in statements:
            for _, table in _real_tables(tree, _scopes(tree)):
                if isinstance(table.this, exp.Identifier):
                    if _dotted(table) not in allowed:
                        extra.add(_dotted(table))
                elif not _function_names(table.this, self.dialect) & allowed:
                    extra.add(table.this.sql(dialect=self.dialect))
        return f"reads table(s) outside the allowed set: {sorted(extra)}" if extra else None


class AllowedColumns(SqlRule):
    """The SQL uses only allowed columns: a fixed set, or the output of another stage
    (for example the columns a retrieve stage put on the menu). A stage's `table.column`
    entries also allow the bare `column`.

    A column of a subquery or CTE is not checked by name, because the inner query that
    produces it is. An alias of the SELECT may be used by its bare name in GROUP BY,
    HAVING, ORDER BY and QUALIFY; if a real column has the same name, some databases
    group or filter on the real column there. `*` is not allowed except in COUNT(*) and
    EXISTS (SELECT * ...), because it reads columns nobody listed.
    """

    name = "allowed_columns"

    def __init__(
        self,
        columns: str | Iterable[str] | None = None,
        from_stage: str | None = None,
        extra: str | Iterable[str] = (),
        stage: str | None = None,
        dialect: str | None = None,
    ) -> None:
        super().__init__(stage, dialect)
        self.source = _NameSource("columns", columns, from_stage)
        self.extra = set(_names(extra))

    def problem(self, statements: list[exp.Expression], trace: Trace) -> str | None:
        allowed = self.source.resolve(trace)
        if isinstance(allowed, str):
            return allowed
        if self.source.from_stage is not None:
            allowed = allowed | {name.rsplit(".", 1)[-1] for name in allowed}
        used: set[str] = set()
        for tree in statements:
            if any(not _is_harmless_star(star) for star in tree.find_all(exp.Star)):
                return "uses * (it reads columns nobody listed)"
            if tree.find(*_node_types("Columns")) is not None or any(
                f.name.lower() == "columns" for f in tree.find_all(exp.Anonymous)
            ):
                return "uses COLUMNS(...), which reads columns nobody listed"
            scopes = _scopes(tree)
            for _select, tables, scope in _selects(tree, scopes):
                clash = _source_problem(tables, _source_names(tables, scope))
                if clash:
                    return clash
            inner: dict[int, dict[str, set[str]]] = {}
            for column in tree.find_all(exp.Column):
                if column.name and not _is_inner_output(column, scopes, inner):
                    used.add(column.name.lower())
            for join in tree.find_all(exp.Join):
                used |= {identifier.name.lower() for identifier in join.args.get("using") or []}
        unknown = sorted(used - allowed - self.extra)
        return f"uses column(s) outside the allowed set: {unknown}" if unknown else None


class NoJoins(SqlRule):
    """No JOINs, for agents that query one pre-joined view or a semantic layer."""

    name = "no_joins"

    def problem(self, statements: list[exp.Expression], trace: Trace) -> str | None:
        return "contains a JOIN" if any(t.find(exp.Join) for t in statements) else None


class NoSubqueries(SqlRule):
    """Each statement is one flat SELECT: no subqueries, CTEs, UNIONs or VALUES lists."""

    name = "no_subqueries"

    def problem(self, statements: list[exp.Expression], trace: Trace) -> str | None:
        for tree in statements:
            flat = (
                isinstance(tree, exp.Select)
                and len(list(tree.find_all(exp.Select))) == 1
                and tree.find(*_NOT_FLAT_NODES) is None
            )
            if not flat:
                return "is not a single flat SELECT"
        return None


class _NameSource:
    """An allowed-name set given directly or taken from another stage's output."""

    def __init__(
        self, what: str, names: str | Iterable[str] | None, from_stage: str | None
    ) -> None:
        if (names is None) == (from_stage is None):
            raise ValueError(f"give exactly one of `{what}` or `from_stage`")
        self.names = None if names is None else set(_names(names))
        self.from_stage = from_stage

    def resolve(self, trace: Trace) -> set[str] | str:
        """The allowed names, or a message saying why they are unavailable."""
        if self.names is not None:
            return self.names
        stage = trace.stage(self.from_stage or "")
        if stage is None:
            return f"cannot check: stage {self.from_stage!r} is missing from the trace"
        return set(_names(_as_items(stage.output)))


def _unreadable_source(tree: exp.Expression, dialect: str | None) -> str | None:
    """A table reference the rules cannot name, so cannot check."""
    for column in tree.find_all(exp.Column):
        # `(TABLE secret)` reads a table, but parses as a column called TABLE.
        if column.name.upper() == "TABLE" and not column.this.args.get("quoted"):
            return "uses the TABLE shorthand, which cannot be analysed; write SELECT ... FROM"
    if tree.find(*_node_types("TableFromRows")) is not None:
        return "reads a table through TABLE(...), which cannot be analysed"
    for table in tree.find_all(exp.Table):
        source = table.this
        if source is None:
            return "reads from a source that cannot be analysed (ROWS FROM)"
        if isinstance(source, exp.Identifier):
            continue
        dynamic = type(source).__name__ == "DynamicIdentifier"
        if dynamic or _function_names(source, dialect) & {"identifier", "table"}:
            return "names a table dynamically (IDENTIFIER / TABLE), which cannot be analysed"
    return None


def _scopes(tree: exp.Expression) -> dict[int, Scope]:
    """The scope of each SELECT, keyed by the SELECT node's id. A scope knows whether a
    name in FROM is a real table or a CTE / subquery."""
    try:
        return {id(scope.expression): scope for scope in traverse_scope(tree)}
    except SqlglotError:
        return {}  # callers then treat every table as real, which is the strict reading


def _real_tables(
    tree: exp.Expression, scopes: dict[int, Scope]
) -> Iterator[tuple[exp.Select | None, exp.Table]]:
    """Every reference to a real table (not a CTE or subquery), with the SELECT that
    reads it. Walks all table nodes, so one the scopes missed is still reported."""
    for table in tree.find_all(exp.Table):
        select = table.find_ancestor(exp.Select)
        scope = scopes.get(id(select)) if select is not None else None
        if (
            scope is not None
            and not table.db
            and isinstance(scope.sources.get(table.alias_or_name), Scope)
        ):
            continue
        yield select, table


def _selects(
    tree: exp.Expression, scopes: dict[int, Scope] | None = None
) -> list[tuple[exp.Select | None, list[exp.Table], Scope | None]]:
    """Each SELECT that reads a real table (functions used as tables aside), with those
    tables and its scope."""
    scopes = _scopes(tree) if scopes is None else scopes
    grouped: dict[int, tuple[exp.Select | None, list[exp.Table], Scope | None]] = {}
    for select, table in _real_tables(tree, scopes):
        if isinstance(table.this, exp.Identifier):
            entry = grouped.setdefault(id(select), (select, [], scopes.get(id(select))))
            entry[1].append(table)
    return list(grouped.values())


def _inner_sources(scope: Scope | None) -> dict[str, Scope]:
    """The subqueries, CTEs and table functions a SELECT actually reads from, by
    lower-cased alias. A CTE that is merely visible does not count."""
    if scope is None:
        return {}
    return {
        name.lower(): source
        for name, (_node, source) in scope.selected_sources.items()
        if isinstance(source, Scope)
    }


def _source_names(tables: list[exp.Table], scope: Scope | None) -> list[str]:
    """The lower-cased name of every source of one SELECT: real tables, then the rest."""
    return [t.alias_or_name.lower() for t in tables] + list(_inner_sources(scope))


def _source_problem(tables: list[exp.Table], sources: list[str]) -> str | None:
    """Sources a column cannot be attributed to with certainty."""
    for name in sources:
        if sources.count(name) > 1:
            return f"two sources are both called {name!r}; give each a distinct alias"
    for table in tables:
        alias = table.args.get("alias")
        if isinstance(alias, exp.TableAlias) and alias.columns:
            return f"renames the columns of {table.name!r} with an alias column list"
    return None


def _is_harmless_star(star: exp.Star) -> bool:
    """COUNT(*) counts rows and EXISTS (SELECT * ...) only tests for one; neither reads
    a column's values."""
    parent = star.parent
    if isinstance(parent, exp.Count):
        return True
    return isinstance(parent, exp.Select) and isinstance(parent.parent, exp.Exists)


def _is_inner_output(
    column: exp.Column, scopes: dict[int, Scope], cache: dict[int, dict[str, set[str]]]
) -> bool:
    """True when the column is produced by a subquery, a CTE, or an alias of its own
    SELECT, rather than read from a table."""
    select = column.find_ancestor(exp.Select)
    scope = scopes.get(id(select)) if select is not None else None
    if select is None or scope is None:
        return False
    if id(select) not in cache:
        cache[id(select)] = {
            alias: _outputs(alias, source) for alias, source in _inner_sources(scope).items()
        }
    inner = cache[id(select)]
    name = column.name.lower()
    if column.table:
        return name in inner.get(column.table.lower(), ())
    # Only a name an inner query really outputs: any other unqualified name could be a
    # column of an outer query's table.
    if any(name in outputs for outputs in inner.values()):
        return True
    return _is_bare_alias_use(column, select)


def _is_bare_alias_use(column: exp.Column, select: exp.Select) -> bool:
    """True for `GROUP BY alias`, `HAVING alias > 5`, `ORDER BY alias`: the alias on its
    own, not inside a function, where it would be the real column of that name."""
    node = column.parent
    while isinstance(node, exp.Binary | exp.Paren | exp.Not | exp.Ordered):
        node = node.parent
    if not isinstance(node, exp.Group | exp.Having | exp.Order | exp.Qualify):
        return False
    if node.parent is not select:
        return False
    aliases = {e.alias.lower() for e in select.expressions if isinstance(e, exp.Alias)}
    return column.name.lower() in aliases


def _outputs(alias: str, source: Scope) -> set[str]:
    """The column names a subquery, CTE or table function exposes under `alias`."""
    expression = source.expression
    names = {alias}
    if isinstance(expression, exp.Query):
        names = {n.lower() for n in expression.named_selects}
    # `WITH c(x, y) AS (...)` and `(...) AS t(x, y)` rename the outputs.
    for node in (expression, expression.parent):
        table_alias = node.args.get("alias") if node is not None else None
        if isinstance(table_alias, exp.TableAlias):
            names |= {c.name.lower() for c in table_alias.columns}
    return names


def _dotted(table: exp.Table) -> str:
    """The table name as written, lower-cased: `orders`, `public.orders`."""
    return ".".join(part.name.lower() for part in table.parts)


def _function_names(func: exp.Func | exp.Expression, dialect: str | None) -> set[str]:
    """Every name a function call goes by, lower-cased: the name it was written with
    and, for functions sqlglot has a class for, the names sqlglot knows."""
    if isinstance(func, exp.Anonymous):
        return {func.name.lower()}
    names = {n.lower() for n in type(func).sql_names()} if isinstance(func, exp.Func) else set()
    if not isinstance(func, exp.Binary):  # AND/OR are functions to sqlglot; skip them
        written = _CALL.match(func.sql(dialect=dialect))
        if written:
            names.add(written.group(1).lower())
    return names


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Cast | exp.Paren):
        node = node.this
    # `_utf8mb4'text'` is the string with a character-set label in front.
    if type(node).__name__ == "Introducer" and node.expression is not None:
        return _unwrap(node.expression)
    return node


def _conjuncts(node: exp.Expression) -> Iterator[exp.Expression]:
    """The conditions that must ALL hold: split on AND, never on OR."""
    pending = [node]
    while pending:
        node = pending.pop()
        while isinstance(node, exp.Paren):
            node = node.this
        if isinstance(node, exp.And):
            pending += [node.expression, node.this]
        else:
            yield node
