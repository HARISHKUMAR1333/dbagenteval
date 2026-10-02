"""A tiny agent with no LLM, to show how an agent plugs into dbagenteval.

It plays the assistant of an online marketplace: each store asks questions about its own
orders. It "understands" a handful of questions through a lookup table, runs the SQL on
an in-memory SQLite database, and returns a Trace with one stage per step. Replace the
lookup with your real agent and keep the Trace.

Run it from the repository root:

    dbagenteval run examples/cases.yaml --agent examples.toy_agent:agent \
        --rules examples.toy_agent:RULES --execute examples.toy_agent:run_sql

`run_sql` is the executor: it runs a case's `reference_sql` to get the correct result.

One case fails on purpose: the "cancelled orders" query returns the right number but has
no store filter, so the tenant rule catches it.
"""

import sqlite3

from dbagenteval import NumbersGrounded, Trace
from dbagenteval.rules import AllowedTables, ReadOnly, TenantFilter

STORE = "s1"

# (store_id, category, status, amount)
_ORDERS = [
    ("s1", "Electronics", "DELIVERED", 12000),
    ("s1", "Electronics", "DELIVERED", 8000),
    ("s1", "Clothing", "CANCELLED", 5000),
    ("s1", "Clothing", "DELIVERED", 3000),
    ("s2", "Electronics", "DELIVERED", 99000),
]

# question -> (SQL, answer template filled from the first row)
_KNOWN = {
    "how many orders are delivered?": (
        "SELECT COUNT(*) AS delivered FROM orders "
        f"WHERE store_id = '{STORE}' AND status = 'DELIVERED'",
        "There are {delivered} delivered orders.",
    ),
    "what is the total order amount by category?": (
        "SELECT category, SUM(amount) AS total FROM orders "
        f"WHERE store_id = '{STORE}' GROUP BY category ORDER BY total DESC",
        "{category} has the highest order amount at {total}.",
    ),
    # Deliberate bug: no store filter. The count is still right, because only this store
    # has a cancelled order — exactly the kind of mistake a final-answer check misses.
    "how many orders are cancelled?": (
        "SELECT COUNT(*) AS cancelled FROM orders WHERE status = 'CANCELLED'",
        "There is {cancelled} cancelled order.",
    ),
}

RULES = [
    ReadOnly(),
    TenantFilter("store_id", value=STORE),
    AllowedTables(["orders"]),
    NumbersGrounded(),
]


def run_sql(sql: str) -> list[dict]:
    """Run one query on the example database and return its rows as dicts."""
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE orders (store_id TEXT, category TEXT, status TEXT, amount INT)")
    db.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", _ORDERS)
    try:
        return [dict(row) for row in db.execute(sql)]
    finally:
        db.close()


def agent(question: str) -> Trace:
    text = question.strip().lower()

    if text in ("hi", "hello"):
        return Trace(question, [
            {"kind": "route", "name": "classify", "output": "small_talk"},
            {"kind": "answer", "output": "Hi! Ask me about your orders."},
        ])  # fmt: skip
    if text not in _KNOWN:
        return Trace(question, [
            {"kind": "route", "name": "classify", "output": "refused"},
            {"kind": "answer", "output": "Sorry, I can only answer questions about orders."},
        ])  # fmt: skip

    sql, template = _KNOWN[text]
    stages = [
        {"kind": "route", "name": "classify", "output": "data_question"},
        {"kind": "retrieve", "output": ["category", "status", "amount"]},
        {"kind": "generate", "output": sql},
    ]
    try:
        rows = run_sql(sql)
    except sqlite3.Error as e:
        stages.append({"kind": "execute", "error": str(e)})
        return Trace(question, stages)
    stages.append({"kind": "execute", "rows": rows})
    stages.append({"kind": "answer", "output": template.format(**rows[0])})
    return Trace(question, stages)
