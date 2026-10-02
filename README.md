# dbagenteval

A testing tool for AI agents that answer questions from a database.

## What is this?

Some AI agents work like this:

1. A user asks a question in plain language, for example "How many orders are delivered?"
2. The agent writes a SQL query.
3. The agent runs the query on a database.
4. The agent writes an answer for the user.

The agent can make a mistake in any of these steps, and the mistake is often hard to see.
The query can run without any error and still give the wrong number.

`dbagenteval` tests such an agent. You give it questions and the correct results. It asks
your agent each question, checks what the agent did, and tells you:

- which questions passed and which failed;
- **which step failed**, and why.

## What it can find

- **A wrong result.** The agent's rows are different from the correct rows.
- **A wrong answer text.** The answer has a number that is not in the result.
- **Unsafe SQL.** The query changes data, reads a table it must not read, or forgets to
  filter by customer.
- **A correct number from a bad query.** The number is right today, but the query is
  wrong, so it will be wrong tomorrow.
- **Something that got worse.** A question passed last week and fails now.

Your agent's prompts stay private. The tool does not need to know how your agent works
inside. It only looks at what the agent gives back.

## Words used in this guide

| Word | Meaning |
|---|---|
| **Agent** | Your program that takes a question and gives an answer from a database |
| **Trace** | A record of what the agent did for one question: its SQL, its rows, its answer |
| **Stage** | One step inside a trace, for example "write the SQL" or "run the SQL" |
| **Case** | One test: a question, and the correct result for it |
| **Reference SQL** | A SQL query that you know is correct. You write it yourself |
| **Executor** | A small function that runs SQL on your database and returns the rows |
| **Rule** | A check on the agent's SQL or answer that needs no correct result, for example "the SQL must only read data" |
| **Judge** | An AI model (LLM) that you ask to check an answer. This is optional |
| **Scorecard** | The result for one question: every check, and the final verdict |
| **Verdict** | The final result of one question: `PASS`, `FAIL`, `ERROR` or `SKIP` |

## Install

You need Python 3.10 or newer.

```bash
pip install dbagenteval
```

## Your first check

This part takes about five minutes. You will make a very small agent and test it. You do
not need an AI model or a real database for this.

### Step 1: make a small agent

Create a file named `shop.py` and copy this code into it.

```python
import sqlite3

from dbagenteval import Trace

# A tiny shop database: (store_id, status, amount)
ORDERS = [
    ("s1", "DELIVERED", 1200),
    ("s1", "DELIVERED", 800),
    ("s1", "CANCELLED", 500),
    ("s2", "DELIVERED", 9900),
]


def run_sql(sql):
    """Run one SQL query and return the rows as a list of dicts."""
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE orders (store_id TEXT, status TEXT, amount INTEGER)")
    db.executemany("INSERT INTO orders VALUES (?, ?, ?)", ORDERS)
    rows = [dict(row) for row in db.execute(sql)]
    db.close()
    return rows


# A pretend agent. A real agent asks an LLM to write the SQL.
# The second query has a mistake: it forgets the store filter.
KNOWN_SQL = {
    "How many orders are delivered?": (
        "SELECT COUNT(*) AS total FROM orders WHERE store_id = 's1' AND status = 'DELIVERED'"
    ),
    "What is the total amount of delivered orders?": (
        "SELECT SUM(amount) AS total FROM orders WHERE status = 'DELIVERED'"
    ),
}


def my_agent(question):
    sql = KNOWN_SQL.get(question)
    if sql is None:
        return Trace.simple(question, answer="Sorry, I can only answer questions about orders.")
    rows = run_sql(sql)
    answer = f"The answer is {rows[0]['total']}."
    return Trace.simple(question, answer=answer, sql=sql, rows=rows)
```

This file has two functions that matter:

- `run_sql` is the **executor**. It runs a SQL query and returns the rows.
- `my_agent` is the **agent**. It takes a question and returns a **trace** with its SQL,
  its rows and its answer.

### Step 2: check one question

Create a second file named `first_check.py` in the same folder.

```python
from dbagenteval import check_question
from shop import my_agent, run_sql

card = check_question(
    my_agent,
    "How many orders are delivered?",
    "SELECT COUNT(*) FROM orders WHERE store_id = 's1' AND status = 'DELIVERED'",
    execute=run_sql,
)
print(card.summary())
```

`check_question` needs four things:

1. your agent function;
2. the question;
3. the reference SQL, which is the correct query for this question;
4. `execute`, the function that runs SQL.

Run it:

```bash
python first_check.py
```

You will see:

```
Q: How many orders are delivered?
  generate     no_error           PASS
  execute      no_error           PASS
  execute      reference_rows     PASS  1 row(s)
  answer       no_error           PASS
  VERDICT      PASS
```

How to read this output:

- Each line is one check. The first word is the stage, the second word is the check name.
- `no_error` means the stage finished without an error.
- `reference_rows` means the agent's rows are the same as the rows of the reference SQL.
- `VERDICT PASS` means every check passed.

### Step 3: find a mistake

Now change the question and the reference SQL in `first_check.py`:

```python
card = check_question(
    my_agent,
    "What is the total amount of delivered orders?",
    "SELECT SUM(amount) FROM orders WHERE store_id = 's1' AND status = 'DELIVERED'",
    execute=run_sql,
)
print(card.summary())
```

Run it again. This time it fails:

```
Q: What is the total amount of delivered orders?
  generate     no_error           PASS
  execute      no_error           PASS
  execute      reference_rows     FAIL  got 1 row(s), first: [{'total': 11900}]; the reference query gives 1 row(s), first: [{'SUM(amount)': 2000}]
  answer       no_error           PASS
  VERDICT      FAIL  (first failing stage: execute)
```

The agent returned 11900, but the correct result is 2000. The agent's query ran without
any error, so you would not see this mistake without a test. The reason is in `shop.py`:
the second query forgets `store_id = 's1'`, so it adds the orders of every store.

## Test many questions with a cases file

Checking one question in Python is good for a start. For real testing, write all your
questions in one file and run them together.

### Step 1: write the cases

Create a file named `cases.yaml` in the same folder.

```yaml
- id: delivered-count
  question: "How many orders are delivered?"
  tags: [easy]
  reference_sql: SELECT COUNT(*) FROM orders WHERE store_id = 's1' AND status = 'DELIVERED'

- id: delivered-amount
  question: "What is the total amount of delivered orders?"
  tags: [easy, money]
  reference_sql: SELECT SUM(amount) FROM orders WHERE store_id = 's1' AND status = 'DELIVERED'

- id: delivered-count-fixed
  question: "How many orders are delivered?"
  tags: [easy]
  expect:
    rows: [{total: 2}]
```

Each case starts with `-`. There are two ways to give the correct result:

- **`reference_sql`**: a correct query. The tool runs it every time, so the correct
  result is always up to date. Use this when your data changes.
- **`expect.rows`**: the correct rows, written by hand. Use this only when the data never
  changes, because the rows you wrote will become old.

### Step 2: check the file for mistakes

```bash
dbagenteval validate cases.yaml
```

```
cases.yaml: 3 case(s) OK
```

If you make a typing mistake, it tells you the line. For example, if you write `expects`
in place of `expect`:

```
bad.yaml:1: unknown field(s) ['expects']; allowed: expect, id, ordered, precision, question, reference_sql, tags
```

### Step 3: run the cases

```bash
dbagenteval run cases.yaml --agent shop:my_agent --execute shop:run_sql
```

`shop:my_agent` means "the function `my_agent` in the file `shop.py`". Run the command
from the folder that has `shop.py`.

```
PASS  delivered-count
FAIL  delivered-amount
      execute / reference_rows: got 1 row(s), first: [{'total': 11900}]; the reference query gives 1 row(s), first: [{'SUM(amount)': 2000}]
PASS  delivered-count-fixed

2 of 3 passed (67%), 1 failed, 0 errors
  first failing stage execute: 1
```

How to read this output:

- One line for each case, with its verdict and its `id`.
- Under a failed case: the stage, the check that failed, and the reason.
- At the end: how many passed, and which stage failed most often.

## Add rules

A rule checks the agent's SQL or answer. It does not need a correct result. Add this to
the end of `shop.py`:

```python
from dbagenteval import NumbersGrounded
from dbagenteval.rules import ReadOnly, TenantFilter

RULES = [
    ReadOnly(),  # the SQL only reads data
    TenantFilter("store_id"),  # the SQL filters by store
    NumbersGrounded(),  # numbers in the answer come from the rows
]
```

Run again with `--rules`:

```bash
dbagenteval run cases.yaml --agent shop:my_agent --execute shop:run_sql --rules shop:RULES
```

```
PASS  delivered-count
FAIL  delivered-amount
      generate / tenant_filter: no store_id filter on ['orders']
PASS  delivered-count-fixed

2 of 3 passed (67%), 1 failed, 0 errors
  first failing stage generate: 1
```

Now the message is more useful. Before, it said the number was wrong. Now it says **why**:
the SQL has no `store_id` filter. It also names an earlier stage, `generate` (writing the
SQL), because that is where the mistake started.

### The rules you can use

| Rule | It passes when |
|---|---|
| `ReadOnly()` | The SQL is one statement and it only reads data. It does not insert, update, delete or lock anything |
| `TenantFilter("store_id")` | Every table in the SQL is filtered by that column |
| `AllowedTables(["orders"])` | The SQL reads only the tables in your list |
| `AllowedColumns(["status", "amount"])` | The SQL uses only the columns in your list, and does not use `SELECT *` |
| `NoJoins()` | The SQL has no JOIN |
| `NoSubqueries()` | The SQL is one simple SELECT, with no query inside another query |
| `NumbersGrounded()` | Every number in the answer text is also in the result rows |

A **tenant** is one customer whose data must stay separate, for example one store or one
company. `TenantFilter` checks that the agent never forgets the filter for that customer.

More about the rules:

- Give your database type to the SQL rules, for example `ReadOnly(dialect="postgres")` or
  `dialect="mysql"`. This helps the tool read your SQL correctly.
- `TenantFilter("store_id", value="s1")` also checks that the filter uses exactly `s1`.
- `TenantFilter("store_id", tables=["orders"])` checks only the tables in the list. Use
  this when some tables do not have the column.
- When the SQL joins several tables, write the table name in the filter, like
  `o.store_id = 's1'`.
- `AllowedTables(from_stage="view")` and `AllowedColumns(from_stage="retrieve")` take the
  allowed names from another stage of the trace. See "Show every step of your agent".
- The rules are strict. When a rule cannot understand the SQL, it fails the SQL.

## See what changed between two runs

Save the result of a run in a file with `--json`:

```bash
dbagenteval run cases.yaml --agent shop:my_agent --execute shop:run_sql --json before.json
```

Now fix the agent. In `shop.py`, add the store filter to the second query:

```python
"SELECT SUM(amount) AS total FROM orders WHERE store_id = 's1' AND status = 'DELIVERED'"
```

Run again, save to a second file, and compare the two files:

```bash
dbagenteval run cases.yaml --agent shop:my_agent --execute shop:run_sql --json after.json
dbagenteval diff before.json after.json
```

```
# dbagenteval diff

**0 regressions, 1 fixes, 0 added, 0 removed**

## Fixes

- delivered-amount (FAIL -> PASS)
```

A **regression** is a case that passed before and fails now. Use `diff` each time you
change a prompt or a model, to make sure nothing got worse.

## Use it with your own agent

You need two functions: an agent function and an executor.

### The agent function

It takes the question and returns what your agent did. Use `Trace.simple` and give it
what you have:

```python
from dbagenteval import Trace


def my_agent(question):
    result = call_my_real_agent(question)  # your own code
    return Trace.simple(
        question,
        answer=result.answer,  # the text shown to the user
        sql=result.sql,  # the SQL the agent wrote
        rows=result.rows,  # the rows the SQL returned, as a list of dicts
    )
```

- If your agent gives only an answer text, return just the text: `return result.answer`.
  The tool then checks the numbers in the text against the reference result.
- If your agent is `async`, write `async def my_agent(question):`. It works the same.
- If running the SQL failed, pass the error message: `Trace.simple(question, sql=sql,
  error="the error text")`.

Rows must be a list of dicts, like `[{"total": 2}]`. Many database libraries return
tuples. The executor below shows how to change them into dicts.

### The executor

It runs SQL on your database and returns the rows as a list of dicts. This form works
with most Python database libraries:

```python
def run_sql(sql):
    connection = connect_to_my_database()  # your own code
    cursor = connection.cursor()
    cursor.execute(sql)
    columns = [column[0] for column in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    connection.close()
    return rows
```

The tool uses the executor to run your **reference SQL**. It never runs the agent's SQL
by itself. Your agent function runs the agent's SQL and puts the rows in the trace.

The tool runs up to 4 cases at the same time. Open a new connection inside `run_sql`, as
shown above, or add `--concurrency 1` to run one case at a time.

### Show every step of your agent

`Trace.simple` records three steps. If your agent has more steps, you can record each of
them. Then the tool can tell you exactly which step failed.

```python
from dbagenteval import Trace


def my_agent(question):
    result = call_my_real_agent(question)  # your own code
    return Trace(
        question,
        [
            {"kind": "route", "name": "classify", "output": result.question_type},
            {"kind": "route", "name": "view", "output": result.table_chosen},
            {"kind": "retrieve", "output": result.columns_found},
            {"kind": "generate", "output": result.sql},
            {"kind": "execute", "rows": result.rows},
            {"kind": "answer", "output": result.answer},
        ],
    )
```

Each stage has a `kind`. Give a `name` when you have two stages of the same kind.

| Kind | Use it for | What to put in it |
|---|---|---|
| `route` | A decision, such as the type of question or which table to use | `output`: the choice |
| `plan` | A plan or a list of ideas taken from the question | `output`: anything |
| `retrieve` | Finding the tables or columns that are needed | `output`: a list of names |
| `generate` | Writing the SQL | `output`: the SQL text |
| `execute` | Running the SQL | `rows`: a list of dicts, or `error`: the error text |
| `answer` | The reply for the user | `output`: the text |

In a case, `expect` then checks a stage by its name:

```yaml
- id: delivered-count
  question: "How many orders are delivered?"
  reference_sql: SELECT COUNT(*) FROM orders WHERE store_id = 's1' AND status = 'DELIVERED'
  expect:
    classify: data_question     # the stage named "classify" must give this value
    view: orders                # the stage named "view" must give this value
    retrieve: [status]          # the retrieve stage must find at least these names
```

## Add an LLM judge (optional)

Some answers are hard to check with simple rules, for example a long answer in sentences.
For these, you can ask an AI model to judge. A judge is a function that takes a prompt
and returns the model's reply as text.

```python
def my_judge(prompt):
    reply = call_my_llm(prompt)  # your own code, with any AI provider
    return reply
```

Use it with `--judge`:

```bash
dbagenteval run cases.yaml --agent shop:my_agent --execute shop:run_sql --judge shop:my_judge
```

Or in Python: `check_question(..., judge=my_judge)`.

The judge adds these checks:

| Check | The judge decides if |
|---|---|
| `reference_answer` | The answer agrees with the result of the reference SQL |
| `faithful` | Everything in the answer is supported by the agent's own rows |
| `relevant` | The answer really answers the question |

Things to know about the judge:

- It must be a normal function, not an `async` function.
- Each judge check calls your AI model one time, so a judge costs money and time.
- The checks without a judge still run. You see both results together.

## Reference

### The fields of a case

| Field | Needed? | Meaning |
|---|---|---|
| `question` | Yes | The question to ask the agent |
| `reference_sql` | No | A correct SQL query. It is run each time, and the agent's result is compared with it |
| `expect` | No | Correct values written by hand. The key `rows` is the correct result rows. Any other key is a stage name |
| `id` | No | A short name for the case. It is shown in the output |
| `tags` | No | Labels to group cases, for example `[easy, money]` |
| `ordered` | No | `true` when the order of the rows matters, for example a "top 5" question. The default is `false` |
| `precision` | No | How many decimal places to compare. The default is `2` |

The file can be `.yaml`, `.yml`, `.json` or `.jsonl` (one case on each line).

### The checks in a scorecard

| Check | It needs | It passes when |
|---|---|---|
| `no_error` | Nothing | The stage finished without an error |
| `expected` | A stage name in `expect` | The stage gave the expected value |
| `recall` | A retrieve stage in `expect` | The stage found every expected name |
| `rows_match` | `expect.rows` | The agent's rows are the expected rows |
| `reference_rows` | `reference_sql` and the agent's rows | The agent's rows are the rows of the reference SQL |
| `reference_numbers` | `reference_sql` and only an answer text | The answer text has the reference value and no other number |
| `reference_answer` | `reference_sql` and a judge | The judge says the answer agrees with the reference result |
| `read_only`, `tenant_filter`, and other rule names | That rule in your rules list | See "The rules you can use" |
| `numbers_grounded` | The `NumbersGrounded()` rule | Every number in the answer is in the agent's rows |
| `faithful`, `relevant` | A judge | See "Add an LLM judge" |

### The verdicts

| Verdict | Meaning |
|---|---|
| `PASS` | At least one check ran, and no check failed |
| `FAIL` | One or more checks failed. The output shows the first stage that failed |
| `ERROR` | The test itself could not run. For example, your agent function raised an error, or the reference SQL is wrong. Fix the test, not the agent |
| `SKIP` | There was nothing to check |

### The commands

| Command | What it does |
|---|---|
| `dbagenteval validate cases.yaml` | Checks the cases file for mistakes. It does not run the agent |
| `dbagenteval run cases.yaml --agent file:function` | Runs every case and prints the results |
| `dbagenteval diff before.json after.json` | Compares two saved runs |

Options for `run`:

| Option | Meaning |
|---|---|
| `--agent file:function` | Your agent function. This option is needed |
| `--execute file:function` | Your executor. It is needed when a case has `reference_sql` |
| `--rules file:NAME` | A list of rules |
| `--judge file:function` | Your judge function |
| `--tags easy,money` | Run only the cases that have one of these tags |
| `--concurrency 1` | How many cases run at the same time. The default is 4 |
| `--json out.json` | Save the full result in a file. `diff` uses this file |
| `--md report.md` | Save a short report that you can share |

Exit codes, which are useful for automatic tests (CI):

| Code | Meaning |
|---|---|
| `0` | Everything passed. For `diff`: no regression |
| `1` | Something did not pass. For `diff`: there is a regression |
| `2` | The command could not run, for example a mistake in the cases file or in an option |

### How rows are compared

The tool compares the agent's rows with the correct rows in a forgiving way:

- **Column names can be different.** `{count: 2}` matches a column named `total`. But if
  both sides have a column with the same name, those two columns are compared.
- **Extra columns are fine.** Write only the columns that matter.
- **Number types do not matter.** `2`, `2.0` and `"2.00"` are equal.
- **Numbers are rounded to 2 decimal places.** Set `precision: 4` in the case for small
  numbers such as rates.
- **The order of rows does not matter**, except when the case has `ordered: true`.
- **An empty value is an empty value.** `null` and an empty text are equal.
- Text that only looks like a number stays text. The postcode `"01234"` is not equal to
  `1234`.

With `expect.rows`, you can give several correct results. The case passes if one of them
matches:

```yaml
expect:
  rows:
    - [{category: Electronics, total: 2000}]
    - [{category: Electronics}]
```

`rows: []` means "the correct result has no rows".

### Use it from Python

Everything the command line does is also available in Python.

```python
from dbagenteval import Case, evaluate, load_cases, run_suite

# Check one trace that you already have
case = Case(question="How many orders are delivered?", expect={"rows": [{"total": 2}]})
card = evaluate(trace, case, rules=RULES)
print(card.verdict)  # "PASS" or "FAIL"
print(card.failed_stage)  # the first stage that failed, or None
print(card.summary())

# Run a whole cases file
result = run_suite(load_cases("cases.yaml"), my_agent, rules=RULES, execute=run_sql)
print(result.summary())  # numbers: total, passed, failed, and more
```

In a notebook or in `async` code, use `await arun_suite(...)` in place of `run_suite(...)`.

### Write your own rule

```python
from dbagenteval import Check
from dbagenteval.scorecard import FAIL, PASS, SKIP


class ShortAnswer(Check):
    name = "short_answer"

    def run(self, trace, case):
        stage = trace.last_of("answer")
        if stage is None:
            return self.result("answer", SKIP)
        length = len(stage.output)
        return self.result(stage.name, PASS if length <= 400 else FAIL, f"{length} characters")
```

Add `ShortAnswer()` to your `RULES` list. A rule returns `PASS`, `FAIL`, or `SKIP` when it
has nothing to check.

## Common problems

| What you see | What to do |
|---|---|
| `cannot import 'shop'` | Run the command from the folder that has `shop.py`. Check the spelling of the file name |
| `some cases have reference_sql; pass --execute` | Add `--execute file:function` to the command |
| `rows must be a list of dicts` | Your rows are tuples. Change them with `dict(zip(columns, row))`, as in the executor example |
| `stage missing from trace` | A key in `expect` is not the name of a stage in your trace. Check the spelling |
| `ERROR` and `the reference SQL failed to run` | Your reference SQL has a mistake. Run it yourself on the database and fix it |
| `ERROR` and `agent raised` | Your agent function raised an error. The message shows the error |
| `tenant_filter` fails on a query with a JOIN | Write the table name in the filter, like `o.store_id = 's1'`, or use `tables=[...]` |
| `run_suite() cannot be called while an event loop is running` | You are in a notebook. Use `await arun_suite(...)` |
| A rule fails SQL that looks correct to you | Give your database type, for example `ReadOnly(dialect="postgres")` |

## Limits you should know

- **The SQL rules are a testing tool, not a security system.** They read the SQL with
  their own parser, and your database may read unusual SQL in a different way. Keep the
  real protection in your database: a read-only user, and row-level security.
- **The rules are strict.** Sometimes a rule fails a query that is safe. For example,
  `TenantFilter` wants the filter inside a subquery too, not only in the outer query.
- **`ReadOnly` has a list of forbidden functions**, such as `pg_sleep`. The list cannot
  have every dangerous function. To add your own, give the full list:
  `ReadOnly(forbidden_functions=[*DEFAULT_FORBIDDEN_FUNCTIONS, "my_function"])`.
  `DEFAULT_FORBIDDEN_FUNCTIONS` is in `dbagenteval.rules`.
- **`NumbersGrounded` is a simple check.** It can fail a correct answer when the agent
  did its own maths, for example a total that it added by itself. It also does not look
  at the plus or minus sign. A judge is better for such answers.
- **`reference_numbers` is a simple check too.** It works well for one value, such as a
  count. For a result with many rows, it cannot know if the answer covers all of them.
  Put the agent's rows in the trace, or use a judge.
- **Not in this version:** tests with follow-up questions, more than one reference SQL
  for a case, and reading cases from a CSV file.

## Examples in this repository

If you download the source code, the `examples` folder has two ready examples. Run them
from the main folder of the repository.

```bash
dbagenteval run examples/cases.yaml --agent examples.toy_agent:agent \
    --rules examples.toy_agent:RULES --execute examples.toy_agent:run_sql
```

This runs five cases on a small online-store agent that records every step. One case
fails on purpose, to show a correct number that comes from an unsafe query.

```bash
python -m examples.reference_check
```

This checks four small agents against one reference SQL and shows each result.

## For developers

```bash
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
mypy
```

To publish a new version: change `__version__` in `src/dbagenteval/__init__.py`, then run

```bash
python -m build
twine check dist/*
twine upload dist/*
```

## License

MIT
