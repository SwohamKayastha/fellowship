import re

from executor import run_pipeline

BENCHMARK = [
    {
        "question": "List all customers",
        "expected_sql": "SELECT \"customerName\" FROM customers;",
    },
    {
        "question": "List all products",
        "expected_sql": "SELECT \"productName\" FROM products;",
    },
    {
        "question": "Count customers per country",
        "expected_sql": "SELECT \"country\", COUNT(*) FROM customers GROUP BY \"country\";",
    },
    {
        "question": "List all orders",
        "expected_sql": "SELECT \"orderNumber\" FROM orders;",
    },
]


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", (sql or "").strip().strip(";")).lower()


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    header_line = " | ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers))
    divider = "-+-".join("-" * widths[idx] for idx in range(len(headers)))
    lines = [header_line, divider]
    for row in rows:
        line = " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers)))
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    rows = []
    total = len(BENCHMARK)
    success = 0
    retry_total = 0
    retry_success = 0

    for item in BENCHMARK:
        question = item["question"]
        expected_sql = item["expected_sql"]
        result = run_pipeline(question)

        generated_sql = result.get("sql", "")
        executed = result.get("status") == "success"
        if executed:
            success += 1

        retry_needed = result.get("retry_count", 0) > 0
        if retry_needed:
            retry_total += 1
            if executed:
                retry_success += 1

        correct = _normalize_sql(generated_sql) == _normalize_sql(expected_sql)

        rows.append(
            [
                question,
                generated_sql,
                "Yes" if executed else "No",
                "Yes" if correct else "No",
                "Yes" if retry_needed else "No",
                result.get("status", ""),
            ]
        )

    table = _format_table(
        [
            "Question",
            "Generated SQL",
            "Executed Successfully",
            "Correct Result",
            "Retry Needed",
            "Final Status",
        ],
        rows,
    )

    print(table)
    print()
    print(f"SQL execution success rate: {success}/{total} ({success/total:.2%})")
    if retry_total:
        print(
            f"Retry success rate: {retry_success}/{retry_total} "
            f"({retry_success/retry_total:.2%})"
        )
    else:
        print("Retry success rate: N/A")
    print(f"Total failed queries: {total - success}")


if __name__ == "__main__":
    main()
