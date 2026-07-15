from typing import List


def print_table(title: str, rows: List[List[str]]) -> None:
    if not rows:
        return

    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    title_line = f"| {title.center(len(border) - 4)} |"

    print(border)
    print(title_line)
    print(border)
    for idx, row in enumerate(rows):
        print(
            "| "
            + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row))
            + " |"
        )
        if idx == 0:
            print(border)
    print(border)
