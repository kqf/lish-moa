import re
import pandas as pd


def main():
    with open("results.md") as f:
        data = f.read()

    fl = r"\d+\.\d+|nan"
    pattern = (
        r"([a-zA-Z0-9_\(\) \t]+)\n"
        rf"CV losses train ({fl}) \+\/\- ({fl})\w*\n"
        rf"CV losses valid ({fl}) \+\/\- ({fl})\w*\n"
        rf"LB: ({fl})"
    )
    matches = re.findall(pattern, data)
    df = pd.DataFrame(matches, columns=[
        "title",
        "train",
        "d train",
        "valid",
        "d valid",
        "lb",
    ])
    print(df)


if __name__ == '__main__':
    main()
