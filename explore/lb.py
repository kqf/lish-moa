import re
import pandas as pd
import matplotlib.pyplot as plt


def parse(data):
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
    for col in df.columns:
        if col != "title":
            df[col] = df[col].astype(float)
    return df


def main():
    with open("results.md") as f:
        data = f.read()

    df = parse(data)
    print(df)

    plt.errorbar(df["lb"], df["valid"], df["d valid"], fmt=".")
    plt.xlabel("Leaderboard Score")
    plt.ylabel("CV score")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
