import pandas as pd
import numpy as np
import os


def round_to_and_save(path, to):
    df = pd.read_csv(PATH)
    df["sales"] = np.round(df["sales"], to)
    df.to_csv(f"Lab_3/submission_darts_rounded_{to}.csv", index=False)


PATH = "Lab_3/submission_darts.csv"

# round to 1
round_to_and_save(PATH, 1)
