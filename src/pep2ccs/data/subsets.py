"""Row filters used by the published experiments."""

import pandas as pd


def select_subset(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "full":
        return frame.reset_index(drop=True)
    if subset == "multi_charge":
        counts = frame.groupby("Sequence")["Charge"].nunique()
        keep = counts[counts > 1].index
        return frame[frame["Sequence"].isin(keep)].reset_index(drop=True)
    raise ValueError("subset must be 'full' or 'multi_charge', got {0}".format(subset))
