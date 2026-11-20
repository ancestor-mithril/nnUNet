from typing import List

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def generate_crossval_split(
        train_identifiers: List[str],
        stratification_labels: dict[str, str] | None = None,
        seed: int = 12345,
        n_splits: int = 5,
) -> List[dict[str, List[str]]]:
    identifiers = np.asarray(train_identifiers)

    if stratification_labels is None:
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iterator = kfold.split(identifiers)
    else:
        missing = [identifier for identifier in identifiers if identifier not in stratification_labels]
        if missing:
            raise ValueError(
                f"Missing stratification labels for "
                f"{len(missing)} identifiers: {missing}"
            )

        labels = np.asarray([stratification_labels[identifier] for identifier in identifiers])

        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iterator = kfold.split(identifiers, labels)

    splits = []

    for train_idx, val_idx in split_iterator:
        splits.append(
            {
                "train": identifiers[train_idx].tolist(),
                "val": identifiers[val_idx].tolist(),
            }
        )

    return splits
