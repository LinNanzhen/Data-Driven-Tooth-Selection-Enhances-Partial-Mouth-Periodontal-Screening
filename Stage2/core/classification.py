from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from core.data_processing import DataProcessor


class CDCClassifier:
    def __init__(self, data_processor: DataProcessor):
        self.data_processor = data_processor

    def classify_with_tooth_selection(
        self,
        df: pd.DataFrame,
        selected_teeth_nhanes: List[str],
    ) -> np.ndarray:
        return self.data_processor.classify_with_tooth_selection(df, selected_teeth_nhanes)
