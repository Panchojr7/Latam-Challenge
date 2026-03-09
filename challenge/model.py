import numpy as np
import pandas as pd
from datetime import datetime
from enum import Enum
from typing import List, Tuple, Union
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


DATE_MONTH_FORMAT = "%d-%b"
TIME_FORMAT = "%H:%M"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
ORIGINAL_DATE = "Fecha-O"
SCHEDULED_DATE = "Fecha-I"
DAY_PERIOD = "period_day"
PEAK_SEASON = "high_season"
MINUTES_DIFF = "min_diff"
IS_DELAYED = "delay"
AIRLINE = "OPERA"
FLIGHT_TYPE = "TIPOVUELO"
MONTH_NUM = "MES"

class DayPeriod(Enum):
    MORNING = "mañana"
    AFTERNOON = "tarde"
    NIGHT = "noche"


class DelayModel:

    def __init__(self):
        self._model = None # Model should be saved in this attribute.
        self.feature_subset: List[str] = [
            "OPERA_Latin American Wings", "MES_7", "MES_10",
            "OPERA_Grupo LATAM", "MES_12", "TIPOVUELO_I",
            "MES_4", "MES_11", "OPERA_Sky Airline", "OPERA_Copa Air",
        ]
        self.required_input_cols: List[str] = [
            "Fecha-I", "Vlo-I", "Ori-I", "Des-I", "Emp-I",
            "Fecha-O", "Vlo-O", "Ori-O", "Des-O", "Emp-O",
            "DIA", "MES", "AÑO", "DIANOM", "TIPOVUELO",
            "OPERA", "SIGLAORI", "SIGLADES",
        ]
        self.calculated_cols: List[str] = [
            "period_day", "high_season", "min_diff", "delay",
        ]

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        if self._validate_columns(data) and self._is_valid_target(target_column):
            enriched_df = self._add_engineered_features(data)
            features = self._encode_categorical_features(enriched_df)
            return features[self.feature_subset], data[[target_column]]
        else:
            inference_features = self._encode_categorical_features(data)
            aligned_features = self._align_to_feature_subset(inference_features)
            return aligned_features[self.feature_subset]

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        x_train, _, y_train, _ = train_test_split(
            features, target, test_size=0.33, random_state=42
        )
        count_y0 = len(y_train[y_train == 0])
        count_y1 = len(y_train[y_train == 1])
        total_samples = len(y_train)
        classifier = LogisticRegression(
            class_weight={1: count_y0 / total_samples, 0: count_y1 / total_samples}
        )
        classifier.fit(x_train, y_train)
        self._model = classifier

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        predictions = self._model.predict(features)
        return predictions.tolist()

    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        delay_threshold = 15
        df[DAY_PERIOD] = df[SCHEDULED_DATE].apply(self._calculate_day_period)
        df[PEAK_SEASON] = df[SCHEDULED_DATE].apply(self._check_peak_season)
        df[MINUTES_DIFF] = df.apply(self._get_time_difference, axis=1)
        df[IS_DELAYED] = np.where(df[MINUTES_DIFF] > delay_threshold, 1, 0)
        return df.copy(True)

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        encoded_df = pd.concat(
            [
                pd.get_dummies(df[AIRLINE], prefix=AIRLINE),
                pd.get_dummies(df[FLIGHT_TYPE], prefix=FLIGHT_TYPE),
                pd.get_dummies(df[MONTH_NUM], prefix=MONTH_NUM),
            ],
            axis=1,
        )
        return encoded_df.copy(True)

    def _align_to_feature_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        template_df = pd.DataFrame(
            0, index=np.arange(df.shape[0]), columns=self.feature_subset
        )
        for col in df.columns:
            if col in template_df.columns:
                template_df[col] = template_df[col] | df[col]
        return template_df

    def _validate_columns(self, df: pd.DataFrame) -> bool:
        current_cols = list(df.columns)
        return all(col in current_cols for col in self.required_input_cols)

    def _is_valid_target(self, target_name: str) -> bool:
        all_valid_cols = self.required_input_cols + self.calculated_cols
        return target_name in all_valid_cols

    @staticmethod
    def _check_peak_season(date_str: str) -> int:
        year = int(date_str.split("-")[0])
        dt = datetime.strptime(date_str, DATETIME_FORMAT)

        # Season ranges
        r1_min = datetime.strptime("15-Dec", DATE_MONTH_FORMAT).replace(year=year)
        r1_max = datetime.strptime("31-Dec", DATE_MONTH_FORMAT).replace(year=year)
        r2_min = datetime.strptime("1-Jan", DATE_MONTH_FORMAT).replace(year=year)
        r2_max = datetime.strptime("3-Mar", DATE_MONTH_FORMAT).replace(year=year)
        r3_min = datetime.strptime("15-Jul", DATE_MONTH_FORMAT).replace(year=year)
        r3_max = datetime.strptime("31-Jul", DATE_MONTH_FORMAT).replace(year=year)
        r4_min = datetime.strptime("11-Sep", DATE_MONTH_FORMAT).replace(year=year)
        r4_max = datetime.strptime("30-Sep", DATE_MONTH_FORMAT).replace(year=year)

        is_peak = (
            (dt >= r1_min and dt <= r1_max) or (dt >= r2_min and dt <= r2_max) or
            (dt >= r3_min and dt <= r3_max) or (dt >= r4_min and dt <= r4_max)
        )
        return 1 if is_peak else 0

    @staticmethod
    def _calculate_day_period(date_str: str) -> str:
        current_time = datetime.strptime(date_str, DATETIME_FORMAT).time()

        morn_min = datetime.strptime("05:00", TIME_FORMAT).time()
        morn_max = datetime.strptime("11:59", TIME_FORMAT).time()
        aft_min = datetime.strptime("12:00", TIME_FORMAT).time()
        aft_max = datetime.strptime("18:59", TIME_FORMAT).time()
        eve_min = datetime.strptime("19:00", TIME_FORMAT).time()
        eve_max = datetime.strptime("23:59", TIME_FORMAT).time()
        night_min = datetime.strptime("00:00", TIME_FORMAT).time()
        night_max = datetime.strptime("4:59", TIME_FORMAT).time()

        if morn_min < current_time < morn_max:
            return DayPeriod.MORNING.value
        elif aft_min < current_time < aft_max:
            return DayPeriod.AFTERNOON.value
        elif (eve_min < current_time < eve_max) or (night_min < current_time < night_max):
            return DayPeriod.NIGHT.value

    @staticmethod
    def _get_time_difference(row: pd.Series) -> int:
        dt_actual = datetime.strptime(row[ORIGINAL_DATE], DATETIME_FORMAT)
        dt_scheduled = datetime.strptime(row[SCHEDULED_DATE], DATETIME_FORMAT)
        return ((dt_actual - dt_scheduled).total_seconds()) / 60
