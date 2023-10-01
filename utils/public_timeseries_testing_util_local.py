from typing import Tuple

import pandas as pd


class MockApi:
    def __init__(
        self,
        train: pd.DataFrame,
        start_date: int = 401,
        end_date: int = 480,
        submission_path: str = "submission.csv",
    ) -> None:
        """Creates a mock API that mimics the real API.

        Args:
            train (pd.DataFrame): The training data.
            start_date (int, optional): The first `date_id` to use for the test. Defaults to 401.
            end_date (int, optional): The last `date_id` to use for the test. Defaults to 480.
            submission_path (str, optional): The path to write the submission file to. Defaults to "submission.csv".

        Warning: NO permission to write `submission.csv` in kaggle notebooks.
        """
        print("Creating mock API...", flush=True)

        self.group_id_column: str = "time_id"
        self.export_group_id_column: bool = False
        self.submission_path: str = submission_path

        test = train[(train["date_id"] >= start_date) & (train["date_id"] <= end_date)].copy()
        test.drop(columns=["target"], inplace=True)

        revealed = test[["stock_id", "date_id", "seconds_in_bucket", "time_id"]].copy()
        revealed["revealed_date_id"] = revealed["date_id"] - 1
        revealed["revealed_time_id"] = revealed["time_id"] - 55
        revealed = pd.merge(
            revealed,
            train[["stock_id", "date_id", "time_id", "target"]],
            how="left",
            left_on=["stock_id", "revealed_date_id", "revealed_time_id"],
            right_on=["stock_id", "date_id", "time_id"],
        )
        revealed.drop(columns=["date_id_y", "time_id_y"], inplace=True)
        revealed.rename(columns={"date_id_x": "date_id", "time_id_x": "time_id"}, inplace=True)
        revealed.rename(columns={"target": "revealed_target"}, inplace=True)
        revealed = revealed[
            [
                "stock_id",
                "date_id",
                "seconds_in_bucket",
                "time_id",
                "revealed_target",
                "revealed_date_id",
                "revealed_time_id",
            ]
        ]

        submission = test[["time_id", "row_id"]].copy()
        submission["target"] = 0.0

        self.frames = [test, revealed, submission]

        self._status = "initialized"
        self.predictions = []

        print("Mock API created.", flush=True)
        print(f"Test date range: {start_date} - {end_date} ({end_date - start_date + 1} days)", flush=True)

    def iter_test(self) -> Tuple[pd.DataFrame]:
        """
        Loads all of the dataframes specified in self.input_paths,
        then yields all rows in those dataframes that equal the current self.group_id_column value.
        """
        if self._status != "initialized":
            raise Exception("WARNING: the real API can only iterate over `iter_test()` once.")

        dataframes = self.frames
        group_order = dataframes[0][self.group_id_column].drop_duplicates().tolist()
        dataframes = [df.set_index(self.group_id_column) for df in dataframes]

        for group_id in group_order:
            self._status = "prediction_needed"
            current_data = []
            for df in dataframes:
                cur_df = df.loc[group_id].copy()
                # returning single line dataframes from df.loc requires special handling
                if not isinstance(cur_df, pd.DataFrame):
                    cur_df = pd.DataFrame({a: b for a, b in zip(cur_df.index.values, cur_df.values)}, index=[group_id])
                    cur_df.index.name = self.group_id_column
                cur_df = cur_df.reset_index(drop=not (self.export_group_id_column))
                current_data.append(cur_df)
            yield tuple(current_data)

            while self._status != "prediction_received":
                print("You must call `predict()` successfully before you can continue with `iter_test()`", flush=True)
                yield None

        with open(self.submission_path, "w") as f_open:
            pd.concat(self.predictions).to_csv(f_open, index=False)
        self._status = "finished"

    def predict(self, user_predictions: pd.DataFrame) -> None:
        """
        Accepts and stores the user's predictions and unlocks iter_test once that is done
        """
        if self._status == "finished":
            raise Exception("You have already made predictions for the full test set.")
        if self._status != "prediction_needed":
            raise Exception("You must get the next test sample from `iter_test()` first.")
        if not isinstance(user_predictions, pd.DataFrame):
            raise Exception("You must provide a DataFrame.")

        self.predictions.append(user_predictions)
        self._status = "prediction_received"