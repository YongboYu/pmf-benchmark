import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler


class DataLoader:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        # Setup logging
        self.logger = logging.getLogger(__name__)

    def load_data(self, dataset: str) -> TimeSeries:
        """Load data from file"""
        data_path = (
            Path(self.config["paths"]["data_dir"]) / "processed" / "time_series_df.h5"
        )
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        data = pd.read_hdf(data_path, key=dataset)
        return TimeSeries.from_dataframe(data)

    def split_data(
        self, series: TimeSeries, model_group: str
    ) -> tuple[TimeSeries, TimeSeries | None, TimeSeries]:
        """Split data into train, validation, and test sets"""
        if model_group in ["baseline", "statistical", "foundation"]:
            train_size = self.config["data"]["simple_split"]
            train, test = series.split_after(train_size)
            return train, None, test

        # Add covariate_regression and DL models to groups that use validation set
        if model_group in [
            "regression",
            "deep_learning",
            "univariate_regression",
            "covariate_regression",
            "univariate_dl",
            "covariate_dl",
        ]:
            test_size = self.config["data"]["test_split"]
            val_size = self.config["data"]["val_split"]

            train_val, test = series.split_after(1 - test_size)
            train, val = train_val.split_after(1 - val_size / (1 - test_size))
            return train, val, test

        raise ValueError(f"Unknown model group: {model_group}")

    def transform_data(
        self,
        train: TimeSeries,
        val: TimeSeries | None,
        test: TimeSeries,
        model_group: str,
    ) -> tuple[TimeSeries, TimeSeries | None, TimeSeries, Any | None]:
        """Transform data based on model group"""
        try:
            # Initialize transformer based on model group
            transformer = None
            transform_configs = self.config.get("transformations", {})

            # For baseline, regression, univariate_regression, covariate_regression, and foundation models
            if model_group in [
                "baseline",
                "regression",
                "foundation",
                "univariate_regression",
                "covariate_regression",
            ]:
                return train, val, test, None

            # For deep learning models (including both univariate_dl and covariate_dl)
            elif model_group in ["deep_learning", "univariate_dl", "covariate_dl"]:
                transformer = Scaler()
                train_t = transformer.fit_transform(train)
                val_t = transformer.transform(val) if val is not None else None
                test_t = transformer.transform(test)
                return train_t, val_t, test_t, transformer

            elif model_group == "statistical":
                transform_config = transform_configs[model_group]
                offset = transform_config["offset"]
                epsilon = transform_config["epsilon"]

                # Step 1: Add offset to raw data
                train_offset = train + offset
                test_offset = test + offset

                # Step 2: Apply log transformation
                train_t = train_offset.map(lambda x: np.log(x))
                test_t = test_offset.map(lambda x: np.log(x))

                # Step 3: Add epsilon in log space
                train_t = train_t + epsilon
                test_t = test_t + epsilon

                return train_t, None, test_t, None

            else:
                raise ValueError(f"Unknown model group: {model_group}")

        except Exception as e:
            self.logger.error(f"Error in {model_group} transformation: {e!s}")
            raise

    def prepare_data(
        self, dataset: str, model_group: str
    ) -> tuple[TimeSeries, TimeSeries | None, TimeSeries, Any, dict[str, Any] | None]:
        """Main function to prepare data for training

        Args:
            dataset: Name of the dataset
            model_group: Type of model to prepare data for

        Returns:
            Tuple containing:
            - train: Training data
            - val: Validation data (or None)
            - test: Test data
            - transformer: The fitted transformer (or None)
            - transform_params: Dictionary of transformation parameters (or None)
        """
        try:
            # Load data
            series = self.load_data(dataset)

            # Split data
            train, val, test = self.split_data(series, model_group)

            # Transform data
            return self.transform_data(train, val, test, model_group)

        except Exception as e:
            raise Exception(f"Error preparing data for {dataset}: {e!s}") from e

    def inverse_transform(
        self, series: TimeSeries, transformer: Any, model_group: str
    ) -> TimeSeries:
        """Inverse transform data"""
        if model_group in [
            "baseline",
            "regression",
            "foundation",
            "univariate_regression",
            "covariate_regression",
        ]:
            return series

        try:
            if model_group == "statistical":
                # Get transformation parameters
                transform_params = self.config["transformations"][model_group]
                epsilon = transform_params.get("epsilon")
                offset = transform_params.get("offset")

                # transform_type = transform_params.get('type')

                # Remove epsilon
                series_no_epsilon = series - epsilon
                series_no_transform = series_no_epsilon.map(lambda x: np.exp(x))

                # Apply inverse transform
                # if transform_type == 'log':
                #     series_no_transform = series_no_epsilon.map(lambda x: np.exp(x))
                # else:  # box-cox
                #     series_no_transform = transformer.inverse_transform(series_no_epsilon)

                # Remove offset
                return series_no_transform - offset

            else:  # deep_learning, univariate_dl, covariate_dl
                return transformer.inverse_transform(series)

        except Exception as e:
            self.logger.error(f"Error in inverse transformation: {e!s}")
            raise

    def create_seq2seq_io_data(
        self,
        train: TimeSeries,
        val: TimeSeries | None,
        test: TimeSeries | None,
        input_length: int,
        output_length: int,
        dataset_type: str,
    ) -> tuple[list[TimeSeries], list[TimeSeries]]:
        """Create sequence-to-sequence dataset from TimeSeries.

        Args:
            train: Training TimeSeries
            val: Validation TimeSeries (optional)
            test: Test TimeSeries
            input_length: Length of input sequences (lags or input_chunk_length)
            output_length: Length of output sequences (horizon)
            dataset_type: Type of dataset to create ('val' or 'test')

        Returns:
            Tuple containing lists of input and output TimeSeries sequences
        """
        input_series_list = []
        output_series_list = []

        if dataset_type == "val":
            if val is None:
                raise ValueError(
                    "Validation set is required when dataset_type is 'val'"
                )

            # Concatenate train and val for input sequences
            combined_series = train.concatenate(val)

            # Calculate number of sequences (length of validation set minus horizon plus 1)
            n_sequences = len(val) - output_length + 1

            # Starting index for the first input sequence
            first_input_start = len(train) - input_length

            # Create sequences
            for i in range(n_sequences):
                # Extract input sequence
                input_start = first_input_start + i
                input_end = input_start + input_length
                input_series = combined_series[input_start:input_end]

                # Extract output sequence (only from validation set)
                output_start = len(train) + i
                output_end = output_start + output_length
                output_series = combined_series[output_start:output_end]

                input_series_list.append(input_series)
                output_series_list.append(output_series)

        elif dataset_type == "test":
            # Concatenate train, val (if exists), and test
            if val is not None:
                combined_series = train.concatenate(val).concatenate(test)
                history_length = len(train) + len(val)
            else:
                combined_series = train.concatenate(test)
                history_length = len(train)

            # Calculate number of sequences (length of test set minus horizon plus 1)
            n_sequences = len(test) - output_length + 1

            # Starting index for the first input sequence
            first_input_start = history_length - input_length

            # Create sequences
            for i in range(n_sequences):
                # Extract input sequence
                input_start = first_input_start + i
                input_end = input_start + input_length
                input_series = combined_series[input_start:input_end]

                # Extract output sequence (only from test set)
                output_start = history_length + i
                output_end = output_start + output_length
                output_series = combined_series[output_start:output_end]

                input_series_list.append(input_series)
                output_series_list.append(output_series)

        else:
            raise ValueError("dataset_type must be either 'val' or 'test'")

        return input_series_list, output_series_list

    def create_expanding_io_data(
        self,
        train: TimeSeries,
        val: TimeSeries | None,
        test: TimeSeries,
        horizon: int,
    ) -> tuple[list[TimeSeries], list[TimeSeries]]:
        """Create expanding window dataset from TimeSeries.

        Args:
            train: Training TimeSeries
            val: Validation TimeSeries (optional)
            test: Test TimeSeries
            horizon: Length of output sequences (prediction horizon)

        Returns:
            Tuple containing lists of input and output TimeSeries sequences
        """
        input_series_list = []
        output_series_list = []

        # Combine training and validation data if validation exists
        history = train.concatenate(val) if val is not None else train

        # Calculate number of sequences (length of test set minus horizon plus 1)
        n_sequences = len(test) - horizon + 1

        # Create sequences with expanding window
        for i in range(n_sequences):
            # Combine historical data with available test data for input
            available_test = test[:i] if i > 0 else None
            if available_test is not None:
                input_series = history.concatenate(available_test)
            else:
                input_series = history

            # Extract output sequence from test set
            output_start = i
            output_end = output_start + horizon
            output_series = test[output_start:output_end]

            input_series_list.append(input_series)
            output_series_list.append(output_series)

        return input_series_list, output_series_list
