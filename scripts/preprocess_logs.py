import sys
from pathlib import Path

# Add the parent directory of the preprocessing module to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ###########

import argparse
import logging
from pathlib import Path

import yaml
from preprocessing.df_generator import DFGenerator
from preprocessing.event_log_processor import EventLogProcessor
from preprocessing.time_series_creator import TimeSeriesCreator
from utils.logging_manager import get_logging_manager


def setup_logging(config: dict) -> logging.Logger:
    """Setup logging configuration using LoggingManager"""
    logging_manager = get_logging_manager(config)
    return logging_manager.get_preprocessing_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Preprocess event logs to time series")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        nargs="+",
        help="Names of the event log files (without extension)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/preprocessing_config.yaml",
        help="Path to preprocessing configuration",
    )

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup logging
    logger = setup_logging(config)

    # Initialize components
    log_processor = EventLogProcessor(config)
    df_generator = DFGenerator(config)
    ts_creator = TimeSeriesCreator(config)

    for dataset in args.dataset:
        try:
            logger.info(f"Processing dataset: {dataset}")

            # Process event log
            processed_log = log_processor.process_log(dataset)

            # Generate DF relations
            df_dict = df_generator.generate_df_relations(processed_log, dataset)

            # Get log timespan
            start_time, end_time = log_processor.get_log_timespan(processed_log)

            # Create time series
            ts_creator.create_time_series(
                df_dict=df_dict,
                start_time=start_time,
                end_time=end_time,
                dataset=dataset,
            )

            logger.info(f"Successfully processed {dataset}")

        except Exception as e:
            logger.error(f"Error processing {dataset}: {e}")
            logger.error("Continuing with next dataset...")
            continue


if __name__ == "__main__":
    main()
