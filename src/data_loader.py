import pandas as pd
import logging
import os
import kagglehub

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, dataset_handle: str):
        self.dataset_handle = dataset_handle

    def fetch_training_data(self) -> pd.DataFrame:
        """
        Simulates fetching batch data from a Data Warehouse by pulling the latest dataset.
        """
        logging.info(f"Connecting to data source via handle: {self.dataset_handle}...")
        
        try:
            # Download the dataset automatically using kagglehub
            path = kagglehub.dataset_download(self.dataset_handle)
            csv_file = os.path.join(path, "card_transdata.csv")
            
            if not os.path.exists(csv_file):
                logging.error("Target CSV not found in the downloaded dataset.")
                raise FileNotFoundError(f"Missing file: {csv_file}")

            df = pd.read_csv(csv_file)
            logging.info(f"Successfully loaded {df.shape[0]} events with {df.shape[1]} features.")
            
            return df
            
        except Exception as e:
            logging.error(f"Pipeline failed during data ingestion: {e}")
            raise e
        
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Ensures the incoming data hasn't suddenly dropped crucial columns.
        """
        required_columns = ['distance_from_home', 'ratio_to_median_purchase_price', 'fraud']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            logging.warning(f"Schema Validation Failed. Missing columns: {missing}")
            return False
        logging.info("Schema validation passed.")
        return True