Creating a robust data integrity checker involves multiple steps, such as loading data, performing consistency checks, and applying anomaly detection techniques. Below is a simplified version of such a tool using Python, demonstrating basic methods for data validation and anomaly detection using Pandas and Scikit-learn. This example assumes datasets are in CSV format:

```python
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

class DataIntegrityChecker:
    def __init__(self, file_path):
        """
        Initialize the data integrity checker with the path to the dataset.

        :param file_path: Path to the CSV file containing the dataset.
        """
        self.file_path = file_path
        self.data = None
        self.scaler = StandardScaler()

    def load_data(self):
        """
        Load dataset from CSV file.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            print("Data loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise
        except pd.errors.EmptyDataError:
            print("Error: No data found at the specified path.")
            raise
        except pd.errors.ParserError as e:
            print("Error parsing data: ", e)
            raise

    def check_missing_values(self):
        """
        Check for missing values in the dataset.
        """
        try:
            if self.data.isnull().sum().sum() > 0:
                print("Warning: Missing values detected.")
                print(self.data.isnull().sum())
            else:
                print("No missing values detected.")
        except Exception as e:
            print(f"Error checking missing values: {e}")

    def check_duplicates(self):
        """
        Check for duplicate records in the dataset.
        """
        try:
            if self.data.duplicated().any():
                print("Warning: Duplicate records detected.")
                print(f"Number of duplicates: {self.data.duplicated().sum()}")
            else:
                print("No duplicate records detected.")
        except Exception as e:
            print(f"Error checking duplicates: {e}")

    def check_outliers(self):
        """
        Detect outliers using the Isolation Forest method.
        """
        try:
            # Scale the data
            numeric_data = self.data.select_dtypes(include=[np.number])
            scaled_data = self.scaler.fit_transform(numeric_data)

            # Apply Isolation Forest
            model = IsolationForest(contamination=0.01, random_state=42)
            model.fit(scaled_data)
            outliers = model.predict(scaled_data)

            # Count number of outliers
            num_outliers = np.sum(outliers == -1)
            if num_outliers > 0:
                print(f"Outliers detected: {num_outliers}")
            else:
                print("No outliers detected.")
        except Exception as e:
            print(f"Error during outlier detection: {e}")

    def run_checks(self):
        """
        Run all data integrity checks.
        """
        self.load_data()
        self.check_missing_values()
        self.check_duplicates()
        self.check_outliers()

if __name__ == "__main__":
    file_path = "path_to_your_dataset.csv"  # Update this path with the actual dataset file path
    checker = DataIntegrityChecker(file_path)
    
    try:
        checker.run_checks()
    except Exception as e:
        print(f"An error occurred while running data integrity checks: {e}")
```

### Key Points:
- **Loading Data:** The `load_data` method reads data from a CSV file and handles errors like file not found and parsing issues.
- **Missing Values:** The `check_missing_values` method checks for and reports missing values.
- **Duplicate Records:** The `check_duplicates` method identifies duplicate records.
- **Outliers:** The `check_outliers` method uses the `IsolationForest` algorithm to detect outliers in numeric data.
- **Error Handling:** Each method includes basic error handling to catch and report issues that occur during execution.

To run this script, make sure you have the required libraries installed, and update the `file_path` variable with the correct path to your dataset.