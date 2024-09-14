import fastparquet as fp
import pandas as pd
import os


class Models():
    def __init__(self):
        # Store the data
        self.data = {}
        self.average_data = {}

        # Limit number of folders for now for speed + testing
        self.x = 5

        self.read_files()
        self.make_average_data()

    def read_files(self):
        """
        Converts parquet data files into pandas DataFrames then processes them
        """
        # Walk through the keywords files
        for root, _, files in os.walk("keywords\\"):
            # Set up data collection for each keyword
            keyword = root.split("\\")[-1]
            if keyword == "":
                continue

            self.data[keyword] = []

            print(f"Currently processing keyword \"{keyword}\"...")
            # Go through every file
            for file_name in files:
                file_path = os.path.join(root, file_name)

                # Convert file into a Dataframe
                df = fp.ParquetFile(file_path).to_pandas()
                
                # Remove the face and pose data
                df = df[~df["type"].isin(["face", "pose"])]

                # Remove all rows that have NaN
                df = df.dropna()

                # Add to keyword data
                self.data[keyword].append(df)

            self.x -= 1
            
            if self.x == 0:
                break

    def make_average_data(self):
        """
        Aggregates the DataFrames for each keyword into a single DataFrame whose values are the
        averages of the values within the original DataFrames
        """
        for keyword in self.data:
            concat_df = pd.concat(self.data[keyword])
            numeric_columns = concat_df.select_dtypes(include=['int16', 'float64']).columns

            self.average_df = concat_df.groupby(["row_id"])[numeric_columns].mean().reset_index()

            print(f"Finished making average dataframe for \"{keyword}\"")

    def get_average_data(self, keyword):
        """
        Attempts to return the average data DataFrame based on keyword
        """
        if keyword not in self.average_data:
            return None
        
        return self.average_data[keyword]
