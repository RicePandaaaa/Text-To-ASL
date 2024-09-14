import fastparquet as fp
import pandas as pd
import os


class Models():
    def __init__(self):
        # Store the data
        self.data = {}
        self.average_data = {}

        # Limit number of folders for now for speed + testing
        self.x = 1

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

                # Change NaNs to 0s
                df = df.fillna(0)

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
            # Combine the dataframes together
            concat_df = pd.concat(self.data[keyword])

            # Extract and average the coordinates
            float_columns = concat_df.select_dtypes(include=["float64"]).columns
            float_columns_means = concat_df.groupby("row_id")[float_columns].mean()

            # Remake the DataFrame
            non_float_columns = concat_df.drop(columns=float_columns).drop_duplicates(subset="row_id")
            concat_df = pd.merge(non_float_columns, float_columns_means.reset_index(), on="row_id")

            # Save the DataFrame
            self.average_data[keyword] = self.trim_dataframe(concat_df)

            print(f"Finished making average dataframe for \"{keyword}\"")

    def trim_dataframe(self, df):

        # Get frames in reverse order (starting from the last)
        frames = reversed(df["frame"].unique())
        non_empty_frame = None

        # Iterate over frames in reverse to find the first non-zero frame
        for frame in frames:
            frame_data = df[df["frame"] == frame]
            
            # Found non-empty frame
            if not ((frame_data[["x", "y", "z"]] == 0).all().all()):
                non_empty_frame = frame
                break

        # Do the trim if there are non-empty frames
        if non_empty_frame is not None:
            return df[df["frame"] <= non_empty_frame]
        
        return None

    def get_average_data(self, keyword):
        """
        Attempts to return the average data DataFrame based on keyword
        """
        if keyword not in self.average_data:
            return None
        
        return self.average_data[keyword]
