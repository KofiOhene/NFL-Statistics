# data_processing.py

from pyspark.sql import SparkSession


def create_spark_session(app_name="NFL Data Analysis"):
    """
    Creates and returns a SparkSession.
    """
    return SparkSession.builder.appName(app_name).getOrCreate()


def load_data(file_path):
    """
    Loads data from a CSV file into a Spark DataFrame.

    :param file_path: Path to the CSV file
    :return: Spark DataFrame
    """
    spark = create_spark_session()
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    return df


def preprocess_data(df, yard_column):
    """
    Preprocess data by removing null values and casting the yard column to integer.

    :param df: Input Spark DataFrame
    :param yard_column: The column to process (e.g., Passing Yards, Rushing Yards)
    :return: Preprocessed Spark DataFrame
    """
    df_clean = df.dropna(subset=["Name", yard_column])
    df_clean = df_clean.withColumn(yard_column, df_clean[yard_column].cast("integer"))
    return df_clean
