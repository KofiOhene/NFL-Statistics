# analysis.py

from pyspark.sql import functions as F
from pyspark.sql.window import Window


# 1. Top 10 QBs based on Passing Touchdowns
def top_qbs_passing_touchdowns(df):
    """
    Returns the top 10 QBs based on passing touchdowns.

    :param df: Spark DataFrame containing QB data
    :return: DataFrame with top 10 QBs by passing touchdowns
    """
    # Cast 'TD Passes' to integer to ensure it can be summed
    df = df.withColumn("TD Passes", df["TD Passes"].cast("integer"))
    return df.groupBy("Name").sum("TD Passes").orderBy(F.desc("sum(TD Passes)")).limit(10)


# 2. Top 10 QBs based on Total Passing Yards
def top_qbs_passing_yards(df):
    """
    Returns the top 10 QBs based on total passing yards.

    :param df: Spark DataFrame containing QB data
    :return: DataFrame with top 10 QBs by passing yards
    """
    # Cast 'Passing Yards' to integer
    df = df.withColumn("Passing Yards", df["Passing Yards"].cast("integer"))
    return df.groupBy("Name").sum("Passing Yards").orderBy(F.desc("sum(Passing Yards)")).limit(10)


# 3. Top 10 QBs based on Rushing Yards
def top_qbs_rushing_yards(df):
    """
    Returns the top 10 QBs based on rushing yards.

    :param df: Spark DataFrame containing QB data
    :return: DataFrame with top 10 QBs by rushing yards
    """
    # Cast 'Rushing Yards' to integer
    df = df.withColumn("Rushing Yards", df["Rushing Yards"].cast("integer"))
    return df.groupBy("Name").sum("Rushing Yards").orderBy(F.desc("sum(Rushing Yards)")).limit(10)


# 4. Top 10 WRs based on Receptions
def top_wrs_receptions(df):
    """
    Returns the top 10 WRs based on receptions.

    :param df: Spark DataFrame containing WR data
    :return: DataFrame with top 10 WRs by receptions
    """
    # Cast 'Receptions' to integer
    df = df.withColumn("Receptions", df["Receptions"].cast("integer"))
    return df.groupBy("Name").sum("Receptions").orderBy(F.desc("sum(Receptions)")).limit(10)


# 5. Top 10 WRs based on Receiving Yards
def top_wrs_receiving_yards(df):
    """
    Returns the top 10 WRs based on receiving yards.

    :param df: Spark DataFrame containing WR data
    :return: DataFrame with top 10 WRs by receiving yards
    """
    # Cast 'Receiving Yards' to integer
    df = df.withColumn("Receiving Yards", df["Receiving Yards"].cast("integer"))
    return df.groupBy("Name").sum("Receiving Yards").orderBy(F.desc("sum(Receiving Yards)")).limit(10)


# 6. Top 10 WRs based on Receiving TDs
def top_wrs_receiving_tds(df):
    """
    Returns the top 10 WRs based on receiving touchdowns.

    :param df: Spark DataFrame containing WR data
    :return: DataFrame with top 10 WRs by receiving touchdowns
    """
    # Cast 'Receiving TDs' to integer
    df = df.withColumn("Receiving TDs", df["Receiving TDs"].cast("integer"))
    return df.groupBy("Name").sum("Receiving TDs").orderBy(F.desc("sum(Receiving TDs)")).limit(10)


# 7. Top 10 RBs based on Rushing Touchdowns
def top_rbs_rushing_tds(df):
    """
    Returns the top 10 RBs based on rushing touchdowns.

    :param df: Spark DataFrame containing RB data
    :return: DataFrame with top 10 RBs by rushing touchdowns
    """
    # Cast 'Rushing TDs' to integer
    df = df.withColumn("Rushing TDs", df["Rushing TDs"].cast("integer"))
    return df.groupBy("Name").sum("Rushing TDs").orderBy(F.desc("sum(Rushing TDs)")).limit(10)


# 8. Top RBs by Average Rushing Yards per Game
def top_rbs_avg_rushing_yards_per_game(df):
    """
    Returns the top RBs by average rushing yards per game.

    :param df: Spark DataFrame containing RB data
    :return: DataFrame with top RBs by average rushing yards per game
    """
    df = df.withColumn("Rushing Yards per Game", (df["Rushing Yards"] / df["Games Played"]).cast("float"))
    return df.groupBy("Name").avg("Rushing Yards per Game").orderBy(F.desc("avg(Rushing Yards per Game)")).limit(10)


def calculate_average_yards(df, yard_column="Passing Yards"):
    """
    Calculates the average yards for each player.

    :param df: Spark DataFrame containing player data
    :param yard_column: The column name that contains yard data
    :return: Spark DataFrame with player names and their average yards
    """
    avg_yards = df.groupBy("Name").avg(yard_column).orderBy(f"avg({yard_column})", ascending=False)
    return avg_yards


def top_n_players(df, yard_column="Passing Yards", n=10):
    """
    Returns the top N players based on the specified yard column.

    :param df: Spark DataFrame containing player data
    :param yard_column: The column name that contains yard data
    :param n: Number of top players to return
    :return: Spark DataFrame with top N players
    """
    top_players = df.groupBy("Name") \
        .sum(yard_column) \
        .orderBy(f"sum({yard_column})", ascending=False) \
        .limit(n)
    return top_players


def show_summary_statistics(df):
    """
    Displays summary statistics for numeric columns in the DataFrame.

    :param df: Spark DataFrame
    """
    df.describe().show()


def calculate_passer_rating(df):
    """
    Calculates the Passer Rating for quarterbacks.

    :param df: Spark DataFrame containing QB data
    :return: DataFrame with passer rating
    """
    # Passer rating formula:
    df = df.withColumn("Comp%", df["Passes Completed"] / df["Passes Attempted"] * 100)
    df = df.withColumn("Yards/Attempt", df["Passing Yards"] / df["Passes Attempted"])
    df = df.withColumn("TD%", df["TD Passes"] / df["Passes Attempted"] * 100)
    df = df.withColumn("Int%", df["Ints"] / df["Passes Attempted"] * 100)

    df = df.withColumn(
        "Passer Rating",
        ((df["Comp%"] - 30) / 20 + (df["Yards/Attempt"] - 3) / 4 + df["TD%"] / 5 + 2.375 - df["Int%"] * 25) * 100 / 6
    )
    return df.select("Name", "Passer Rating").orderBy("Passer Rating", ascending=False)


def calculate_game_consistency(df, yard_column="Passing Yards"):
    """
    Calculate the standard deviation of passing yards for each quarterback to measure consistency.

    :param df: Spark DataFrame containing player data
    :param yard_column: The column to measure consistency on (e.g., Passing Yards)
    :return: DataFrame with standard deviation for each player
    """
    consistency = df.groupBy("Name").agg(F.stddev(yard_column).alias("Consistency")).orderBy("Consistency")
    return consistency


def home_vs_away_performance(df, yard_column="Passing Yards"):
    """
    Analyze home vs away performance for quarterbacks.

    :param df: Spark DataFrame containing player data
    :param yard_column: The column to analyze (e.g., Passing Yards)
    :return: DataFrame showing home vs away performance
    """
    home_performance = df.filter(df["Home or Away"] == "Home") \
        .groupBy("Name") \
        .agg(F.avg(yard_column).alias("Avg Home Yards"))

    away_performance = df.filter(df["Home or Away"] == "Away") \
        .groupBy("Name") \
        .agg(F.avg(yard_column).alias("Avg Away Yards"))

    return home_performance.join(away_performance, "Name")


def calculate_yards_per_carry(df):
    """
    Calculates yards per carry for running backs.

    :param df: Spark DataFrame containing RB data
    :return: DataFrame with yards per carry (YPC) for each running back
    """
    df = df.withColumn("Yards_Per_Carry", df["Rushing Yards"] / df["Rushing Attempts"])
    return df.select("Name", "Yards_Per_Carry").orderBy("Yards_Per_Carry", ascending=False)


def calculate_td_rate_rb(df):
    """
    Calculates touchdown rate (TDs per rushing attempt) for running backs.

    :param df: Spark DataFrame containing RB data
    :return: DataFrame with TD rate for each running back
    """
    df = df.withColumn("TD_Rate", df["Rushing TDs"] / df["Rushing Attempts"] * 100)
    return df.select("Name", "TD_Rate").orderBy("TD_Rate", ascending=False)


def calculate_yards_per_reception(df):
    """
    Calculates yards per reception for wide receivers and tight ends.

    :param df: Spark DataFrame containing WR/TE data
    :return: DataFrame with yards per reception (YPR) for each player
    """
    df = df.withColumn("Yards_Per_Reception", df["Receiving Yards"] / df["Receptions"])
    return df.select("Name", "Yards_Per_Reception").orderBy("Yards_Per_Reception", ascending=False)


def calculate_td_percentage(df):
    """
    Calculates the percentage of receptions that result in touchdowns for wide receivers and tight ends.

    :param df: Spark DataFrame containing WR/TE data
    :return: DataFrame with TD percentage for each player
    """
    df = df.withColumn("TD_Percentage", df["Receiving TDs"] / df["Receptions"] * 100)
    return df.select("Name", "TD_Percentage").orderBy("TD_Percentage", ascending=False)


def calculate_yards_per_game(df):
    """
        Calculates the average receiving yards per game for wide receivers and tight ends.

        :param df: Spark DataFrame containing WR/TE data
        :return: DataFrame with yards per game for each player
        """
    df = df.withColumn("Yards_Per_Game", df["Receiving Yards"] / df["Games Played"])
    return df.select("Name", "Yards_Per_Game").orderBy("Yards_Per_Game", ascending=False)


def calculate_yac_percentage(df):
    """
    Calculates yards after catch percentage for wide receivers and tight ends.

    :param df: Spark DataFrame containing WR/TE data
    :return: DataFrame with YAC percentage for each player
    """
    df = df.withColumn("YAC_Percentage", df["Yards After Catch"] / df["Receiving Yards"] * 100)
    return df.select("Name", "YAC_Percentage").orderBy("YAC_Percentage", ascending=False)


def calculate_rb_consistency(df):
    """
    Calculate the standard deviation of rushing yards for each running back.

    :param df: Spark DataFrame containing RB data
    :return: DataFrame with standard deviation of rushing yards for each RB
    """
    consistency = df.groupBy("Name").agg(F.stddev("Rushing Yards").alias("Consistency")).orderBy("Consistency")
    return consistency
