import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scripts.data_processing import load_data, preprocess_data
from scripts.analysis import top_qbs_passing_yards, top_wrs_receiving_yards, top_qbs_passing_touchdowns


def plot_qb_passing_yards(qb_data):
    """
    Plots the top 10 QBs by passing yards.
    """
    top_qbs = top_qbs_passing_yards(qb_data).toPandas()  # Avoid using 'qb_df' as the variable name
    plt.figure(figsize=(10, 6))
    sns.barplot(x='sum(Passing Yards)', y='Name', data=top_qbs, palette='Blues_d')
    plt.title('Top 10 QBs by Passing Yards', fontsize=16)
    plt.xlabel('Total Passing Yards')
    plt.ylabel('Quarterback')
    plt.tight_layout()
    plt.show()


def plot_wr_receiving_yards(wr_te_data):
    """
    Plots the top 10 WRs by receiving yards.
    """
    top_wrs = top_wrs_receiving_yards(wr_te_data).toPandas()  # Avoid using 'wr_te_df' as the variable name
    plt.figure(figsize=(10, 6))
    sns.barplot(x='sum(Receiving Yards)', y='Name', data=top_wrs, palette='Greens_d')
    plt.title('Top 10 WRs by Receiving Yards', fontsize=16)
    plt.xlabel('Total Receiving Yards')
    plt.ylabel('Wide Receiver')
    plt.tight_layout()
    plt.show()


def plot_qb_td_distribution(qb_data):
    top_qbs = top_qbs_passing_touchdowns(qb_data).toPandas()
    plt.figure(figsize=(8, 8))
    plt.pie(top_qbs['sum(TD Passes)'], labels=top_qbs['Name'], autopct='%1.1f%%', startangle=140,
            colors=sns.color_palette("Blues_d"))
    plt.title('TD Distribution Among Top 5 QBs')
    plt.tight_layout()
    plt.show()


def plot_qb_performance_over_time(qb_data, player_name):
    player_data = qb_data.filter(qb_data["Name"] == player_name).toPandas()
    print(f"Data for {player_name}: {player_data.shape}")  # Check if data is loaded
    if not player_data.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Week', y='Passing Yards', data=player_data, marker="o", color="b")
        plt.title(f'{player_name} Passing Yards Over Time')
        plt.xlabel('Week')
        plt.ylabel('Passing Yards')
        plt.tight_layout()
        plt.show()
    else:
        print(f"No data available for {player_name}.")


def plot_passing_yards_vs_tds(qb_data):
    qb_data_pd = qb_data.select("Passing Yards", "TD Passes").toPandas()
    if qb_data_pd.empty:
        print("No data available for Passing Yards vs Passing TDs plot.")
        return
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Passing Yards', y='TD Passes', data=qb_data_pd, color="blue")
    plt.title('Passing Yards vs Passing TDs')
    plt.xlabel('Passing Yards')
    plt.ylabel('Passing TDs')
    plt.tight_layout()
    plt.show()

    print(qb_data.columns)


def plot_passing_yards_distribution(qb_data):
    qb_data_pd = qb_data.select("Passing Yards").toPandas()
    plt.figure(figsize=(8, 6))
    sns.boxplot(y='Passing Yards', data=qb_data_pd)
    plt.title('Distribution of Passing Yards')
    plt.ylabel('Passing Yards')
    plt.tight_layout()
    plt.show(block=False)


def plot_qb_performance_heatmap(qb_data):
    qb_data_pd = qb_data.select("Passing Yards", "TD Passes", "Ints", "Sacks").toPandas()

    # Convert non-numeric values like '--' to NaN
    qb_data_pd.replace('--', np.nan, inplace=True)

    # Convert columns to numeric types
    qb_data_pd = qb_data_pd.apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values, or alternatively, you can fill them with 0 or the mean
    qb_data_pd.dropna(inplace=True)

    # Now you can safely calculate the correlation matrix and plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(qb_data_pd.corr(), annot=True, cmap="coolwarm")
    plt.title('Correlation Between QB Metrics')
    plt.tight_layout()
    plt.show()


def plot_home_vs_away_passing_yards(qb_data):
    qb_data_pd = qb_data.select("Passing Yards", "Home or Away").toPandas()
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Home or Away', y='Passing Yards', data=qb_data_pd)
    plt.title('Home vs Away Passing Yards Distribution')
    plt.tight_layout()
    plt.show()


def plot_rushing_yards_histogram(rb_data):
    rb_data_pd = rb_data.select("Rushing Yards").toPandas()
    plt.figure(figsize=(8, 6))
    sns.histplot(rb_data_pd['Rushing Yards'], bins=20, kde=True, color="green")
    plt.title('Distribution of Rushing Yards')
    plt.xlabel('Rushing Yards')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


# Call with player_name, e.g., plot_qb_performance_over_time(qb_df_clean, "Tom Brady")


if __name__ == "__main__":
    # Load data
    qb_file_path = "data/Game_Logs_Quarterback.csv"
    wr_te_file_path = "data/Game_Logs_Wide_Receiver_and_Tight_End.csv"

    # Preprocess QB data
    qb_data = load_data(qb_file_path)  # Changed variable name to 'qb_data' to avoid shadowing
    qb_data_clean = preprocess_data(qb_data, "Passing Yards")

    # Preprocess WR/TE data
    wr_te_data = load_data(wr_te_file_path)  # Changed variable name to 'wr_te_data' to avoid shadowing
    wr_te_data_clean = preprocess_data(wr_te_data, "Receiving Yards")

    # Generate visualizations
    plot_qb_passing_yards(qb_data_clean)
    plot_wr_receiving_yards(wr_te_data_clean)
    plot_qb_td_distribution(qb_data_clean)
    plot_qb_performance_over_time(qb_data_clean, "Brady, Tom")  # Example player
    plot_passing_yards_vs_tds(qb_data_clean)
    plot_passing_yards_distribution(qb_data_clean)
    plot_qb_performance_heatmap(qb_data_clean)
    plot_home_vs_away_passing_yards(qb_data_clean)
    plot_rushing_yards_histogram(qb_data_clean)