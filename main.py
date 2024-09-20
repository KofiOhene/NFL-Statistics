# main.py

from scripts.analysis import calculate_average_yards, top_n_players, show_summary_statistics
from scripts.analysis import calculate_passer_rating, calculate_game_consistency, home_vs_away_performance
from scripts.analysis import calculate_yards_per_carry, calculate_td_rate_rb, calculate_rb_consistency
from scripts.analysis import calculate_yards_per_reception, calculate_yac_percentage
from scripts.analysis import top_qbs_passing_touchdowns, top_qbs_passing_yards, top_qbs_rushing_yards
from scripts.analysis import top_wrs_receptions, top_wrs_receiving_yards, top_wrs_receiving_tds
from scripts.analysis import top_rbs_rushing_tds, top_rbs_avg_rushing_yards_per_game
from scripts.data_processing import load_data, preprocess_data
from clustering import run_kmeans_clustering, analyze_clusters

if __name__ == "__main__":
    # Define the file paths for the datasets using relative paths
    qb_file_path = "data/Game_Logs_Quarterback.csv"
    rb_file_path = "data/Game_Logs_Runningback.csv"
    wr_te_file_path = "data/Game_Logs_Wide_Receiver_and_Tight_End.csv"

    # Load data using the relative file paths
    qb_df = load_data(qb_file_path)
    rb_df = load_data(rb_file_path)
    wr_te_df = load_data(wr_te_file_path)

    # Preprocess data and handle null values
    qb_df_clean = preprocess_data(qb_df, "Passing Yards")
    rb_df_clean = preprocess_data(rb_df, "Rushing Yards")
    wr_te_df_clean = preprocess_data(wr_te_df, "Receiving Yards")  # This will ensure 'Position' is not null

    # Proceed with analysis...

    # 1. Perform basic analysis for quarterbacks: average passing yards
    avg_qb_yards = calculate_average_yards(qb_df_clean, "Passing Yards")
    print("Top 10 Quarterbacks by Average Passing Yards:")
    avg_qb_yards.show(10)

    # 2. Advanced Analysis: Calculate QB passer rating
    print("Top 10 Quarterbacks by Passer Rating:")
    qb_ratings = calculate_passer_rating(qb_df_clean)
    qb_ratings.show(10)

    # 3. Advanced Analysis: Calculate player consistency (standard deviation)
    print("QB Consistency (Standard Deviation of Passing Yards):")
    qb_consistency = calculate_game_consistency(qb_df_clean, "Passing Yards")
    qb_consistency.show(10)

    # 4. Performance by location (Home vs Away)
    print("Home vs Away Performance (Passing Yards):")
    home_vs_away = home_vs_away_performance(qb_df_clean, "Passing Yards")
    home_vs_away.show()

    # Perform analysis for running backs: top 5 rushing yards
    print("Top 5 Running Backs by Rushing Yards:")
    top_rbs = top_n_players(rb_df_clean, "Rushing Yards", n=5)
    top_rbs.show()

    # Running Back Analysis
    print("Yards per Carry for Running Backs:")
    ypc_rb = calculate_yards_per_carry(rb_df_clean)
    ypc_rb.show(10)

    print("Touchdown Rate for Running Backs:")
    td_rate_rb = calculate_td_rate_rb(rb_df_clean)
    td_rate_rb.show(10)

    print("Consistency of Running Backs (Standard Deviation of Rushing Yards):")
    rb_consistency = calculate_rb_consistency(rb_df_clean)
    rb_consistency.show(10)

    # Wide Receiver and Tight End Analysis
    print("Yards per Reception for Wide Receivers and Tight Ends:")
    ypr_wr_te = calculate_yards_per_reception(wr_te_df_clean)
    ypr_wr_te.show(10)

    # Top 10 QBs based on Passing Touchdowns
    print("Top 10 QBs based on Passing Touchdowns:")
    top_qbs_td = top_qbs_passing_touchdowns(qb_df_clean)
    top_qbs_td.show(10)

    # Top 10 QBs based on Total Passing Yards
    print("Top 10 QBs based on Total Passing Yards:")
    top_qbs_yards = top_qbs_passing_yards(qb_df_clean)
    top_qbs_yards.show(10)

    # Top 10 QBs based on Rushing Yards
    print("Top 10 QBs based on Rushing Yards:")
    top_qbs_rush_yards = top_qbs_rushing_yards(qb_df_clean)
    top_qbs_rush_yards.show(10)

    # Top 10 WRs based on Receptions
    print("Top 10 WRs based on Receptions:")
    top_wrs_recep = top_wrs_receptions(wr_te_df_clean)
    top_wrs_recep.show(10)

    # Top 10 WRs based on Receiving Yards
    print("Top 10 WRs based on Receiving Yards:")
    top_wrs_yards = top_wrs_receiving_yards(wr_te_df_clean)
    top_wrs_yards.show(10)

    # Top 10 WRs based on Receiving TDs
    print("Top 10 WRs based on Receiving TDs:")
    top_wrs_tds = top_wrs_receiving_tds(wr_te_df_clean)
    top_wrs_tds.show(10)

    # Top 10 RBs based on Rushing Touchdowns
    print("Top 10 RBs based on Rushing Touchdowns:")
    top_rbs_tds = top_rbs_rushing_tds(rb_df_clean)
    top_rbs_tds.show(10)

    # Top RBs by Average Rushing Yards per Game
    print("Top RBs by Average Rushing Yards per Game:")
    top_rbs_avg_yards = top_rbs_avg_rushing_yards_per_game(rb_df_clean)
    top_rbs_avg_yards.show(10)

    # Skip Yards After Catch analysis if the column doesn't exist
    if "Yards After Catch" in wr_te_df_clean.columns:
        print("Yards After Catch Percentage for Wide Receivers and Tight Ends:")
        yac_percentage_wr_te = calculate_yac_percentage(wr_te_df_clean)
        yac_percentage_wr_te.show(10)
    else:
        print("Yards After Catch column not found. Skipping YAC analysis.")

    # Show summary statistics for wide receivers and tight ends
    # print("Summary statistics for Wide Receivers and Tight Ends:")
    # show_summary_statistics(wr_te_df_clean)

# Assuming you have qb_df_clean loaded
clustered_qb_df = run_kmeans_clustering(qb_df_clean)
analyze_clusters(clustered_qb_df)
