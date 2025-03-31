# -*- coding: utf-8 -*-
"""
Access models run for COVID-19 Inpatient bed supply and Demand
"""

# Install packages if not already installed
# pip install access pandas geopandas matplotlib

# Import libraries
import os
from icecream import ic
import pandas as pd
import geopandas as gpd
from access import Access, weights #, Datasets
import warnings

# access package will have some issues with Python3 but not a problem now
warnings.filterwarnings("ignore", category=FutureWarning)

ic.disable() # disable print statements
#ic.enable()

# Input data paths
root_dir         = os.path.dirname(os.path.dirname(__file__))  # goes up from code/
input_dir        = os.path.join(root_dir, "input_data", "ACCESS_POSTER_INPUT")
output_dir = "../produced_data/ACCESS_POSTER_OUTPUT"
os.makedirs(output_dir, exist_ok=True)

# Load cost matrix once
travel_times_df = pd.read_csv(os.path.join(input_dir, "zcta_hosp_drivetimes_minutes.csv"))

# Get weeks from filenames (supply file pattern)
supply_files = [f for f in os.listdir(input_dir) if f.startswith("cov_inpat_avg_supply_")]
weeks = [f.replace("cov_inpat_avg_supply_", "").replace(".csv", "") for f in supply_files]

# Define weighting functions
gaussian = weights.gaussian(20)
#gravity = weights.gravity(scale=60, alpha=-1)

# Loop over weeks
for week in sorted(weeks):
    print(f"✨Running access model for week: {week}")

    # File paths
    supply_path = os.path.join(input_dir, f"cov_inpat_avg_supply_{week}.csv")
    demand_path = os.path.join(input_dir, f"cov_inpat_avg_demand_{week}.csv")

    # Load data
    pat_demand_df = pd.read_csv(demand_path)
    hosp_supply_df = pd.read_csv(supply_path)

    # Run access model
    A = Access(
        demand_df=pat_demand_df,
        demand_index="ZCTA",
        demand_value="IP_COV_DEMAND",
        supply_df=hosp_supply_df,
        supply_index="THCIC_ID",
        supply_value="inpatient_beds_7_day_avg_interp",
        cost_df=travel_times_df,
        cost_origin="ZCTA",
        cost_dest="THCIC_ID",
        cost_name="DRIVETIME_MINUTES",

        # Needed for FCA Ratio Model
        neighbor_cost_df=travel_times_df,
        neighbor_cost_origin="ZCTA",
        neighbor_cost_dest="THCIC_ID",
        neighbor_cost_name="DRIVETIME_MINUTES"
    )

    # Run models
    #A.weighted_catchment(name="gravity", weight_fn=gravity)
    #A.fca_ratio(name="FCA") # , max_cost=90
    A.two_stage_fca(name="2SFCA_60min", max_cost=60) #, max_cost=30
    A.enhanced_two_stage_fca(name="E2SFCA", weight_fn=gaussian)
    A.three_stage_fca(name="3SFCA")

    # Add COLLECTION_WEEK column
    access_df = A.access_df.reset_index()
    access_df["COLLECTION_WEEK"] = week

    # Save to output file
    output_file = f"access_scores_all_models_{week}.csv"
    access_df.to_csv(os.path.join(output_dir, output_file), index=False)
    print(f"🌸 Saved: {output_file}")

