# Import necessary libraries
import fastf1 as ff1
from fastf1 import plotting
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

# Enable the cache
ff1.Cache.enable_cache('/Users/Jackson/Documents/f1/f1_cache')

# Set plotting styles and suppress future warnings
plotting.setup_mpl(misc_mpl_mods=False, color_scheme=None)

# Define the list of recent races
# We'll get the last 6 races up to the latest race in 2023
current_year = 2024
races_to_analyze = 6

# Get the schedule for the current year
schedule = ff1.get_event_schedule(current_year)

# Filter out any events that haven't occurred yet
completed_events = schedule[schedule['EventDate'] < pd.Timestamp.now()]

# Ensure we have enough races
if len(completed_events) < races_to_analyze:
    races_to_analyze = len(completed_events)

# Get the last completed race index
last_race_index = completed_events.index[-1]

# Get the indexes for the last 6 races
race_indexes = range(last_race_index - races_to_analyze + 1, last_race_index + 1)

# Get the list of events for the last 6 races
events = schedule.loc[race_indexes]

# Drivers to analyze
drivers = ['NOR', 'PIA']  # Norris and Piastri's driver codes

# Initialize dictionary to hold telemetry data
telemetry_data = {}

for idx, event in events.iterrows():
    print(f"Processing {event['EventName']}...")

    try:
        # Load the race session
        session = ff1.get_session(event['EventDate'].year, event['EventName'], 'R')  # 'R' for Race
        session.load()

        # Initialize dictionary to hold data for this event
        telemetry_data[event['EventName']] = {}

        for driver_code in drivers:
            # Get the driver's laps
            laps = session.laps.pick_driver(driver_code)

            # Check if laps are available
            if laps.empty:
                print(f"No laps found for {driver_code} in {event['EventName']}")
                continue

            # Get the fastest lap for the driver
            fastest_lap = laps.pick_fastest()

            # Get telemetry data for the fastest lap
            telemetry = fastest_lap.get_telemetry()

            # Store the telemetry data
            telemetry_data[event['EventName']][driver_code] = telemetry

    except Exception as e:
        print(f"An error occurred for {event['EventName']}: {e}")
        continue

# Generate heat maps for each track
for event_name, drivers_data in telemetry_data.items():
    for driver_code, telemetry in drivers_data.items():
        print(f"Generating heat map for {event_name} - {driver_code}")

        # Create a scatter plot colored by speed
        x = telemetry['X']  # X coordinate of the car position
        y = telemetry['Y']  # Y coordinate of the car position
        speed = telemetry['Speed']  # Speed of the car

        # Create a figure and axis
        plt.figure(figsize=(12, 6))
        cmap = mpl.cm.plasma  # Color map for speed visualization

        # Create the scatter plot
        sc = plt.scatter(x, y, c=speed, cmap=cmap, s=1)
        plt.axis('off')
        plt.title(f"{event_name} - {driver_code}")

        # Add a color bar
        cbar = plt.colorbar(sc)
        cbar.set_label('Speed [km/h]')

        # Save the figure
        plt.savefig(f"{event_name}_{driver_code}_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()