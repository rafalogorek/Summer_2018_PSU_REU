######################################################################################
#
# Deep Layer Mean Steering Flow Statistics Calculator
#
# Author: Rafal Ogorek
#
# Description: This program makes use of reanalysis data to calculate various
#              statistics for the deep layer mean steering flow winds along the
#              coastline of the southeastern U.S. from 1979 to 2017. Several
#              graphics are also generated to help visualize this data better.
#
# Last updated: 6/5/2018
#
######################################################################################

import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_toolkits.basemap import Basemap


# Description: Converts floats to Datetime objects
# Input: -measurement_times: An array of floats (which should represent a time and date)
# Output: -dates_and_times: An array of Datetime objects that have been converted from
#                           the floats in the measurement_times array
def convertToDatetime(measurement_times):
    dates_and_times = []
    for mt in measurement_times:
        # Convert the current serial time to a number of seconds
        serial_time = mt
        seconds = (serial_time - 25569) * 86400.0

        # Use the number of seconds to get a Datetime object
        dates_and_times.append(datetime.utcfromtimestamp(seconds))

    # If using the timeseries_temp.npz dataset:
    # Dates/times should range from May 30, 1979 at 0000Z to September 28, 2017 at
    # 1800Z, with increments of 6 hours in between. Data only covers times during
    # the Atlantic hurricane season (from May 30 to November 28).
    #
    # For instance, entries 0 through 731 cover from May 30, 1979 at 0000Z to
    # November 28, 1979 at 1800Z; entries 732 through 1463 cover from May 30, 1980
    # at 0000Z to November 28, 1980 at 1800Z; entries 1464 through 2195 cover from
    # May 30, 1981 at 0000Z to November 28, 1981 at 1800Z; etc...

    return dates_and_times


# Description: Populates a wind speed frequency array with the frequency of wind
#              speeds within each 1 m/s interval represented in the wind speed
#              frequency array
# Input: -wind_speeds: An array that stores an array of floats. Each array within
#                      this array represents a different location, while each float
#                      is a wind speed measurement in m/s
#        -wind_speed_freq: An array that will store the frequency of a range of wind
#                          speeds that correspond to its indices (i.e. index 0 will
#                          store the frequency of wind speeds between 0 m/s and 1 m/s;
#                          index 1 will store the frequency of wind speeds between
#                          1 m/s and 2 m/s; etc...). When passed into the functions,
#                          all entries should be 0 since no frequencies have been
#                          calculated yet.
# Output: Nothing, but the wind_speed_freq array should have been modified to now
#         contain the correct frequency value for each wind speed range
def getFrequencies(wind_speeds, wind_speed_freq):
    for loc in wind_speeds:
        for wind_speed in loc:
            # Make the wind speed a positive value and round it down to the nearest
            # integer. Then, increment the frequency of the wind speed range that this
            # value falls in.
            current_ws = int(math.floor(abs(wind_speed)))
            wind_speed_freq[current_ws] = wind_speed_freq[current_ws] + 1


# Description: Obtains the wind speeds recorded at all locations within a specified
#              time interval
# Input: -wind_speeds: An array that stores an array of floats. Each array within
#                      this array represents a different location, while each float
#                      is a wind speed measurement in m/s
#        -left_bound: The index that represents the beginning of the time interval
#                     (included in the time interval that is returned)
#        -right_bound: The index that represents the end of the time interval (not
#                      included in the time interval that is returned)
# Output: -wind_speed_interval: An array that stores an array of floats. It includes
#                               all locations, but only includes the wind speeds that
#                               were measured in the specified time period
def getWindSpeedInterval(wind_speeds, left_bound, right_bound):
    # Intialize wind_speed_interval to have same size as wind_speeds
    wind_speed_interval = [None] * len(wind_speeds)
    i = 0

    # Get data for all locations, but only during the specified time period
    for loc in wind_speeds:
        wind_speed_interval[i] = loc[left_bound:right_bound]
        i = i + 1

    return wind_speed_interval


# Description: Determines how many of the wind speed measurements in the wind speed
#              frequency array can be classified as stagnant flow
# Input: -wind_speed_freq: An array that will store the frequency of a range of wind
#                          speeds that correspond to its indices (i.e. index 0 will
#                          store the frequency of wind speeds between 0 m/s and 1 m/s;
#                          index 1 will store the frequency of wind speeds between
#                          1 m/s and 2 m/s; etc...).
#        -stagnant_flow: An integer signifying the upper threshold for what wind speed
#                        is considered to be stagnant flow
# Output: -num_stag_flow: The number of measurements in the wind_speed_freq array that
#                         are classified as stagnant flow
def calcNumStagFlow(wind_speed_freq, stagnant_flow):
    i = 0
    num_stag_flow = 0

    # Add the value for the current index in wind_speed_freq to the number of stagnant
    # flow measurements until the index that represents the upper threshold for stagnant
    # flow is reached
    while i < stagnant_flow:
        num_stag_flow = num_stag_flow + wind_speed_freq[i]
        i = i + 1

    return num_stag_flow


# Description: Splits the wind speed data into three different arrays, based on the
#              time of the hurricane season that the measurement was taken at
# Input: -wind_speeds: An array that stores an array of floats. Each array within
#                      this array represents a different location, while each float
#                      is a wind speed measurement in m/s
# Output: -wind_speeds_EHS: An array that stores an array of floats. Each array within
#                           this array represents a different location, while each float
#                           is a wind speed measurement in m/s. All wind speeds in this
#                           array were measured in the early part of the Atlantic hurricane
#                           season (late May, June, or July)
#         -wind_speeds_MHS: Like wind_speeds_EHS, except all wind speeds in this array
#                           were measured in the middle part of the Atlantic hurricane
#                           season (August or September)
#         -wind_speeds_LHS: Like wind_speeds_EHS and wind_speeds_MHS, except all wind
#                           speeds in this array were measured in the late part of the
#                           Atlantic hurricane season (October or November)
def divideBySeason(wind_speeds):
    wind_speeds_EHS = [[None]] * len(wind_speeds)
    wind_speeds_MHS = [[None]] * len(wind_speeds)
    wind_speeds_LHS = [[None]] * len(wind_speeds)
    i = 0

    # Go through each location
    while i < len(wind_speeds):
        j = 0
        # Go through each wind speed measurement
        while j < len(wind_speeds[i]):
            # If data was obtained prior to August, add it to the early hurricane season
            # wind speed array
            if (j % 732) < 252:
                if wind_speeds_EHS[i] == [None]:
                    wind_speeds_EHS[i] = [wind_speeds[i][j]]
                else:
                    wind_speeds_EHS[i].append(wind_speeds[i][j])
            # If data was obtained in October or later, add it to the late hurricane season
            # wind speed array
            elif (j % 732) >= 496:
                if wind_speeds_LHS[i] == [None]:
                    wind_speeds_LHS[i] = [wind_speeds[i][j]]
                else:
                    wind_speeds_LHS[i].append(wind_speeds[i][j])
            # Otherwise, the wind speed measurement was take in August or September, so
            # add it to the mid hurricane season wind speed array
            else:
                if wind_speeds_MHS[i] == [None]:
                    wind_speeds_MHS[i] = [wind_speeds[i][j]]
                else:
                    wind_speeds_MHS[i].append(wind_speeds[i][j])
            j = j + 1
        i = i + 1

    return wind_speeds_EHS, wind_speeds_MHS, wind_speeds_LHS


######################################################################################
##                                   Begin Program                                  ##
######################################################################################

# Currently defining stagnant steering flow to be 2 m/s
stagnant_flow = 2

# Load data
data = np.load(sys.argv[1])

# Obtain the locations, times/dates, and the wind speed measurements from the data
locs = data['loc']
measurement_times = data['mydate']
wind_speeds = data['ts']

# Convert floats in the measurement_times array to datetime objects
dates_and_times = convertToDatetime(measurement_times)

# Get rid of data points that are not along the U.S. coast or north of Cape Hatteras
temp_locs = []
for loc in locs:
    # Get rid of points north or east of North Carolina, west of Corpus Christi, and
    # south of Key West
    if loc[0] > -98 and loc[0] <= -75 and loc[1] > 24 and loc[1] <= 36:
        # Also get rid of points along Mexican Coast and around The Bahamas
        if not (loc[0] < -96 and loc[1] < 26) and not (loc[0] > -79 and loc[1] < 30):
            temp_locs.append(loc)

locs = temp_locs

# Determine the maximum wind speed recorded (direction doesn't matter)
max_wind_speed = max(np.amax(wind_speeds), abs(np.amin(wind_speeds)))

######################################################################################
##                         All Locations, Whole Time Period                         ##
######################################################################################

# Analyze all data together, regardless of time or location

# Create an array to store how frequently a wind speed value was recorded (at all
# locations). The zeroeth index of the array will store the frequency of recorded wind
# speeds between 0 m/s and 1 m/s; the first index of the array will store the frequency
# of wind speeds between 1 m/s and 2 m/s; etc...
wind_speed_freq_all = [0] * (max_wind_speed + 1)

# Create wind speed array that will only contain data for the desired period of time
wind_speeds_79_16 = getWindSpeedInterval(wind_speeds, 0, 27816)

# Populate the wind speed frequency array
getFrequencies(wind_speeds_79_16, wind_speed_freq_all)

# Generate a histogram to show the frequency distribution for the wind speeds at all
# measured locations from 1979 to 2017
plt.figure(1, figsize = (20,10))
plt.bar(np.arange(len(wind_speed_freq_all)), wind_speed_freq_all)
plt.xlim(xmax=51)
plt.ylabel('Frequency')
plt.xlabel('Wind Speed (m/s)')
plt.title('Frequency of Wind Speeds Recorded on Southeast U.S. Coasts During ' +
          'Hurricane Seasons from 1979 up to 2017')
plt.savefig('Figures/Wind_Speeds_All_Locations_79-16_Histogram.png')
plt.show()


# Create arrays to store how frequently a wind speed value was recorded (at all
# locations). The zeroeth index of the array will store the frequency of recorded wind
# speeds between 0 m/s and 1 m/s; the first index of the array will store the frequency
# of wind speeds between 1 m/s and 2 m/s; etc...
wind_speed_freq_all_EHS = [0] * (max_wind_speed + 1)
wind_speed_freq_all_MHS = [0] * (max_wind_speed + 1)
wind_speed_freq_all_LHS = [0] * (max_wind_speed + 1)

# Divide the wind speed data up by which part of the hurricane season they were taken in
# (early hurricane season (June, July), mid  hurricane season (August, September), and
# late hurricane season (October, November))
wind_speeds_79_16_EHS, wind_speeds_79_16_MHS, wind_speeds_79_16_LHS = divideBySeason(wind_speeds_79_16)

# Populate the wind speed frequency arrays for each of the three parts of the hurricane season
getFrequencies(wind_speeds_79_16_EHS, wind_speed_freq_all_EHS)
getFrequencies(wind_speeds_79_16_MHS, wind_speed_freq_all_MHS)
getFrequencies(wind_speeds_79_16_LHS, wind_speed_freq_all_LHS)

# Generate histograms to show the frequency distribution for the wind speeds at all
# measured locations for each of the three parts of the hurricane season from 1979 to 2017
plt.figure(1, figsize = (20,10))
plt.bar(np.arange(len(wind_speed_freq_all_MHS)), wind_speed_freq_all_EHS)
plt.xlim(xmax=51)
plt.ylabel('Frequency')
plt.xlabel('Wind Speed (m/s)')
plt.title('Frequency of Wind Speeds Recorded on Southeast U.S. Coasts During ' +
          'Late May, June, and July from 1979 up to 2017')
plt.savefig('Figures/Wind_Speeds_All_Locations_79-16_EHS_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,10))
plt.bar(np.arange(len(wind_speed_freq_all_MHS)), wind_speed_freq_all_MHS)
plt.xlim(xmax=51)
plt.ylabel('Frequency')
plt.xlabel('Wind Speed (m/s)')
plt.title('Frequency of Wind Speeds Recorded on Southeast U.S. Coasts During ' +
          'August and September from 1979 up to 2017')
plt.savefig('Figures/Wind_Speeds_All_Locations_79-16_MHS_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,10))
plt.bar(np.arange(len(wind_speed_freq_all_LHS)), wind_speed_freq_all_LHS)
plt.xlim(xmax=51)
plt.ylabel('Frequency')
plt.xlabel('Wind Speed (m/s)')
plt.title('Frequency of Wind Speeds Recorded on Southeast U.S. Coasts During ' +
          'October and November from 1979 up to 2017')
plt.savefig('Figures/Wind_Speeds_All_Locations_79-16_LHS_Histogram.png')
plt.show()

######################################################################################
##                        All Locations, Halved Time Periods                        ##
######################################################################################

# Split the data into two time periods: One from 1979 up to 1998, and the other from 1998
# up to 2017

#print(len(dates_and_times))
#print(dates_and_times[13907])
#print(dates_and_times[27815])

# Create arrays to store how frequently a wind speed value was recorded (at all locations
# for each time period. The zeroeth index of the array will store the frequency of recorded
# wind speeds between 0 m/s and 1 m/s; the first index of the array will store the frequency
# of wind speeds between 1 m/s and 2 m/s; etc...
wind_speed_freq_79_97 = [0] * (max_wind_speed + 1)
wind_speed_freq_98_16 = [0] * (max_wind_speed + 1)

# Create wind speed arrays that will only contain data for the desired period of time
wind_speeds_79_97 = getWindSpeedInterval(wind_speeds, 0, 13908)
wind_speeds_98_16 = getWindSpeedInterval(wind_speeds, 13908, 27816)

# Populate the wind speed frequency arrays
getFrequencies(wind_speeds_79_97, wind_speed_freq_79_97)
getFrequencies(wind_speeds_98_16, wind_speed_freq_98_16)

# Generate histograms to show the frequency distribution for the wind speeds at all
# measured locations from 1979 up to 1998 and from 1998 up to 2017
plt.figure(1, figsize = (20,10))
plt.subplot(211)
plt.bar(np.arange(len(wind_speed_freq_79_97)), wind_speed_freq_79_97)
plt.xlim(xmax=51)
plt.ylim(ymax=550000)
plt.ylabel('Frequency')
plt.xlabel('Wind Speed (m/s)')
plt.title('Frequency of Wind Speeds Recorded on Southeast U.S. Coasts During ' +
          'Hurricane Seasons from 1979 up to 1998')

plt.subplot(212)
plt.bar(np.arange(len(wind_speed_freq_98_16)), wind_speed_freq_98_16)
plt.xlim(xmax=51)
plt.ylim(ymax=550000)
plt.ylabel('Frequency')
plt.xlabel('Wind Speed (m/s)')
plt.title('Frequency of Wind Speeds Recorded on Southeast U.S. Coasts During ' +
          'Hurricane Seasons from 1998 up to 2017')

plt.subplots_adjust(left = 0.125, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.2,
                    hspace = 0.5)
plt.savefig('Figures/Wind_Speeds_All_Locations_79-97_98-16_Histogram.png')
plt.show()

# Plot the two time periods side by side, rather than on two separate plots
fig, ax = plt.subplots(figsize = (20,10))
bar_width = 0.35
ax.bar(np.arange(len(wind_speed_freq_79_97)), wind_speed_freq_79_97, bar_width,
                color='b', label='1979-1997')
ax.bar(np.arange(len(wind_speed_freq_98_16)) + bar_width, wind_speed_freq_98_16,
                bar_width, color='r', label='1998-2016')

ax.set_xlim(xmax=51)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Frequency')
ax.set_title('Frequency of Wind Speeds Recorded on Southeast U.S. Coasts During ' +
             'Hurricane Seasons\nfrom 1979 up to 1998 vs. Hurricane Seasons from ' +
             '1998 up to 2017')

ax.legend()
fig.savefig('Figures/Wind_Speeds_All_Locations_79-97_98-16_SBS_Histogram.png')
plt.show()

# Determine differences in the wind speed frequencies between the two time periods
i = 0
wind_speed_freq_diff = [0] * len(wind_speed_freq_79_97)
while i < len(wind_speed_freq_79_97):
    wind_speed_freq_diff[i] = wind_speed_freq_98_16[i] - wind_speed_freq_79_97[i]
    i = i + 1

# Create a histogram that shows the difference between the wind speed frequencies for
# the two time periods
plt.figure(1, figsize = (20,10))
plt.bar(np.arange(len(wind_speed_freq_diff)), wind_speed_freq_diff)
plt.xlim(xmax=51)
plt.ylabel('Frequency Difference')
plt.xlabel('Wind Speed (m/s)')
plt.title('Difference Between Frequency of Wind Speeds Recorded During ' +
          'Hurricane Seasons\nfrom 1979 up to 1998 and During Hurricane ' +
          'Seasons from 1998 up to 2017')
plt.savefig('Figures/Wind_Speed_Diff_Between_78-97_and_98-16_All_Locations_Histogram.png')
plt.show()


# Create arrays to store how frequently a wind speed value was recorded (at all
# locations). The zeroeth index of the array will store the frequency of recorded wind
# speeds between 0 m/s and 1 m/s; the first index of the array will store the frequency
# of wind speeds between 1 m/s and 2 m/s; etc...
wind_speed_freq_79_97_EHS = [0] * (max_wind_speed + 1)
wind_speed_freq_79_97_MHS = [0] * (max_wind_speed + 1)
wind_speed_freq_79_97_LHS = [0] * (max_wind_speed + 1)
wind_speed_freq_98_16_EHS = [0] * (max_wind_speed + 1)
wind_speed_freq_98_16_MHS = [0] * (max_wind_speed + 1)
wind_speed_freq_98_16_LHS = [0] * (max_wind_speed + 1)

# Divide the wind speed data up by which part of the hurricane season they were taken in
# (early hurricane season (June, July), mid  hurricane season (August, September), and
# late hurricane season (October, November))
wind_speeds_79_97_EHS, wind_speeds_79_97_MHS, wind_speeds_79_97_LHS = divideBySeason(wind_speeds_79_97)
wind_speeds_98_16_EHS, wind_speeds_98_16_MHS, wind_speeds_98_16_LHS = divideBySeason(wind_speeds_98_16)

# Populate the wind speed frequency arrays for each of the three parts of the hurricane
# season for each time period
getFrequencies(wind_speeds_79_97_EHS, wind_speed_freq_79_97_EHS)
getFrequencies(wind_speeds_79_97_MHS, wind_speed_freq_79_97_MHS)
getFrequencies(wind_speeds_79_97_LHS, wind_speed_freq_79_97_LHS)
getFrequencies(wind_speeds_98_16_EHS, wind_speed_freq_98_16_EHS)
getFrequencies(wind_speeds_98_16_MHS, wind_speed_freq_98_16_MHS)
getFrequencies(wind_speeds_98_16_LHS, wind_speed_freq_98_16_LHS)

# Plot the two time periods side by side for each part of the hurricane season
fig, ax = plt.subplots(figsize = (20,10))
bar_width = 0.35
ax.bar(np.arange(len(wind_speed_freq_79_97_EHS)), wind_speed_freq_79_97_EHS, bar_width,
                color='b', label='1979-1997')
ax.bar(np.arange(len(wind_speed_freq_98_16_EHS)) + bar_width, wind_speed_freq_98_16_EHS,
                bar_width, color='r', label='1998-2016')
ax.set_xlim(xmax=51)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Frequency')
ax.set_title('Frequency of Wind Speeds Recorded on Southeast U.S. Coasts in late May, ' +
             'June, and July During\nHurricane Seasons from 1979 up to 1998 vs. Hurricane ' +
             'Seasons from 1998 up to 2017')
ax.legend()
fig.savefig('Figures/Wind_Speeds_All_Locations_79-97_98-16_EHS_SBS_Histogram.png')
plt.show()

fig, ax = plt.subplots(figsize = (20,10))
bar_width = 0.35
ax.bar(np.arange(len(wind_speed_freq_79_97_MHS)), wind_speed_freq_79_97_MHS, bar_width,
                color='b', label='1979-1997')
ax.bar(np.arange(len(wind_speed_freq_98_16_MHS)) + bar_width, wind_speed_freq_98_16_MHS,
                bar_width, color='r', label='1998-2016')
ax.set_xlim(xmax=51)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Frequency')
ax.set_title('Frequency of Wind Speeds Recorded on Southeast U.S. Coasts in August ' +
             'and September During\nHurricane Seasons from 1979 up to 1998 vs. Hurricane ' +
             'Seasons from 1998 up to 2017')
ax.legend()
fig.savefig('Figures/Wind_Speeds_All_Locations_79-97_98-16_MHS_SBS_Histogram.png')
plt.show()

fig, ax = plt.subplots(figsize = (20,10))
bar_width = 0.35
ax.bar(np.arange(len(wind_speed_freq_79_97_LHS)), wind_speed_freq_79_97_LHS, bar_width,
                color='b', label='1979-1997')
ax.bar(np.arange(len(wind_speed_freq_98_16_LHS)) + bar_width, wind_speed_freq_98_16_LHS,
                bar_width, color='r', label='1998-2016')
ax.set_xlim(xmax=51)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Frequency')
ax.set_title('Frequency of Wind Speeds Recorded on Southeast U.S. Coasts in October ' +
             'and November During\nHurricane Seasons from 1979 up to 1998 vs. Hurricane ' +
             'Seasons from 1998 up to 2017')
ax.legend()
fig.savefig('Figures/Wind_Speeds_All_Locations_79-97_98-16_LHS_SBS_Histogram.png')
plt.show()

# Determine differences in the wind speed frequencies between the two time periods
# for each part of the hurricane season
i = 0
wind_speed_freq_EHS_diff = [0] * len(wind_speed_freq_79_97_EHS)
while i < len(wind_speed_freq_79_97_EHS):
    wind_speed_freq_EHS_diff[i] = wind_speed_freq_98_16_EHS[i] - wind_speed_freq_79_97_EHS[i]
    i = i + 1

i = 0
wind_speed_freq_MHS_diff = [0] * len(wind_speed_freq_79_97_MHS)
while i < len(wind_speed_freq_79_97_MHS):
    wind_speed_freq_MHS_diff[i] = wind_speed_freq_98_16_MHS[i] - wind_speed_freq_79_97_MHS[i]
    i = i + 1

i = 0
wind_speed_freq_LHS_diff = [0] * len(wind_speed_freq_79_97_LHS)
while i < len(wind_speed_freq_79_97_LHS):
    wind_speed_freq_LHS_diff[i] = wind_speed_freq_98_16_LHS[i] - wind_speed_freq_79_97_LHS[i]
    i = i + 1

# Create histograms that show the difference between the wind speed frequencies for
# the two time periods for each part of the hurricane season
plt.figure(1, figsize = (20,10))
plt.bar(np.arange(len(wind_speed_freq_EHS_diff)), wind_speed_freq_EHS_diff)
plt.xlim(xmax=51)
plt.ylabel('Frequency Difference')
plt.xlabel('Wind Speed (m/s)')
plt.title('Difference Between Frequency of Wind Speeds Recorded in Late May, June, and ' +
          'July During Hurricane Seasons\nfrom 1979 up to 1998 and During Hurricane ' +
          'Seasons from 1998 up to 2017')
plt.savefig('Figures/Wind_Speed_Diff_Between_78-97_and_98-16_EHS_All_Locations_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,10))
plt.bar(np.arange(len(wind_speed_freq_MHS_diff)), wind_speed_freq_MHS_diff)
plt.xlim(xmax=51)
plt.ylabel('Frequency Difference')
plt.xlabel('Wind Speed (m/s)')
plt.title('Difference Between Frequency of Wind Speeds Recorded in August and September ' +
          'During Hurricane Seasons\nfrom 1979 up to 1998 and During Hurricane ' +
          'Seasons from 1998 up to 2017')
plt.savefig('Figures/Wind_Speed_Diff_Between_78-97_and_98-16_MHS_All_Locations_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,10))
plt.bar(np.arange(len(wind_speed_freq_LHS_diff)), wind_speed_freq_LHS_diff)
plt.xlim(xmax=51)
plt.ylabel('Frequency Difference')
plt.xlabel('Wind Speed (m/s)')
plt.title('Difference Between Frequency of Wind Speeds Recorded in October and November ' +
          'During Hurricane Seasons\nfrom 1979 up to 1998 and During Hurricane ' +
          'Seasons from 1998 up to 2017')
plt.savefig('Figures/Wind_Speed_Diff_Between_78-97_and_98-16_LHS_All_Locations_Histogram.png')
plt.show()


######################################################################################
##                                 Print Statistics                                 ##
######################################################################################

# Create a map showing the points that the data was obtained from
plt.figure(1, figsize = (10,10))
m = Basemap(llcrnrlon = -100, llcrnrlat = 20, urcrnrlon = -60, urcrnrlat = 50,
            projection = 'tmerc', resolution ='i', lon_0 = -80, lat_0 = 35)
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary(fill_color='#99ffff')
# m.fillcontinents(color='#cc9966',lake_color='#99ffff')
m.drawparallels(np.arange(20,50,10),labels=[1,1,0,0])
m.drawmeridians(np.arange(-100,-60,10),labels=[0,0,0,1])

i = 0
for loc in locs:
    m.scatter(loc[0], loc[1], 8, marker = 'o', color = 'r', latlon = True)

plt.title('Wind Speed Measurement Locations')
plt.savefig('Figures/Wind_Speed_Measurement_Locations.png')
plt.show()

# Print some notable statistics about the data
print('\nTotal Number of Stagnant Flow Measurements Observed at All Locations from 1979 ' +
      'up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed at All Locations from 1979 ' +
      'up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed at All Locations from 1998 ' +
      'up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16, stagnant_flow)))
print('Mean Observed Wind Speed Among All Locations from 1979 up to 2017: ' + 
      str(np.mean(wind_speeds_79_16)) + ' m/s')
print('Mean Observed Wind Speed Among All Locations from 1979 up to 1998: ' +
      str(np.mean(wind_speeds_79_97)) + ' m/s')
print('Mean Observed Wind Speed Among All Locations from 1998 up to 2017: ' +
      str(np.mean(wind_speeds_98_16)) + ' m/s')
print('Standard Deviation of Observed Wind Speed Among All Locations from 1979 up to ' +
      '2017: ' + str(np.std(wind_speeds_79_16)) + ' m/s')
print('Standard Deviation of Observed Wind Speed Among All Locations from 1979 up to ' +
      '1998: ' + str(np.std(wind_speeds_79_97)) + ' m/s')
print('Standard Deviation of Observed Wind Speed Among All Locations from 1998 up to ' +
      '2017: ' + str(np.std(wind_speeds_98_16)) + ' m/s\n')

print('Total Number of Stagnant Flow Measurements Observed in Late May, June, and July ' +
      'at All Locations from 1979 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all_EHS, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in Late May, June, and July ' +
      'at All Locations from 1979 up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97_EHS, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in Late May, June, and July ' +
      'at All Locations from 1998 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16_EHS, stagnant_flow)))
print('Mean Observed Wind Speed Among All Locations in Late May, June, and July from 1979 up to 2017: ' +
      str(np.mean(wind_speeds_79_16_EHS)) + ' m/s')
print('Mean Observed Wind Speed Among All Locations in Late May, June, and July from 1979 up to 1998: ' +
      str(np.mean(wind_speeds_79_97_EHS)) + ' m/s')
print('Mean Observed Wind Speed Among All Locations in Late May, June, and July from 1998 up to 2017: ' +
      str(np.mean(wind_speeds_98_16_EHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed Among All Locations in Late May, June, and July ' +
      'from 1979 up to 2017: ' + str(np.std(wind_speeds_79_16_EHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed Among All Locations in Late May, June, and July ' +
      'from 1979 up to 1998: ' + str(np.std(wind_speeds_79_97_EHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed Among All Locations in Late May, June, and July ' +
      'from 1998 up to 2017: ' + str(np.std(wind_speeds_98_16_EHS)) + ' m/s\n')

print('Total Number of Stagnant Flow Measurements Observed in August and September ' +
      'at All Locations from 1979 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all_MHS, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in August and September ' +
      'at All Locations from 1979 up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97_MHS, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in August and September ' +
      'at All Locations from 1998 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16_MHS, stagnant_flow)))
print('Mean Observed Wind Speed Among All Locations in August and September from 1979 up to 2017: ' +
      str(np.mean(wind_speeds_79_16_MHS)) + ' m/s')
print('Mean Observed Wind Speed Among All Locations in August and September from 1979 up to 1998: ' +
      str(np.mean(wind_speeds_79_97_MHS)) + ' m/s')
print('Mean Observed Wind Speed Among All Locations in August and September from 1998 up to 2017: ' +
      str(np.mean(wind_speeds_98_16_MHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed Among All Locations in August and September ' +
      'from 1979 up to 2017: ' + str(np.std(wind_speeds_79_16_MHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed Among All Locations in August and September ' +
      'from 1979 up to 1998: ' + str(np.std(wind_speeds_79_97_MHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed Among All Locations in August and September ' +
      'from 1998 up to 2017: ' + str(np.std(wind_speeds_98_16_MHS)) + ' m/s\n')

print('Total Number of Stagnant Flow Measurements Observed in October and November ' +
      'at All Locations from 1979 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all_LHS, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in October and November ' +
      'at All Locations from 1979 up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97_LHS, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in October and November ' +
      'at All Locations from 1998 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16_LHS, stagnant_flow)))
print('Mean Observed Wind Speed Among All Locations in October and November from 1979 up to 2017: ' +
      str(np.mean(wind_speeds_79_16_LHS)) + ' m/s')
print('Mean Observed Wind Speed Among All Locations in October and November from 1979 up to 1998: ' +
      str(np.mean(wind_speeds_79_97_LHS)) + ' m/s')
print('Mean Observed Wind Speed Among All Locations in October and November from 1998 up to 2017: ' +
      str(np.mean(wind_speeds_98_16_LHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed Among All Locations in October and November ' +
      'from 1979 up to 2017: ' + str(np.std(wind_speeds_79_16_LHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed Among All Locations in October and November ' +
      'from 1979 up to 1998: ' + str(np.std(wind_speeds_79_97_LHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed Among All Locations in October and November ' +
      'from 1998 up to 2017: ' + str(np.std(wind_speeds_98_16_LHS)) + ' m/s\n')
