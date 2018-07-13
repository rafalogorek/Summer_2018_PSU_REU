######################################################################################
#                                                                                    #
# Deep Layer Mean Steering Flow Statistics Calculator                                #
#                                                                                    #
# Author: Rafal Ogorek                                                               #
#                                                                                    #
# Description: This program makes use of reanalysis data to calculate various        #
#              statistics for the deep layer mean steering flow winds along the      #
#              coastline of the southeastern U.S. from 1979 to 2017. Several         #
#              graphics are also generated to help visualize this data better.       #
#                                                                                    #
# Last updated: 7/13/2018                                                            #
#                                                                                    #
######################################################################################

import math
import sys
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
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
#                          1 m/s and 2 m/s; etc...). When passed into the function,
#                          all entries should be 0 since no frequencies have been
#                          calculated yet.
# Output: Nothing, but the wind_speed_freq array should have been modified to now
#         contain the correct frequency value for each wind speed range
def getFrequencies(wind_speeds, wind_speed_freq):
    for loc in wind_speeds:
        for wind_speed in loc:
            # Make the wind speed a positive value (if necessary)  and round it down to
            # the nearest integer. Then, increment the frequency of the wind speed range
            # that this value falls in.
            if not np.isnan(wind_speed):
                current_ws = int(math.floor(abs(wind_speed)))
                if current_ws < len(wind_speed_freq):
                    wind_speed_freq[current_ws] = wind_speed_freq[current_ws] + 1
                else:
                    # Optionally, disregard high wind speed measurements
                    current_ws = current_ws


# Description: Obtains the wind speeds recorded at all locations within a specified
#              time interval
# Input: -wind_speeds: An array that stores an array of floats. Each array within
#                      this array represents a different location, while each float
#                      is a wind speed measurement in m/s
#        -left_bound: The index that represents the beginning of the desired time interval
#                     (included in the time interval that is returned)
#        -right_bound: The index that represents the end of the desired time interval (not
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


# Description: Splits the wind speed data into five different arrays, based on the
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
#         -wind_speeds_June: Same as the above arrays, except all wind speed measurements
#                            were taken in June (plus the last 2 days of May)
#         -wind_speeds_July: Same as the above arrays, except all wind speed measurements
#                            were taken in July
#         -wind_speeds_Aug: Same as the above arrays, except all wind speed measurements
#                           were taken in August
#         -wind_speeds_Sep: Same as the above arrays, except all wind speed measurements
#                           were taken in September
#         -wind_speeds_Oct: Same as the above arrays, except all wind speed measurements
#                           were taken in October
#         -wind_speeds_Nov: Same as the above arrays, except all wind speed measurements
#                           were taken in November
def divideBySeason(wind_speeds):
    wind_speeds_EHS = [[None]] * len(wind_speeds)
    wind_speeds_MHS = [[None]] * len(wind_speeds)
    wind_speeds_LHS = [[None]] * len(wind_speeds)
    wind_speeds_June = [[None]] * len(wind_speeds)
    wind_speeds_July = [[None]] * len(wind_speeds)
    wind_speeds_Aug = [[None]] * len(wind_speeds)
    wind_speeds_Sep = [[None]] * len(wind_speeds)
    wind_speeds_Oct = [[None]] * len(wind_speeds)
    wind_speeds_Nov = [[None]] * len(wind_speeds)
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

                # Also add the measurement to the June or July array
                # Late May/June case
                if (j % 732) < 128:
                    if wind_speeds_June[i] == [None]:
                        wind_speeds_June[i] = [wind_speeds[i][j]]
                    else:
                        wind_speeds_June[i].append(wind_speeds[i][j])
                # July case
                else:
                    if wind_speeds_July[i] == [None]:
                        wind_speeds_July[i] = [wind_speeds[i][j]]
                    else:
                        wind_speeds_July[i].append(wind_speeds[i][j])

            # If data was obtained in October or later, add it to the late hurricane season
            # wind speed array
            elif (j % 732) >= 496:
                if wind_speeds_LHS[i] == [None]:
                    wind_speeds_LHS[i] = [wind_speeds[i][j]]
                else:
                    wind_speeds_LHS[i].append(wind_speeds[i][j])

                # Also add the measurement to the October or November array
                # October case
                if (j % 732) < 620:
                    if wind_speeds_Oct[i] == [None]:
                        wind_speeds_Oct[i] = [wind_speeds[i][j]]
                    else:
                        wind_speeds_Oct[i].append(wind_speeds[i][j])
                # November case
                else:
                    if wind_speeds_Nov[i] == [None]:
                        wind_speeds_Nov[i] = [wind_speeds[i][j]]
                    else:
                        wind_speeds_Nov[i].append(wind_speeds[i][j])

            # Otherwise, the wind speed measurement was take in August or September, so
            # add it to the mid hurricane season wind speed array
            else:
                if wind_speeds_MHS[i] == [None]:
                    wind_speeds_MHS[i] = [wind_speeds[i][j]]
                else:
                    wind_speeds_MHS[i].append(wind_speeds[i][j])

                # Also add the measurement to the August or September array
                # August case
                if (j % 732) < 376:
                    if wind_speeds_Aug[i] == [None]:
                        wind_speeds_Aug[i] = [wind_speeds[i][j]]
                    else:
                        wind_speeds_Aug[i].append(wind_speeds[i][j])
                # September case
                else:
                    if wind_speeds_Sep[i] == [None]:
                        wind_speeds_Sep[i] = [wind_speeds[i][j]]
                    else:
                        wind_speeds_Sep[i].append(wind_speeds[i][j])

            j = j + 1

        i = i + 1

    return wind_speeds_EHS, wind_speeds_MHS, wind_speeds_LHS, wind_speeds_June, wind_speeds_July, wind_speeds_Aug, wind_speeds_Sep, wind_speeds_Oct, wind_speeds_Nov


# Description: Averages wind speeds over a specified time interval.
# Input: -wind_speeds: An array that stores an array of floats. Each array within
#                      this array represents a different location, while each float
#                      is a wind speed measurement in m/s
#        -dates_and_times: An array that stores Datetime objects. Each index in the array
#                          corresponds to the time that a measurement was taken at
#        -time_interval: An integer representing the period of time the winds should be
#                        averaged over. A time interval of 1 would correspond to averaging
#                        over every 6 hours (this would just be the time period between
#                        measurements), a time interval of 2 would correspond to averaging
#                        over every 12 hours, etc. The time interval needs to be able to
#                        divide the length of the wind speed array in order for this
#                        function to work properly
# Output: -avg_wind_speeds: An array that stores an array of floats. It includes
#                           all locations, but should be a condensed version of
#                           wind_speeds where it stores the average wind speeds based
#                           on the specified time interval
#         -new_dates_and_times: An array that stores Datetime objects. Each index in the
#                               array now represents the beginning time of when the
#                               averaged wind speeds were measured
def averageWindsOverTime_v1(wind_speeds, dates_and_times, time_interval):
    # Set the size of the average wind speed array
    avg_wind_speeds = [[None]] * len(wind_speeds)

    # For each location, average over the specified time interval
    i = 0
    while i < len(wind_speeds):
        j = 0
        while j < len(wind_speeds[i]):
            if avg_wind_speeds[i] == [None]:
                avg_wind_speeds[i] = [np.sum(wind_speeds[i][j:(j+time_interval)])/time_interval]
            else:
                avg_wind_speeds[i].append(np.sum(wind_speeds[i][j:(j+time_interval)])/time_interval)
            j = j + time_interval
        i = i + 1

    # Update the dates_and_times array to account for the newly averaged wind speeds
    new_dates_and_times = []
    i = 0
    while i < len(wind_speeds[0]):
        new_dates_and_times.append(dates_and_times[i])
        i = i + time_interval

    return avg_wind_speeds, new_dates_and_times


# Description: Averages wind speeds over a specified time interval using a moving average
#              filter.
# Input: -wind_speeds: An array that stores an array of floats. Each array within
#                      this array represents a different location, while each float
#                      is a wind speed measurement in m/s
#        -dates_and_times: An array that stores Datetime objects. Each index in the array
#                          corresponds to the time that a measurement was taken at
#        -time_interval: An integer representing the period of time the winds should be
#                        averaged over. A time interval of 1 would correspond to averaging
#                        over every 6 hours (this would just be the time period between
#                        measurements), a time interval of 2 would correspond to averaging
#                        over every 12 hours, etc. The time interval needs to be able to
#                        divide the length of the wind speed array in order for this
#                        function to work properly
# Output: -avg_wind_speeds: An array that stores an array of floats. It includes
#                           all locations, but should be a condensed version of
#                           wind_speeds where it stores the average wind speeds based
#                           on the specified time interval
#         -new_dates_and_times: An array that stores Datetime objects. Each index in the
#                               array now represents the beginning time of when the
#                               averaged wind speeds were measured
def averageWindsOverTime_v2(wind_speeds, dates_and_times, time_interval):
    # Define arrays
    avg_wind_speeds = [[None]] * len(wind_speeds)
    new_dates_and_times = []

    # For each location, average over the specified time interval
    # Also get corresponding days/times
    i = 0
    while i < len(wind_speeds):
        j = time_interval / 2
        while j < (len(wind_speeds[i]) - (time_interval / 2) + 1):

            # Only add times during the first loop
            if i == 0:
                new_dates_and_times.append(dates_and_times[j])

            if avg_wind_speeds[i] == [None]:
                avg_wind_speeds[i] = [np.sum(wind_speeds[i][(j - (time_interval / 2)):(j + (time_interval / 2))])/time_interval]
            else:
                avg_wind_speeds[i].append(np.sum(wind_speeds[i][(j - (time_interval / 2)):(j + (time_interval / 2))])/time_interval)

            if (j % 732) != 732 - (time_interval / 2):
                j = j + 1
            else:
                j = j + time_interval

        i = i + 1

    return avg_wind_speeds, new_dates_and_times


# Description: Averages wind speeds for each time of the hurricane season between
#              all given hurricane seasons and all locations.
# Input: -wind_speeds: An array that stores an array of floats. Each array within
#                      this array represents a different location, while each float
#                      is a wind speed measurement in m/s
#        -dates_and_times: An array that stores Datetime objects. Each index in the array
#                          corresponds to the time that a measurement was taken at
#        -time_interval: An integer representing the period of time the winds should be
#                        averaged over. A time interval of 1 would correspond to averaging
#                        over every 6 hours (this would just be the time period between
#                        measurements), a time interval of 2 would correspond to averaging
#                        over every 12 hours, etc. The time interval needs to be able to
#                        divide the length of the wind speed array in order for this
#                        function to work properly
# Output: -avg_wind_speeds: An array of floats, where each float is the average wind
#                           speed between all locations and all hurricane seasons
#                           at the current time
def averageWindsEachTime(wind_speeds, dates_and_times, time_interval):
    avg_wind_speeds = []
    new_dates_and_times = []

    i = 0
    # Go through each time where measurements were taken during hurricane seasons
    while i < 732:
        j = i
        sum_wind_speeds = 0
        count = 0

        # Go through all hurricane seasons
        while j < len(wind_speeds[0]):
            k = 0

            # Go through all locations
            while k < len(wind_speeds):

                if not np.isnan(wind_speeds[k][j]):
                    # Sum the wind speeds among all locations
                    sum_wind_speeds = sum_wind_speeds + wind_speeds[k][j]
                    count = count + 1

                k = k + 1

            j = j + 732

        # Average the wind speed for the current time
        if count != 0:
            avg_wind_speeds.append(sum_wind_speeds/count)
        else:
            # If all measurements were removed, set the average to NaN
            avg_wind_speeds.append(float('nan'))

        i = i + 1

    #new_dates_and_times = dates_and_times
    # Optionally, also perform a low pass filter averaging method to smoothen the curve
    temp_wind_speeds = avg_wind_speeds
    avg_wind_speeds = []
    i = time_interval / 2
    while i < (len(temp_wind_speeds) - (time_interval / 2) + 1):
        # Update times
        new_dates_and_times.append(dates_and_times[i])

        # Average wind speeds
        avg_wind_speeds.append(np.sum(temp_wind_speeds[(i - (time_interval / 2)):(i + (time_interval / 2))])/time_interval)

        i = i + 1

    return avg_wind_speeds, new_dates_and_times


# Description: Averages wind speeds between all locations for the whole time period
# Input: -wind_speeds: An array that stores an array of floats. Each array within
#                      this array represents a different location, while each float
#                      is a wind speed measurement in m/s
# Output: -avg_wind_speeds: An array of floats, where each float is the average wind
#                           speed between all locations for the corresponding time
def averageWindsAmongAllPoints(wind_speeds):
    # Define the average wind speed array
    avg_wind_speeds = []

    # For each time, average the wind speed between all locations
    i = 0
    while i < len(wind_speeds[0]):
        # Reset the sum of the wind speeds and location count
        sum_wind_speeds = 0
        count = 0

        for loc in wind_speeds:
            if not np.isnan(loc[i]):
                # Sum the wind speeds among all locations
                sum_wind_speeds = sum_wind_speeds + loc[i]
                count = count + 1

        # Average the wind speed for the current time
        if count != 0:
            avg_wind_speeds.append(sum_wind_speeds/count)
        else:
            # If all measurements were removed, set the average to NaN
            avg_wind_speeds.append(float('nan'))

        i = i + 1

    return avg_wind_speeds


# Description: Normalizes wind speed frequency counts based on the total frequency count
# Input: -wind_speed_freq: An array that stores frequency counts of certain wind speeds
# Output: -norm_wind_speed_freq: An array that stores the normalized percentages of
#                                wind speed frequencies
def normalizeWindSpeeds(wind_speed_freq):
    norm_wind_speed_freq = []

    # Total count of wind speed frequencies
    sum_freq = np.sum(np.absolute(wind_speed_freq))

    # Normalize each wind speed bucket
    for freq in wind_speed_freq:
        norm_wind_speed_freq.append(float(freq)/float(sum_freq))

    return norm_wind_speed_freq


# Description: Filters out wind speeds taken during tropical cyclone landfalls or
#              near-landfalls
# Input: -wind_speeds: An array that stores an array of floats. Each array within
#                      this array represents a different location, while each float
#                      is a wind speed measurement in m/s
#        -dates_and_times: An array that stores Datetime objects. Each index in the array
#                          corresponds to the time that a measurement was taken at
#        -locs: An array storing the coordinates of the locations where wind speed
#               measurements were taken
#        -times_to_remove: An array of Datetime objects that indicates the times when
#                          a tropical cyclone may be affecting the DLM wind speeds
#        -locs_to_remove: An array containing the locations that were impacted by
#                         the tropical cyclone at the corresponding index in the
#                         times_to_remove array
# Output: -new_wind_speeds: Updated version of the wind_speeds array, with any wind
#                           speeds measured during ongoing tropical cyclones set to
#                           NaN
def removeTCWinds(wind_speeds, dates_and_times, locs, times_to_remove, locs_to_remove):
    new_wind_speeds = wind_speeds

    # Loop through times and look for times that have measurements that need to be removed
    i = 0
    while i < len(times_to_remove):
        j = 0

        # Remove measurements for all locations that need to have them removed at the
        # current time
        while j < len(locs_to_remove[i]):
            # Determine the proper location index
            k = 0
            while k < len(locs):
                if np.all(locs_to_remove[i][j] == locs[k]):
                    break

                k = k + 1

            # Remove the measurement by setting it to NaN
            new_wind_speeds[k][dates_and_times.index(times_to_remove[i])] = float('nan')
            j = j + 1

        i = i + 1

    return new_wind_speeds


# Description: Reads in HURDAT2 best track data and populates arrays with times and
#              locations of tropical cyclones
# Input: -filename: A string containing the name of the text file that contains the
#                   hurricane best track data
#        -locs: An array storing the coordinates of the locations where wind speed
#               measurements were taken
# Output: -times_to_remove: An array of Datetime objects that indicates the times when
#                           a tropical cyclone may be affecting the DLM wind speeds
#         -locs_to_remove: An array containing arrays of locations that were impacted
#                          by the tropical cyclone at the corresponding index in the
#                          times_to_remove array
def readBestTracks(filename, locs):
    times_to_remove = []
    locs_to_remove = []

    # Read file
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    # Parse data line by line
    for line in lines:
        # Data had to have been taken between May 30 and November 28
        if (((int(line[4:6]) > 5) and (int(line[4:6]) < 11)) or (((int(line[4:6]) == 5) and \
           (int(line[6:8]) >= 30))) or (((int(line[4:6]) == 11) and (int(line[6:8]) <= 28)))):

           # Only take times at 0000, 0600, 1200, and 1800
           if (line[10:14] == '0000') or (line[10:14] == '0600') or (line[10:14] == '1200') or (line[10:14] == '1800'):

               # Only consider tropical storms and hurricanes
               if (line[19:21] == 'TS') or (line[19:21] == 'HU'):

                   # Remove any tropical cyclones that were too far away from the region
                   # of interest
                   if (float(line[23:27]) < 38) and (float(line[23:27]) > 20) and (float(line[30:35]) < 105) and (float(line[30:35]) > 72):
                       # The radius (in degrees) used to determine how far away from the
                       # eye of the tropical cyclone should data points be removed
                       if line[19:21] == 'HU':
                           deg_radius = max(60 * 8, float(line[49:53]), float(line[55:59]), float(line[61:65]), float(line[67:71])) / 60
                       else:
                           deg_radius = max(60 * 6, float(line[49:53]), float(line[55:59]), float(line[61:65]), float(line[67:71])) / 60

                       # Get eye location
                       eye_loc = [float(line[30:35]) * -1, float(line[23:27])]

                       # Go through the location array and see which locations fall in
                       # the defined radius
                       in_range = 0
                       locs_in_range = []
                       for loc in locs:
                           # Add any locations seeing at least tropical storm force winds
                           # at the current time to an array
                           if (abs(loc[0] - eye_loc[0]) <= deg_radius) and (abs(loc[1] - eye_loc[1]) <= deg_radius):
                               in_range = 1
                               locs_in_range.append(loc)

                       # Indicate that there are locations that need to be removed at
                       # this time
                       if in_range != 0:
                           # Add the time that needs to be removed
                           times_to_remove.append(datetime.strptime(line[0:14], '%Y%m%d, %H%M'))

                           # Add the locations that need to be removed
                           locs_to_remove.append(locs_in_range)

    return times_to_remove, locs_to_remove


######################################################################################
##                                   Begin Program                                  ##
######################################################################################

# Currently defining stagnant steering flow to be 2 m/s
stagnant_flow = 2

# Regional abbreviation
region = sys.argv[2]

# Dictionary to match abbreviations to names
location_names = {}
location_names['AL'] = ['All_Locations', 'Southeast U.S. Coasts', 'All Locations']
location_names['AC'] = ['Atlantic_Coast', 'the Southeast U.S. Atlantic Coast',
                        'Locations Along the Southeast U.S. Atlantic Coast']
location_names['GOM'] = ['Gulf_of_Mexico_Coast', 'the Gulf of Mexico Coast',
                         'Locations Along the Gulf of Mexico Coast']
location_names['TX'] = ['TX_Coast', 'the Texas Coast', 'Locations Along the Texas Coast']
location_names['LA-MS'] = ['LA-MS_Coast', 'Louisiana and Mississippi Coasts',
                           'Locations Along the Louisiana and Mississippi Coasts']
location_names['AL-FL'] = ['AL-FL_Pan_Coast', 'Alabama and Florida Panhandle Coasts',
                           'Locations Along the Alabama and Florida Panhandle Coasts']
location_names['WFL'] = ['West_FL_Coast', 'the West Florida Coast', 'Locations Along the West Florida Coast']
location_names['EFL'] = ['East_FL_Coast', 'the East Florida Coast', 'Locations Along the East Florida Coast']
location_names['GA-SC'] = ['GA-SC_Coast', 'Georgia and South Carolina Coasts',
                           'Locations Along the Georgia and South Carolina Coasts']
location_names['NC'] = ['NC_Coast', 'the North Carolina Coast', 'Locations Along the North Carolina Coast']

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
temp_wind_speeds = []
i = 0
while i < len(locs):
    # Get rid of points north or east of North Carolina, west of Corpus Christi, and
    # south of Key West
    if locs[i][0] > -98 and locs[i][0] <= -75 and locs[i][1] > 24 and locs[i][1] <= 36:
        # Also get rid of points along Mexican Coast and around The Bahamas. Removed North Carolina points as well
        if not (locs[i][0] < -96 and locs[i][1] < 26) and not (locs[i][0] > -79 and locs[i][1] < 30) and not (locs[i][1] > 33.75 or (locs[i][1] == 33.75 and locs[i][0] >= -78)):
            # Cases for each specified region
            if region == 'AC':
                # Atlantic coast case
                if locs[i][0] >= -80.25 or (locs[i][0] == -81.75 and locs[i][1] > 28.5) or (locs[i][0] == -81 and locs[i][1] > 26.25):
                    temp_locs.append(locs[i])
                    temp_wind_speeds.append(wind_speeds[i])
            elif region == 'GOM':
                # Gulf of Mexico coast case
                if locs[i][0] <= -82.5 or (locs[i][0] == -81.75 and locs[i][1] <= 28.5) or (locs[i][0] == -81 and locs[i][1] <= 26.25):
                    temp_locs.append(locs[i])
                    temp_wind_speeds.append(wind_speeds[i])
            elif region == 'TX':
                # Texas coast case
                if locs[i][0] <= -94.5:
                    temp_locs.append(locs[i])
                    temp_wind_speeds.append(wind_speeds[i])
            elif region == 'LA-MS':
                # Lousiana/Mississippi coast case
                if locs[i][0] > -94.5 and locs[i][0] <= -88.5:
                    temp_locs.append(locs[i])
                    temp_wind_speeds.append(wind_speeds[i])
            elif region == 'AL-FL':
                # Alabama/Florida Panhandle coast case
                if locs[i][0] > -88.5 and locs[i][0] <= -84:
                    temp_locs.append(locs[i])
                    temp_wind_speeds.append(wind_speeds[i])
            elif region == 'WFL':
                # West Florida coast case
                if (locs[i][0] > -84 and locs[i][0] <= -82.5) or (locs[i][0] == -81.75 and locs[i][1] <= 28.5) or (locs[i][0] == -81 and locs[i][1] <= 26.25):
                    temp_locs.append(locs[i])
                    temp_wind_speeds.append(wind_speeds[i])
            elif region == 'EFL':
                # East Florida coast case
                if (locs[i][0] == -81.75 and locs[i][1] > 28.5 and locs[i][1] <= 30.75) or (locs[i][0] == -81 and locs[i][1] > 26.25 and locs[i][1] <= 30.75) or \
                   (locs[i][0] == -80.25 and locs[i][1] <= 30.75) or (locs[i][0] == -79.5 and locs[i][1] <= 30.75):
                    temp_locs.append(locs[i])
                    temp_wind_speeds.append(wind_speeds[i])
            elif region == 'GA-SC':
                # Georgia/South Carolina coast case
                if (locs[i][0] == -81.75 and locs[i][1] > 30.75) or (locs[i][0] == -81 and locs[i][1] > 30.75) or (locs[i][0] == -80.25 and locs[i][1] > 30.75) or \
                   (locs[i][0] == -79.5 and locs[i][1] > 30.75) or (locs[i][0] == -78.75 and locs[i][1] <= 33.75) or (locs[i][0] == -78 and locs[i][1] == 33):
                    temp_locs.append(locs[i])
                    temp_wind_speeds.append(wind_speeds[i])
            #elif region == 'NC':
            #    # North Carolina coast case
            #    if locs[i][1] > 33.75 or (locs[i][1] == 33.75 and locs[i][0] >= -78):
            #        temp_locs.append(locs[i])
            #        temp_wind_speeds.append(wind_speeds[i])
            else:
                # General case for all locations
                temp_locs.append(locs[i])
                temp_wind_speeds.append(wind_speeds[i])

    i = i + 1

locs = temp_locs

# Ensures wind direction doesn't matter by making any negative wind speeds positive
# (All wind speeds should already be positive though)
wind_speeds = map(abs, temp_wind_speeds)

# Determine the maximum wind speed recorded
max_wind_speed = int(math.ceil(np.amax(wind_speeds)))
# Or set a predefined max to disregard larger wind speed recordings
# Comment this line out if you don't want any maximum limit on what values to plot
max_wind_speed = 45

# Read in hurricane best track data
times_to_remove, locs_to_remove = readBestTracks('best_tracks.txt', locs)

# Remove DLM winds contaminated with tropical cyclone winds
wind_speeds = removeTCWinds(wind_speeds, dates_and_times, locs, times_to_remove, locs_to_remove)

######################################################################################
##                         All Locations, Whole Time Period                         ##
######################################################################################

# Analyze all data together, regardless of time or location

# Create an array to store how frequently a wind speed value was recorded (at all
# locations). The zeroeth index of the array will store the frequency of recorded wind
# speeds between 0 m/s and 1 m/s; the first index of the array will store the frequency
# of wind speeds between 1 m/s and 2 m/s; etc...
wind_speed_freq_all = [0] * max_wind_speed

# Create wind speed array that will only contain data for the desired period of time
wind_speeds_79_16 = getWindSpeedInterval(wind_speeds, 0, 27816)

# Get updated time interval as well
dates_and_times_79_16 = dates_and_times[0:27816]

# Populate the wind speed frequency array
getFrequencies(wind_speeds_79_16, wind_speed_freq_all)

current_year = 2005
current_year_index = current_year - 1979

# Average the wind speed between all locations at each time
avg_wind_speeds_79_16 = averageWindsAmongAllPoints(wind_speeds_79_16)

# Plot averaged wind speeds over the course of a year
plt.figure(1, figsize = (20,10))
plt.rc('axes', titlesize = 25)
plt.rc('axes', labelsize = 20)
plt.rc('xtick', labelsize = 15)
plt.rc('ytick', labelsize = 15)
plt.plot(dates_and_times_79_16[(current_year_index * 732):(732 * (current_year_index + 1))], avg_wind_speeds_79_16[(current_year_index * 732):(732 * (current_year_index + 1))])
plt.ylabel('Wind Speed (m/s)')
plt.xlabel('Time')
plt.title('Average Wind Speeds During the ' + str(current_year) + ' Hurricane Season on ' +
          location_names[region][1])
#plt.savefig('Figures/Time_Series/' + str(current_year) + '/Filtered_' + location_names[region][0]  + '_Time_Series_' + str(current_year) + '.png')
#plt.savefig('Figures/Time_Series/Yearly/Filtered_' + location_names[region][0]  + '_Time_Series_' + str(current_year) + '.png')
plt.show()

# Normalize wind speed frequencies
norm_wind_speed_freq_all = normalizeWindSpeeds(wind_speed_freq_all)

# Generate a histogram to show the frequency distribution for the wind speeds at all
# measured locations from 1979 to 2017
plt.figure(1, figsize = (20,10))
#plt.bar(np.arange(len(wind_speed_freq_all)), wind_speed_freq_all)
plt.bar(np.arange(len(norm_wind_speed_freq_all)), norm_wind_speed_freq_all)
plt.xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    plt.ylim(ymax = 500000)
plt.ylabel('Frequency (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nDuring ' +
          'Hurricane Seasons from 1979 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_Histogram.png')
plt.savefig('Figures/WS_79-16/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_Histogram.png')
plt.show()

# Create arrays to store how frequently a wind speed value was recorded (at all
# locations). The zeroeth index of the array will store the frequency of recorded wind
# speeds between 0 m/s and 1 m/s; the first index of the array will store the frequency
# of wind speeds between 1 m/s and 2 m/s; etc...
wind_speed_freq_all_EHS = [0] * max_wind_speed
wind_speed_freq_all_MHS = [0] * max_wind_speed
wind_speed_freq_all_LHS = [0] * max_wind_speed
wind_speed_freq_all_June = [0] * max_wind_speed
wind_speed_freq_all_July = [0] * max_wind_speed
wind_speed_freq_all_Aug = [0] * max_wind_speed
wind_speed_freq_all_Sep = [0] * max_wind_speed
wind_speed_freq_all_Oct = [0] * max_wind_speed
wind_speed_freq_all_Nov = [0] * max_wind_speed

# Divide the wind speed data up by which part of the hurricane season they were taken in
# (early hurricane season (June, July), mid  hurricane season (August, September), and
# late hurricane season (October, November)). Also get data for each month
# individually
wind_speeds_79_16_EHS, wind_speeds_79_16_MHS, wind_speeds_79_16_LHS, wind_speeds_79_16_June, wind_speeds_79_16_July, \
wind_speeds_79_16_Aug, wind_speeds_79_16_Sep, wind_speeds_79_16_Oct, wind_speeds_79_16_Nov = divideBySeason(wind_speeds_79_16)

# Populate the wind speed frequency arrays for each of the three parts of the hurricane season
# (and each month)
getFrequencies(wind_speeds_79_16_EHS, wind_speed_freq_all_EHS)
getFrequencies(wind_speeds_79_16_MHS, wind_speed_freq_all_MHS)
getFrequencies(wind_speeds_79_16_LHS, wind_speed_freq_all_LHS)
getFrequencies(wind_speeds_79_16_June, wind_speed_freq_all_June)
getFrequencies(wind_speeds_79_16_July, wind_speed_freq_all_July)
getFrequencies(wind_speeds_79_16_Aug, wind_speed_freq_all_Aug)
getFrequencies(wind_speeds_79_16_Sep, wind_speed_freq_all_Sep)
getFrequencies(wind_speeds_79_16_Oct, wind_speed_freq_all_Oct)
getFrequencies(wind_speeds_79_16_Nov, wind_speed_freq_all_Nov)

# Normalize wind speed frequencies
norm_wind_speed_freq_all_EHS = normalizeWindSpeeds(wind_speed_freq_all_EHS)
norm_wind_speed_freq_all_MHS = normalizeWindSpeeds(wind_speed_freq_all_MHS)
norm_wind_speed_freq_all_LHS = normalizeWindSpeeds(wind_speed_freq_all_LHS)
norm_wind_speed_freq_all_June = normalizeWindSpeeds(wind_speed_freq_all_June)
norm_wind_speed_freq_all_July = normalizeWindSpeeds(wind_speed_freq_all_July)
norm_wind_speed_freq_all_Aug = normalizeWindSpeeds(wind_speed_freq_all_Aug)
norm_wind_speed_freq_all_Sep = normalizeWindSpeeds(wind_speed_freq_all_Sep)
norm_wind_speed_freq_all_Oct = normalizeWindSpeeds(wind_speed_freq_all_Oct)
norm_wind_speed_freq_all_Nov = normalizeWindSpeeds(wind_speed_freq_all_Nov)

# Generate histograms to show the frequency distribution for the wind speeds at all
# measured locations for each of the three parts of the hurricane season from 1979 to 2017
# (also each month)
plt.figure(1, figsize = (20,10))
plt.bar(np.arange(len(norm_wind_speed_freq_all_MHS)), norm_wind_speed_freq_all_EHS)
plt.xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    plt.ylim(ymax = 220000)
plt.ylabel('Frequency (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nDuring ' +
          'Late May, June, and July from 1979 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_EHS_Histogram.png')
plt.savefig('Figures/WS_79-16_EHS/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_EHS_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,10))
plt.bar(np.arange(len(norm_wind_speed_freq_all_June)), norm_wind_speed_freq_all_June)
plt.xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    plt.ylim(ymax = 220000)
plt.ylabel('Frequency (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nDuring ' +
          'the Month of June from 1979 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_June_Histogram.png')
plt.savefig('Figures/WS_79-16_June/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_June_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,10))
plt.bar(np.arange(len(norm_wind_speed_freq_all_July)), norm_wind_speed_freq_all_July)
plt.xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    plt.ylim(ymax = 220000)
plt.ylabel('Frequency (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nDuring ' +
          'the Month of July from 1979 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_July_Histogram.png')
plt.savefig('Figures/WS_79-16_July/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_July_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,10))
plt.bar(np.arange(len(norm_wind_speed_freq_all_MHS)), norm_wind_speed_freq_all_MHS)
plt.xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    plt.ylim(ymax = 220000)
plt.ylabel('Frequency (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nDuring ' +
          'August and September from 1979 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_MHS_Histogram.png')
plt.savefig('Figures/WS_79-16_MHS/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_MHS_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,10))
plt.bar(np.arange(len(norm_wind_speed_freq_all_Aug)), norm_wind_speed_freq_all_Aug)
plt.xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    plt.ylim(ymax = 220000)
plt.ylabel('Frequency (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nDuring ' +
          'the Month of August from 1979 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_Aug_Histogram.png')
plt.savefig('Figures/WS_79-16_Aug/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_Aug_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,10))
plt.bar(np.arange(len(norm_wind_speed_freq_all_Sep)), norm_wind_speed_freq_all_Sep)
plt.xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    plt.ylim(ymax = 220000)
plt.ylabel('Frequency (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nDuring ' +
          'the Month of September from 1979 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_Sep_Histogram.png')
plt.savefig('Figures/WS_79-16_Sep/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_Sep_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,10))
plt.bar(np.arange(len(norm_wind_speed_freq_all_LHS)), norm_wind_speed_freq_all_LHS)
plt.xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    plt.ylim(ymax = 220000)
plt.ylabel('Frequency (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nDuring ' +
          'October and November from 1979 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_LHS_Histogram.png')
plt.savefig('Figures/WS_79-16_LHS/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_LHS_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,10))
plt.bar(np.arange(len(norm_wind_speed_freq_all_Oct)), norm_wind_speed_freq_all_Oct)
plt.xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    plt.ylim(ymax = 220000)
plt.ylabel('Frequency (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nDuring ' +
          'the Month of October from 1979 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_Oct_Histogram.png')
plt.savefig('Figures/WS_79-16_Oct/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_Oct_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,10))
plt.bar(np.arange(len(norm_wind_speed_freq_all_Nov)), norm_wind_speed_freq_all_Nov)
plt.xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    plt.ylim(ymax = 220000)
plt.ylabel('Frequency (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nDuring ' +
          'the Month of November from 1979 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_Nov_Histogram.png')
plt.savefig('Figures/WS_79-16_Nov/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_Nov_Histogram.png')
plt.show()

######################################################################################
##                        All Locations, Halved Time Periods                        ##
######################################################################################

# Split the data into two time periods: One from 1979 up to 1998, and the other from 1998
# up to 2017

# Create arrays to store how frequently a wind speed value was recorded (at all locations
# for each time period. The zeroeth index of the array will store the frequency of recorded
# wind speeds between 0 m/s and 1 m/s; the first index of the array will store the frequency
# of wind speeds between 1 m/s and 2 m/s; etc...
wind_speed_freq_79_97 = [0] * max_wind_speed
wind_speed_freq_98_16 = [0] * max_wind_speed

# Create wind speed arrays that will only contain data for the desired period of time
wind_speeds_79_97 = getWindSpeedInterval(wind_speeds, 0, 13908)
wind_speeds_98_16 = getWindSpeedInterval(wind_speeds, 13908, 27816)

# Get updated time intervals as well
dates_and_times_79_97 = dates_and_times[0:13908]
dates_and_times_98_16 = dates_and_times[13908:27816]

# Populate the wind speed frequency arrays
getFrequencies(wind_speeds_79_97, wind_speed_freq_79_97)
getFrequencies(wind_speeds_98_16, wind_speed_freq_98_16)

# Normalize wind speed frequencies
norm_wind_speed_freq_79_97 = normalizeWindSpeeds(wind_speed_freq_79_97)
norm_wind_speed_freq_98_16 = normalizeWindSpeeds(wind_speed_freq_98_16)

# Average the wind speed at each time
avg_wind_speeds_79_97, dates_and_times_1y = averageWindsEachTime(wind_speeds_79_97, dates_and_times_79_97, 12)
avg_wind_speeds_98_16, dates_and_times_1y = averageWindsEachTime(wind_speeds_98_16, dates_and_times_79_97, 12)

# Plot the averaged wind speeds
plt.figure(1, figsize = (20,10))
plt.rc('axes', titlesize = 25)
plt.rc('axes', labelsize = 20)
plt.rc('xtick', labelsize = 15)
plt.rc('ytick', labelsize = 15)
plt.plot(dates_and_times_1y, avg_wind_speeds_79_97)
plt.plot(dates_and_times_1y, avg_wind_speeds_98_16)
plt.ylim(ymin = 0)
plt.ylabel('Wind Speed (m/s)')
plt.xlabel('Time')
plt.title('Average Wind Speeds During Hurricane Seasons from 1979 through 1997 ' +
          'Compared to\nHurricane Seasons from 1998 through 2016 on ' + location_names[region][1])
plt.savefig('Figures/Time_Series/Yearly/Filtered_' + location_names[region][0]  + '_Time_Series_3d_Avg_Comparison.png')
plt.show()

# Generate histograms to show the frequency distribution for the wind speeds at all
# measured locations from 1979 up to 1998 and from 1998 up to 2017
plt.figure(1, figsize = (20,10))
plt.subplot(211)
plt.bar(np.arange(len(norm_wind_speed_freq_79_97)), norm_wind_speed_freq_79_97)
plt.xlim(xmax = max_wind_speed + 1)
#plt.ylim(ymax = 250000)
plt.ylabel('Frequency (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nDuring ' +
          'Hurricane Seasons from 1979 up to 1998')

plt.subplot(212)
plt.bar(np.arange(len(norm_wind_speed_freq_98_16)), norm_wind_speed_freq_98_16)
plt.xlim(xmax = max_wind_speed + 1)
#plt.ylim(ymax = 250000)
plt.ylabel('Frequency (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nDuring ' +
          'Hurricane Seasons from 1998 up to 2017')

plt.subplots_adjust(left = 0.125, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.2,
                    hspace = 0.5)
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-97_98-16_Histogram.png')
plt.savefig('Figures/WS_79-97_98-16/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-97_98-16_Histogram.png')
plt.show()

# Plot the two time periods side by side, rather than on two separate plots
fig, ax = plt.subplots(figsize = (20,15))
bar_width = 0.35
ax.bar(np.arange(len(norm_wind_speed_freq_79_97)), norm_wind_speed_freq_79_97, bar_width,
                color='b', label='1979-1997')
ax.bar(np.arange(len(norm_wind_speed_freq_98_16)) + bar_width, norm_wind_speed_freq_98_16,
                bar_width, color='r', label='1998-2016')

ax.set_xlim(xmax = max_wind_speed + 1)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Frequency (%)')
ax.set_title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nDuring ' +
             'Hurricane Seasons from 1979 up to 1998\nvs. Hurricane Seasons from ' +
             '1998 up to 2017')

ax.legend()
fig.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-97_98-16_SBS_Histogram.png')
fig.savefig('Figures/WS_79-97_98-16_SBS/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-97_98-16_SBS_Histogram.png')
plt.show()

# Determine differences in the wind speed frequencies between the two time periods
i = 0
wind_speed_freq_diff = [0] * len(wind_speed_freq_79_97)
norm_wind_speed_freq_diff = [0] * len(norm_wind_speed_freq_79_97)
while i < len(wind_speed_freq_79_97):
    wind_speed_freq_diff[i] = wind_speed_freq_98_16[i] - wind_speed_freq_79_97[i]
    norm_wind_speed_freq_diff[i] = norm_wind_speed_freq_98_16[i] - norm_wind_speed_freq_79_97[i]
    i = i + 1

# Create a histogram that shows the difference between the wind speed frequencies for
# the two time periods
plt.figure(1, figsize = (20,15))
plt.bar(np.arange(len(norm_wind_speed_freq_diff)), norm_wind_speed_freq_diff)
plt.xlim(xmax = max_wind_speed + 1)
plt.ylabel('Frequency Difference (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Difference Between Normalized Frequency of Wind Speeds Recorded on\n' + location_names[region][1] +
          ' During Hurricane Seasons from 1979 up to\n1998 and During Hurricane ' +
          'Seasons from 1998 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_Histogram.png')
plt.savefig('Figures/WS_Diff_79-97_98-16/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_Histogram.png')
plt.show()


# Create arrays to store how frequently a wind speed value was recorded (at all
# locations). The zeroeth index of the array will store the frequency of recorded wind
# speeds between 0 m/s and 1 m/s; the first index of the array will store the frequency
# of wind speeds between 1 m/s and 2 m/s; etc...
wind_speed_freq_79_97_EHS = [0] * max_wind_speed
wind_speed_freq_79_97_MHS = [0] * max_wind_speed
wind_speed_freq_79_97_LHS = [0] * max_wind_speed
wind_speed_freq_79_97_June = [0] * max_wind_speed
wind_speed_freq_79_97_July = [0] * max_wind_speed
wind_speed_freq_79_97_Aug = [0] * max_wind_speed
wind_speed_freq_79_97_Sep = [0] * max_wind_speed
wind_speed_freq_79_97_Oct = [0] * max_wind_speed
wind_speed_freq_79_97_Nov = [0] * max_wind_speed
wind_speed_freq_98_16_EHS = [0] * max_wind_speed
wind_speed_freq_98_16_MHS = [0] * max_wind_speed
wind_speed_freq_98_16_LHS = [0] * max_wind_speed
wind_speed_freq_98_16_June = [0] * max_wind_speed
wind_speed_freq_98_16_July = [0] * max_wind_speed
wind_speed_freq_98_16_Aug = [0] * max_wind_speed
wind_speed_freq_98_16_Sep = [0] * max_wind_speed
wind_speed_freq_98_16_Oct = [0] * max_wind_speed
wind_speed_freq_98_16_Nov = [0] * max_wind_speed

# Divide the wind speed data up by which part of the hurricane season they were taken in
# (early hurricane season (June, July), mid hurricane season (August, September), and
# late hurricane season (October, November)). Also get data for each month individually
wind_speeds_79_97_EHS, wind_speeds_79_97_MHS, wind_speeds_79_97_LHS, wind_speeds_79_97_June, wind_speeds_79_97_July, \
wind_speeds_79_97_Aug, wind_speeds_79_97_Sep, wind_speeds_79_97_Oct, wind_speeds_79_97_Nov = divideBySeason(wind_speeds_79_97)
wind_speeds_98_16_EHS, wind_speeds_98_16_MHS, wind_speeds_98_16_LHS, wind_speeds_98_16_June, wind_speeds_98_16_July, \
wind_speeds_98_16_Aug, wind_speeds_98_16_Sep, wind_speeds_98_16_Oct, wind_speeds_98_16_Nov = divideBySeason(wind_speeds_98_16)

# Populate the wind speed frequency arrays for each of the three parts of the hurricane
# season for each time period (and each month)
getFrequencies(wind_speeds_79_97_EHS, wind_speed_freq_79_97_EHS)
getFrequencies(wind_speeds_79_97_MHS, wind_speed_freq_79_97_MHS)
getFrequencies(wind_speeds_79_97_LHS, wind_speed_freq_79_97_LHS)
getFrequencies(wind_speeds_79_97_June, wind_speed_freq_79_97_June)
getFrequencies(wind_speeds_79_97_July, wind_speed_freq_79_97_July)
getFrequencies(wind_speeds_79_97_Aug, wind_speed_freq_79_97_Aug)
getFrequencies(wind_speeds_79_97_Sep, wind_speed_freq_79_97_Sep)
getFrequencies(wind_speeds_79_97_Oct, wind_speed_freq_79_97_Oct)
getFrequencies(wind_speeds_79_97_Nov, wind_speed_freq_79_97_Nov)
getFrequencies(wind_speeds_98_16_EHS, wind_speed_freq_98_16_EHS)
getFrequencies(wind_speeds_98_16_MHS, wind_speed_freq_98_16_MHS)
getFrequencies(wind_speeds_98_16_LHS, wind_speed_freq_98_16_LHS)
getFrequencies(wind_speeds_98_16_June, wind_speed_freq_98_16_June)
getFrequencies(wind_speeds_98_16_July, wind_speed_freq_98_16_July)
getFrequencies(wind_speeds_98_16_Aug, wind_speed_freq_98_16_Aug)
getFrequencies(wind_speeds_98_16_Sep, wind_speed_freq_98_16_Sep)
getFrequencies(wind_speeds_98_16_Oct, wind_speed_freq_98_16_Oct)
getFrequencies(wind_speeds_98_16_Nov, wind_speed_freq_98_16_Nov)

# Normalize wind speed frequencies
norm_wind_speed_freq_79_97_EHS = normalizeWindSpeeds(wind_speed_freq_79_97_EHS)
norm_wind_speed_freq_98_16_EHS = normalizeWindSpeeds(wind_speed_freq_98_16_EHS)
norm_wind_speed_freq_79_97_MHS = normalizeWindSpeeds(wind_speed_freq_79_97_MHS)
norm_wind_speed_freq_98_16_MHS = normalizeWindSpeeds(wind_speed_freq_98_16_MHS)
norm_wind_speed_freq_79_97_LHS = normalizeWindSpeeds(wind_speed_freq_79_97_LHS)
norm_wind_speed_freq_98_16_LHS = normalizeWindSpeeds(wind_speed_freq_98_16_LHS)
norm_wind_speed_freq_79_97_June = normalizeWindSpeeds(wind_speed_freq_79_97_June)
norm_wind_speed_freq_98_16_June = normalizeWindSpeeds(wind_speed_freq_98_16_June)
norm_wind_speed_freq_79_97_July = normalizeWindSpeeds(wind_speed_freq_79_97_July)
norm_wind_speed_freq_98_16_July = normalizeWindSpeeds(wind_speed_freq_98_16_July)
norm_wind_speed_freq_79_97_Aug = normalizeWindSpeeds(wind_speed_freq_79_97_Aug)
norm_wind_speed_freq_98_16_Aug = normalizeWindSpeeds(wind_speed_freq_98_16_Aug)
norm_wind_speed_freq_79_97_Sep = normalizeWindSpeeds(wind_speed_freq_79_97_Sep)
norm_wind_speed_freq_98_16_Sep = normalizeWindSpeeds(wind_speed_freq_98_16_Sep)
norm_wind_speed_freq_79_97_Oct = normalizeWindSpeeds(wind_speed_freq_79_97_Oct)
norm_wind_speed_freq_98_16_Oct = normalizeWindSpeeds(wind_speed_freq_98_16_Oct)
norm_wind_speed_freq_79_97_Nov = normalizeWindSpeeds(wind_speed_freq_79_97_Nov)
norm_wind_speed_freq_98_16_Nov = normalizeWindSpeeds(wind_speed_freq_98_16_Nov)

# Plot the two time periods side by side for each part of the hurricane season
fig, ax = plt.subplots(figsize = (20,15))
bar_width = 0.35
ax.bar(np.arange(len(norm_wind_speed_freq_79_97_EHS)), norm_wind_speed_freq_79_97_EHS, bar_width,
                color='b', label='1979-1997')
ax.bar(np.arange(len(norm_wind_speed_freq_98_16_EHS)) + bar_width, norm_wind_speed_freq_98_16_EHS,
                bar_width, color='r', label='1998-2016')
ax.set_xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    ax.set_ylim(ymax = 120000)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Frequency (%)')
ax.set_title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + ' in Late\nMay, ' +
             'June, and July During Hurricane Seasons from 1979 up to 1998\nvs. Hurricane ' +
             'Seasons from 1998 up to 2017')
ax.legend()
fig.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-97_98-16_EHS_SBS_Histogram.png')
fig.savefig('Figures/WS_79-97_98-16_EHS_SBS/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-97_98-16_EHS_SBS_Histogram.png')
plt.show()

fig, ax = plt.subplots(figsize = (20,15))
bar_width = 0.35
ax.bar(np.arange(len(norm_wind_speed_freq_79_97_MHS)), norm_wind_speed_freq_79_97_MHS, bar_width,
                color='b', label='1979-1997')
ax.bar(np.arange(len(norm_wind_speed_freq_98_16_MHS)) + bar_width, norm_wind_speed_freq_98_16_MHS,
                bar_width, color='r', label='1998-2016')
ax.set_xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    ax.set_ylim(ymax = 120000)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Frequency (%)')
ax.set_title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + ' in\nAugust ' +
             'and September During Hurricane Seasons from 1979 up to\n1998 vs. Hurricane ' +
             'Seasons from 1998 up to 2017')
ax.legend()
fig.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-97_98-16_MHS_SBS_Histogram.png')
fig.savefig('Figures/WS_79-97_98-16_MHS_SBS/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-97_98-16_MHS_SBS_Histogram.png')
plt.show()

#fig, ax = plt.subplots(figsize = (20,15))
#bar_width = 0.35
#ax.bar(np.arange(len(wind_speed_freq_79_97_Aug)), wind_speed_freq_79_97_Aug, bar_width,
#                color='b', label='1979-1997')
#ax.bar(np.arange(len(wind_speed_freq_98_16_Aug)) + bar_width, wind_speed_freq_98_16_Aug,
#                bar_width, color='r', label='1998-2016')
#ax.set_xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    ax.set_ylim(ymax = 120000)
#ax.set_xlabel('Wind Speed (m/s)')
#ax.set_ylabel('Frequency')
#ax.set_title('Frequency of Wind Speeds Recorded on ' + location_names[region][1] + ' in the\nMonth ' +
#             'of August During Hurricane Seasons from 1979 up to 1998\nvs. Hurricane ' +
#             'Seasons from 1998 up to 2017')
#ax.legend()
#fig.savefig('Figures/' + location_names[region][0]  + '/Wind_Speeds_' + location_names[region][0] + '_79-97_98-16_Aug_SBS_Histogram.png')
#fig.savefig('Figures/WS_79-97_98-16_Aug_SBS/Wind_Speeds_' + location_names[region][0] + '_79-97_98-16_Aug_SBS_Histogram.png')
#plt.show()

#fig, ax = plt.subplots(figsize = (20,15))
#bar_width = 0.35
#ax.bar(np.arange(len(wind_speed_freq_79_97_Sep)), wind_speed_freq_79_97_Sep, bar_width,
#                color='b', label='1979-1997')
#ax.bar(np.arange(len(wind_speed_freq_98_16_Sep)) + bar_width, wind_speed_freq_98_16_Sep,
#                bar_width, color='r', label='1998-2016')
#ax.set_xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    ax.set_ylim(ymax = 120000)
#ax.set_xlabel('Wind Speed (m/s)')
#ax.set_ylabel('Frequency')
#ax.set_title('Frequency of Wind Speeds Recorded on ' + location_names[region][1] + ' in the\nMonth ' +
#             'of September During Hurricane Seasons from 1979 up to\n1998 vs. Hurricane ' +
#             'Seasons from 1998 up to 2017')
#ax.legend()
#fig.savefig('Figures/' + location_names[region][0]  + '/Wind_Speeds_' + location_names[region][0] + '_79-97_98-16_Sep_SBS_Histogram.png')
#fig.savefig('Figures/WS_79-97_98-16_Sep_SBS/Wind_Speeds_' + location_names[region][0] + '_79-97_98-16_Sep_SBS_Histogram.png')
#plt.show()

fig, ax = plt.subplots(figsize = (20,15))
bar_width = 0.35
ax.bar(np.arange(len(norm_wind_speed_freq_79_97_LHS)), norm_wind_speed_freq_79_97_LHS, bar_width,
                color='b', label='1979-1997')
ax.bar(np.arange(len(norm_wind_speed_freq_98_16_LHS)) + bar_width, norm_wind_speed_freq_98_16_LHS,
                bar_width, color='r', label='1998-2016')
ax.set_xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    ax.set_ylim(ymax = 120000)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Frequency (%)')
ax.set_title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + ' in\nOctober ' +
             'and November During Hurricane Seasons from 1979 up to\n1998 vs. Hurricane ' +
             'Seasons from 1998 up to 2017')
ax.legend()
fig.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-97_98-16_LHS_SBS_Histogram.png')
fig.savefig('Figures/WS_79-97_98-16_LHS_SBS/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-97_98-16_LHS_SBS_Histogram.png')
plt.show()

# Determine differences in the wind speed frequencies between the two time periods
# for each part of the hurricane season (and for each month)
i = 0
wind_speed_freq_EHS_diff = [0] * len(wind_speed_freq_79_97_EHS)
norm_wind_speed_freq_EHS_diff = [0] * len(norm_wind_speed_freq_79_97_EHS)
while i < len(wind_speed_freq_79_97_EHS):
    wind_speed_freq_EHS_diff[i] = wind_speed_freq_98_16_EHS[i] - wind_speed_freq_79_97_EHS[i]
    norm_wind_speed_freq_EHS_diff[i] = norm_wind_speed_freq_98_16_EHS[i] - norm_wind_speed_freq_79_97_EHS[i]
    i = i + 1

i = 0
wind_speed_freq_MHS_diff = [0] * len(wind_speed_freq_79_97_MHS)
norm_wind_speed_freq_MHS_diff = [0] * len(norm_wind_speed_freq_79_97_MHS)
while i < len(wind_speed_freq_79_97_MHS):
    wind_speed_freq_MHS_diff[i] = wind_speed_freq_98_16_MHS[i] - wind_speed_freq_79_97_MHS[i]
    norm_wind_speed_freq_MHS_diff[i] = norm_wind_speed_freq_98_16_MHS[i] - norm_wind_speed_freq_79_97_MHS[i]
    i = i + 1

i = 0
wind_speed_freq_LHS_diff = [0] * len(wind_speed_freq_79_97_LHS)
norm_wind_speed_freq_LHS_diff = [0] * len(norm_wind_speed_freq_79_97_LHS)
while i < len(wind_speed_freq_79_97_LHS):
    wind_speed_freq_LHS_diff[i] = wind_speed_freq_98_16_LHS[i] - wind_speed_freq_79_97_LHS[i]
    norm_wind_speed_freq_LHS_diff[i] = norm_wind_speed_freq_98_16_LHS[i] - norm_wind_speed_freq_79_97_LHS[i]
    i = i + 1

i = 0
wind_speed_freq_June_diff = [0] * len(wind_speed_freq_79_97_June)
norm_wind_speed_freq_June_diff = [0] * len(norm_wind_speed_freq_79_97_June)
while i < len(wind_speed_freq_79_97_June):
    wind_speed_freq_June_diff[i] = wind_speed_freq_98_16_June[i] - wind_speed_freq_79_97_June[i]
    norm_wind_speed_freq_June_diff[i] = norm_wind_speed_freq_98_16_June[i] - norm_wind_speed_freq_79_97_June[i]
    i = i + 1

i = 0
wind_speed_freq_July_diff = [0] * len(wind_speed_freq_79_97_July)
norm_wind_speed_freq_July_diff = [0] * len(norm_wind_speed_freq_79_97_July)
while i < len(wind_speed_freq_79_97_July):
    wind_speed_freq_July_diff[i] = wind_speed_freq_98_16_July[i] - wind_speed_freq_79_97_July[i]
    norm_wind_speed_freq_July_diff[i] = norm_wind_speed_freq_98_16_July[i] - norm_wind_speed_freq_79_97_July[i]
    i = i + 1

i = 0
wind_speed_freq_Aug_diff = [0] * len(wind_speed_freq_79_97_Aug)
norm_wind_speed_freq_Aug_diff = [0] * len(norm_wind_speed_freq_79_97_Aug)
while i < len(wind_speed_freq_79_97_Aug):
    wind_speed_freq_Aug_diff[i] = wind_speed_freq_98_16_Aug[i] - wind_speed_freq_79_97_Aug[i]
    norm_wind_speed_freq_Aug_diff[i] = norm_wind_speed_freq_98_16_Aug[i] - norm_wind_speed_freq_79_97_Aug[i]
    i = i + 1

i = 0
wind_speed_freq_Sep_diff = [0] * len(wind_speed_freq_79_97_Sep)
norm_wind_speed_freq_Sep_diff = [0] * len(norm_wind_speed_freq_79_97_Sep)
while i < len(wind_speed_freq_79_97_Sep):
    wind_speed_freq_Sep_diff[i] = wind_speed_freq_98_16_Sep[i] - wind_speed_freq_79_97_Sep[i]
    norm_wind_speed_freq_Sep_diff[i] = norm_wind_speed_freq_98_16_Sep[i] - norm_wind_speed_freq_79_97_Sep[i]
    i = i + 1

i = 0
wind_speed_freq_Oct_diff = [0] * len(wind_speed_freq_79_97_Oct)
norm_wind_speed_freq_Oct_diff = [0] * len(norm_wind_speed_freq_79_97_Oct)
while i < len(wind_speed_freq_79_97_Oct):
    wind_speed_freq_Oct_diff[i] = wind_speed_freq_98_16_Oct[i] - wind_speed_freq_79_97_Oct[i]
    norm_wind_speed_freq_Oct_diff[i] = norm_wind_speed_freq_98_16_Oct[i] - norm_wind_speed_freq_79_97_Oct[i]
    i = i + 1

i = 0
wind_speed_freq_Nov_diff = [0] * len(wind_speed_freq_79_97_Nov)
norm_wind_speed_freq_Nov_diff = [0] * len(norm_wind_speed_freq_79_97_Nov)
while i < len(wind_speed_freq_79_97_Nov):
    wind_speed_freq_Nov_diff[i] = wind_speed_freq_98_16_Nov[i] - wind_speed_freq_79_97_Nov[i]
    norm_wind_speed_freq_Nov_diff[i] = norm_wind_speed_freq_98_16_Nov[i] - norm_wind_speed_freq_79_97_Nov[i]
    i = i + 1

# Create histograms that show the difference between the wind speed frequencies for
# the two time periods for each part of the hurricane season
plt.figure(1, figsize = (20,15))
plt.bar(np.arange(len(norm_wind_speed_freq_EHS_diff)), norm_wind_speed_freq_EHS_diff)
plt.xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    plt.ylim(ymax = 9000, ymin = -7000)
plt.ylabel('Frequency Difference (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Difference Between Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nin Late ' +
          'May, June, and July During Hurricane Seasons from 1979 up to 1998\nand During Hurricane ' +
          'Seasons from 1998 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_EHS_Histogram.png')
plt.savefig('Figures/WS_Diff_79-97_98-16_EHS/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_EHS_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,15))
plt.bar(np.arange(len(norm_wind_speed_freq_June_diff)), norm_wind_speed_freq_June_diff)
plt.xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    plt.ylim(ymax = 9000, ymin = -7000)
plt.ylabel('Frequency Difference')
plt.xlabel('Wind Speed (m/s)')
plt.title('Difference Between Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nin the Month ' +
          'of June During Hurricane Seasons from 1979 up to 1998\nand During Hurricane ' +
          'Seasons from 1998 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_June_Histogram.png')
plt.savefig('Figures/WS_Diff_79-97_98-16_June/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_June_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,15))
plt.bar(np.arange(len(norm_wind_speed_freq_July_diff)), norm_wind_speed_freq_July_diff)
plt.xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    plt.ylim(ymax = 9000, ymin = -7000)
plt.ylabel('Frequency Difference')
plt.xlabel('Wind Speed (m/s)')
plt.title('Difference Between Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nin the Month ' +
          'of July During Hurricane Seasons from 1979 up to 1998\nand During Hurricane ' +
          'Seasons from 1998 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_July_Histogram.png')
plt.savefig('Figures/WS_Diff_79-97_98-16_July/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_July_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,15))
plt.bar(np.arange(len(norm_wind_speed_freq_MHS_diff)), norm_wind_speed_freq_MHS_diff)
plt.xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    plt.ylim(ymax = 9000, ymin = -7000)
plt.ylabel('Frequency Difference (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Difference Between Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nin August ' +
          'and September During Hurricane Seasons from 1979 up to 1998\nand During Hurricane ' +
          'Seasons from 1998 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_MHS_Histogram.png')
plt.savefig('Figures/WS_Diff_79-97_98-16_MHS/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_MHS_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,15))
plt.bar(np.arange(len(norm_wind_speed_freq_Aug_diff)), norm_wind_speed_freq_Aug_diff)
plt.xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    plt.ylim(ymax = 9000, ymin = -7000)
plt.ylabel('Frequency Difference (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Difference Between Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nin the Month ' +
          'of August During Hurricane Seasons from 1979 up to 1998\nand During Hurricane ' +
          'Seasons from 1998 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_Aug_Histogram.png')
plt.savefig('Figures/WS_Diff_79-97_98-16_Aug/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_Aug_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,15))
plt.bar(np.arange(len(norm_wind_speed_freq_Sep_diff)), norm_wind_speed_freq_Sep_diff)
plt.xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    plt.ylim(ymax = 9000, ymin = -7000)
plt.ylabel('Frequency Difference')
plt.xlabel('Wind Speed (m/s)')
plt.title('Difference Between Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nin the Month ' +
          'of September During Hurricane Seasons from 1979 up to 1998\nand During Hurricane ' +
          'Seasons from 1998 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_Sep_Histogram.png')
plt.savefig('Figures/WS_Diff_79-97_98-16_Sep/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_Sep_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,15))
plt.bar(np.arange(len(norm_wind_speed_freq_LHS_diff)), norm_wind_speed_freq_LHS_diff)
plt.xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    plt.ylim(ymax = 9000, ymin = -7000)
plt.ylabel('Frequency Difference (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Difference Between Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nin October ' +
          'and November During Hurricane Seasons from 1979 up to 1998\nand During Hurricane ' +
          'Seasons from 1998 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_LHS_Histogram.png')
plt.savefig('Figures/WS_Diff_79-97_98-16_LHS/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_LHS_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,15))
plt.bar(np.arange(len(norm_wind_speed_freq_Oct_diff)), norm_wind_speed_freq_Oct_diff)
plt.xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    plt.ylim(ymax = 9000, ymin = -7000)
plt.ylabel('Frequency Difference')
plt.xlabel('Wind Speed (m/s)')
plt.title('Difference Between Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nin the Month ' +
          'of October During Hurricane Seasons from 1979 up to 1998\nand During Hurricane ' +
          'Seasons from 1998 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_Oct_Histogram.png')
plt.savefig('Figures/WS_Diff_79-97_98-16_Oct/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_Oct_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,15))
plt.bar(np.arange(len(norm_wind_speed_freq_Nov_diff)), norm_wind_speed_freq_Nov_diff)
plt.xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    plt.ylim(ymax = 9000, ymin = -7000)
plt.ylabel('Frequency Difference')
plt.xlabel('Wind Speed (m/s)')
plt.title('Difference Between Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nin the Month ' +
          'of November During Hurricane Seasons from 1979 up to 1998\nand During Hurricane ' +
          'Seasons from 1998 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_Nov_Histogram.png')
plt.savefig('Figures/WS_Diff_79-97_98-16_Nov/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_Nov_Histogram.png')
plt.show()

######################################################################################
##                                 Plot Time Series                                 ##
######################################################################################

#current_year = 1979
#current_year_index = current_year - 1979

# Average the wind speed between all locations at each time
#avg_wind_speeds_79_16 = averageWindsAmongAllPoints(wind_speeds_79_16)

# Plot averaged wind speeds over the course of a year
#plt.figure(1, figsize = (20,10))
#plt.plot(dates_and_times_79_16[(current_year_index * 732):(732 * (current_year_index + 1))],
#         avg_wind_speeds_79_16[(current_year_index * 732):(732 * (current_year_index + 1))])
#plt.ylabel('Wind Speed (m/s)')
#plt.xlabel('Time')
#plt.title('Average Wind Speed During the ' + str(current_year) + ' Hurricane Season on ' +
#          location_names[region][1])
#plt.savefig('Figures/Time_Series/' + str(current_year) + '/' + location_names[region][0]  + '_Time_Series_' + str(current_year) + '.png')
#plt.savefig('Figures/Time_Series/Yearly/' + location_names[region][0]  + '_Time_Series_' + str(current_year) + '.png')
#plt.show()


# Average the wind speeds by day
#avg_1d_wind_speeds_79_16, dates_and_times_79_16_1d = averageWindsOverTime_v1(wind_speeds_79_16, dates_and_times_79_16, 4)

# Average the wind speeds every six days
#avg_6d_wind_speeds_79_16, dates_and_times_79_16_6d = averageWindsOverTime_v1(wind_speeds_79_16, dates_and_times_79_16, 24)

# Average the wind speeds every twelve days
#avg_12d_wind_speeds_79_16, dates_and_times_79_16_12d = averageWindsOverTime_v1(wind_speeds_79_16, dates_and_times_79_16, 48)

# Average the wind speed by year (length of a hurricane season)
#avg_1y_wind_speeds_79_16, dates_and_times_79_16_1y = averageWindsOverTime(wind_speeds_79_16, dates_and_times_79_16, 732)

# Plot a time series for one location throughout the whole time period
#plt.figure(1, figsize = (20,10))
#plt.plot(dates_and_times_79_16, wind_speeds_79_16[0])
#plt.xlim(xmax = max_wind_speed + 1)
#plt.ylabel('Wind Speed (m/s)')
#plt.xlabel('Time')
#plt.title('Wind Speed Over Time During Hurricane Seasons from 1979 up to 2017 at a Location on ' +
#          location_names[region][1])
#plt.savefig('Figures/' + location_names[region][0]  + '/Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_Histogram.png')
#plt.savefig('Figures/WS_Diff_79-97_98-16/Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_Histogram.png')
#plt.show()

# Plot two time series on the same graph for the same location. The first time series
# goes from 1979 up to 1998, while the second time series spans from 1998 up to 2017

# Plot a time series for the whole region (averaged) throughout the whole time period

# Plot two time series on the same graph for the whole region (averaged). The first time series
# goes from 1979 up to 1998, while the second time series spans from 1998 up to 2017



# Average the wind speed between all locations at each time
#avg_1d_wind_speeds_79_16 = averageWindsAmongAllPoints(avg_1d_wind_speeds_79_16)

# Plot wind speeds averaged every day and averaged among all points
#plt.figure(1, figsize = (20,10))
#plt.plot(dates_and_times_79_16_1d[(current_year_index * (732/4)):((732/4) * (current_year_index + 1))],
#         avg_1d_wind_speeds_79_16[(current_year_index * (732/4)):((732/4) * (current_year_index + 1))])
#plt.xlim(xmax = max_wind_speed + 1)
#plt.ylabel('Wind Speed (m/s)')
#plt.xlabel('Time')
#plt.title('Wind Speed (Averaged Every Day) Over Time During the ' + str(current_year) + ' Hurricane Season on ' +
#          location_names[region][1])
#plt.savefig('Figures/One_Day_Averages/v1/Time_Series/' + str(current_year) + '/' + location_names[region][0]  + '_1d_Avg_Time_Series_' + str(current_year) + '.png')
#plt.savefig('Figures/One_Day_Averages/v1/Time_Series/Yearly/' + location_names[region][0]  + '_1d_Avg_Time_Series_' + str(current_year) + '.png')
#plt.show()

# Plot two time series (averaged by day) on the same graph for the same location. The first time series
# goes from 1979 up to 1998, while the second time series spans from 1998 up to 2017

# Plot a time series (averaged by day) for the whole region (averaged) throughout the whole time period

# Plot two time series (averaged by day) on the same graph for the whole region (averaged). The first time series
# goes from 1979 up to 1998, while the second time series spans from 1998 up to 2017


# Plot a time series (averaged every six days) for one location throughout the whole time period
#plt.figure(1, figsize = (20,10))
#plt.plot(dates_and_times_79_16_6d, avg_6d_wind_speeds_79_16[0])
#plt.xlim(xmax = max_wind_speed + 1)
#plt.ylabel('Wind Speed (m/s)')
#plt.xlabel('Time')
#plt.title('Wind Speed Over Time During Hurricane Seasons from 1979 up to 2017 at a Location on ' +
#          location_names[region][1])
#plt.savefig('Figures/' + location_names[region][0]  + '/Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_Histogram.png')
#plt.savefig('Figures/WS_Diff_79-97_98-16/Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_Histogram.png')
#plt.show()

# Plot two time series (averaged every six days) on the same graph for the same location. The first time series
# goes from 1979 up to 1998, while the second time series spans from 1998 up to 2017

# Plot a time series (averaged every six days) for the whole region (averaged) throughout the whole time period

# Plot two time series (averaged every six days) on the same graph for the whole region (averaged). The first time series
# goes from 1979 up to 1998, while the second time series spans from 1998 up to 2017


#print('Here')
#print(len(dates_and_times_79_16_1y))
#print(len(avg_1y_wind_speeds_79_16))
#print(len(avg_1y_wind_speeds_79_16[0]))
# Plot a time series (averaged by year) for one location throughout the whole time period
#plt.figure(1, figsize = (20,10))
#plt.plot(dates_and_times_79_16_1y, avg_1y_wind_speeds_79_16[0])
#plt.xlim(xmax = max_wind_speed + 1)
#plt.ylabel('Wind Speed (m/s)')
#plt.xlabel('Time')
#plt.title('Wind Speed Over Time During Hurricane Seasons from 1979 up to 2017 at a Location on ' +
#          location_names[region][1])
#plt.savefig('Figures/' + location_names[region][0]  + '/Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_Histogram.png')
#plt.savefig('Figures/WS_Diff_79-97_98-16/Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_Histogram.png')
#plt.show()

# Plot two time series (averaged by year) on the same graph for the same location. The first time series
# goes from 1979 up to 1998, while the second time series spans from 1998 up to 2017

# Plot a time series (averaged by year) for the whole region (averaged) throughout the whole time period

# Plot two time series (averaged by year) on the same graph for the whole region (averaged). The first time series
# goes from 1979 up to 1998, while the second time series spans from 1998 up to 2017


# Create the same graphs for each part of the hurricane season

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
plt.savefig('Figures/' + location_names[region][0]  + '/Wind_Speed_Measurement_Locations_' + location_names[region][0] + '.png')
plt.savefig('Figures/Measurement_Locations/Wind_Speed_Measurement_Locations_' + location_names[region][0] + '.png')
plt.show()

# Print some notable statistics about the data
print('\nTotal Number of Stagnant Flow Measurements Observed at ' + location_names[region][2] + ' from 1979 ' +
      'up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed at ' + location_names[region][2] + ' from 1979 ' +
      'up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed at ' + location_names[region][2] + ' from 1998 ' +
      'up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16, stagnant_flow)))
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' from 1979 up to 2017: ' + 
      str(np.mean(wind_speeds_79_16)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' from 1979 up to 1998: ' +
      str(np.mean(wind_speeds_79_97)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' from 1998 up to 2017: ' +
      str(np.mean(wind_speeds_98_16)) + ' m/s')
print('Median Observed Wind Speed for ' + location_names[region][2] + ' from 1979 up to 2017: ' +
      str(np.median(wind_speeds_79_16)) + ' m/s')
print('Median Observed Wind Speed for ' + location_names[region][2] + ' from 1979 up to 1998: ' +
      str(np.median(wind_speeds_79_97)) + ' m/s')
print('Median Observed Wind Speed for ' + location_names[region][2] + ' from 1998 up to 2017: ' +
      str(np.median(wind_speeds_98_16)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' from 1979 up to ' +
      '2017: ' + str(np.std(wind_speeds_79_16)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' from 1979 up to ' +
      '1998: ' + str(np.std(wind_speeds_79_97)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' from 1998 up to ' +
      '2017: ' + str(np.std(wind_speeds_98_16)) + ' m/s\n')

print('Total Number of Stagnant Flow Measurements Observed in Late May, June, and July ' +
      'at ' + location_names[region][2] + ' from 1979 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all_EHS, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in Late May, June, and July ' +
      'at ' + location_names[region][2] + ' from 1979 up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97_EHS, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in Late May, June, and July ' +
      'at ' + location_names[region][2] + ' from 1998 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16_EHS, stagnant_flow)))
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in Late May, June, and July from 1979 up to 2017: ' +
      str(np.mean(wind_speeds_79_16_EHS)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in Late May, June, and July from 1979 up to 1998: ' +
      str(np.mean(wind_speeds_79_97_EHS)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in Late May, June, and July from 1998 up to 2017: ' +
      str(np.mean(wind_speeds_98_16_EHS)) + ' m/s')
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in Late May, June, and July from 1979 up to 2017: ' +
      str(np.median(wind_speeds_79_16_EHS)) + ' m/s')
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in Late May, June, and July from 1979 up to 1998: ' +
      str(np.median(wind_speeds_79_97_EHS)) + ' m/s')
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in Late May, June, and July from 1998 up to 2017: ' +
      str(np.median(wind_speeds_98_16_EHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in Late May, June, and July ' +
      'from 1979 up to 2017: ' + str(np.std(wind_speeds_79_16_EHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in Late May, June, and July ' +
      'from 1979 up to 1998: ' + str(np.std(wind_speeds_79_97_EHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in Late May, June, and July ' +
      'from 1998 up to 2017: ' + str(np.std(wind_speeds_98_16_EHS)) + ' m/s\n')

print('Total Number of Stagnant Flow Measurements Observed in August and September ' +
      'at ' + location_names[region][2] + ' from 1979 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all_MHS, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in August and September ' +
      'at ' + location_names[region][2] + ' from 1979 up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97_MHS, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in August and September ' +
      'at ' + location_names[region][2] + ' from 1998 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16_MHS, stagnant_flow)))
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in August and September from 1979 up to 2017: ' +
      str(np.mean(wind_speeds_79_16_MHS)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in August and September from 1979 up to 1998: ' +
      str(np.mean(wind_speeds_79_97_MHS)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in August and September from 1998 up to 2017: ' +
      str(np.mean(wind_speeds_98_16_MHS)) + ' m/s')
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in August and September from 1979 up to 2017: ' +
      str(np.median(wind_speeds_79_16_MHS)) + ' m/s')
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in August and September from 1979 up to 1998: ' +
      str(np.median(wind_speeds_79_97_MHS)) + ' m/s')
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in August and September from 1998 up to 2017: ' +
      str(np.median(wind_speeds_98_16_MHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in August and September ' +
      'from 1979 up to 2017: ' + str(np.std(wind_speeds_79_16_MHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in August and September ' +
      'from 1979 up to 1998: ' + str(np.std(wind_speeds_79_97_MHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in August and September ' +
      'from 1998 up to 2017: ' + str(np.std(wind_speeds_98_16_MHS)) + ' m/s\n')

print('Total Number of Stagnant Flow Measurements Observed in October and November ' +
      'at ' + location_names[region][2] + ' from 1979 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all_LHS, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in October and November ' +
      'at ' + location_names[region][2] + ' from 1979 up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97_LHS, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in October and November ' +
      'at ' + location_names[region][2] + ' from 1998 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16_LHS, stagnant_flow)))
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in October and November from 1979 up to 2017: ' +
      str(np.mean(wind_speeds_79_16_LHS)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in October and November from 1979 up to 1998: ' +
      str(np.mean(wind_speeds_79_97_LHS)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in October and November from 1998 up to 2017: ' +
      str(np.mean(wind_speeds_98_16_LHS)) + ' m/s')
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in October and November from 1979 up to 2017: ' +
      str(np.median(wind_speeds_79_16_LHS)) + ' m/s')
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in October and November from 1979 up to 1998: ' +
      str(np.median(wind_speeds_79_97_LHS)) + ' m/s')
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in October and November from 1998 up to 2017: ' +
      str(np.median(wind_speeds_98_16_LHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in October and November ' +
      'from 1979 up to 2017: ' + str(np.std(wind_speeds_79_16_LHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in October and November ' +
      'from 1979 up to 1998: ' + str(np.std(wind_speeds_79_97_LHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in October and November ' +
      'from 1998 up to 2017: ' + str(np.std(wind_speeds_98_16_LHS)) + ' m/s\n')

print('Total Number of Stagnant Flow Measurements Observed in August ' +
      'at ' + location_names[region][2] + ' from 1979 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all_Aug, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in August ' +
      'at ' + location_names[region][2] + ' from 1979 up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97_Aug, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in August ' +
      'at ' + location_names[region][2] + ' from 1998 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16_Aug, stagnant_flow)))
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in August from 1979 up to 2017: ' +
      str(np.mean(wind_speeds_79_16_Aug)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in August from 1979 up to 1998: ' +
      str(np.mean(wind_speeds_79_97_Aug)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in August from 1998 up to 2017: ' +
      str(np.mean(wind_speeds_98_16_Aug)) + ' m/s')
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in August from 1979 up to 2017: ' +
      str(np.median(wind_speeds_79_16_Aug)) + ' m/s')
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in August from 1979 up to 1998: ' +
      str(np.median(wind_speeds_79_97_Aug)) + ' m/s')
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in August from 1998 up to 2017: ' +
      str(np.median(wind_speeds_98_16_Aug)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in August ' +
      'from 1979 up to 2017: ' + str(np.std(wind_speeds_79_16_Aug)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in August ' +
      'from 1979 up to 1998: ' + str(np.std(wind_speeds_79_97_Aug)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in August ' +
      'from 1998 up to 2017: ' + str(np.std(wind_speeds_98_16_Aug)) + ' m/s\n')

print('Total Number of Stagnant Flow Measurements Observed in September ' +
      'at ' + location_names[region][2] + ' from 1979 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all_Sep, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in September ' +
      'at ' + location_names[region][2] + ' from 1979 up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97_Sep, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in September ' +
      'at ' + location_names[region][2] + ' from 1998 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16_Sep, stagnant_flow)))
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in September from 1979 up to 2017: ' +
      str(np.mean(wind_speeds_79_16_Sep)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in September from 1979 up to 1998: ' +
      str(np.mean(wind_speeds_79_97_Sep)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in September from 1998 up to 2017: ' +
      str(np.mean(wind_speeds_98_16_Sep)) + ' m/s')
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in September from 1979 up to 2017: ' +
      str(np.median(wind_speeds_79_16_Sep)) + ' m/s')
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in September from 1979 up to 1998: ' +
      str(np.median(wind_speeds_79_97_Sep)) + ' m/s')
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in September from 1998 up to 2017: ' +
      str(np.median(wind_speeds_98_16_Sep)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in September ' +
      'from 1979 up to 2017: ' + str(np.std(wind_speeds_79_16_Sep)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in September ' +
      'from 1979 up to 1998: ' + str(np.std(wind_speeds_79_97_Sep)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in September ' +
      'from 1998 up to 2017: ' + str(np.std(wind_speeds_98_16_Sep)) + ' m/s\n')
