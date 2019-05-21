######################################################################################
#                                                                                    #
# Deep Layer Mean Steering Flow Statistics Calculator                                #
#                                                                                    #
# Author: Rafal Ogorek (rafalogorek@yahoo.com)                                       #
#                                                                                    #
# Description: This program makes use of reanalysis data to calculate various        #
#              statistics for the deep layer mean steering flow winds along the      #
#              coastline of the southeastern U.S. from 1979 to 2017. Several         #
#              graphics are also generated to help visualize this data better.       #
#                                                                                    #
#              The program can be run by typing the following command:               #
#                                                                                    #
#              python DLM_stats.py <data_file> <location>                            #
#                                                                                    #
#              <data_file> is the name of the file with the deep layer mean data.    #
#              For our use, this file is currently timeseries_June2018.npz.          #
#                                                                                    #
#              <location> is an string abbreviation for the specific region to       #
#              look at. Below are all possible locations that can be looked at,      #
#              along with the string that needs to be put in as an argument for      #
#              the program to run for that location:                                 #
#                                                                                    #
#              'AL': All condsidered locations                                       #
#              'NTX': North Texas coast                                              #
#              'STX': South Texas coast                                              #
#              'TX': Texas coast (NTX + STX)                                         #
#              'LA-MS': Louisiana/Mississippi coast                                  #
#              'AL-FL': Alabama/Florida panhandle coast                              #
#              'WFL': West Florida coast                                             #
#              'EFL': East Florida coast                                             #
#              'NFL': North Florida coast                                            #
#              'SFL': South Florida coast                                            #
#              'FL': Florida coast (WFL + EFL or NFL + SFL)                          #
#              'GA-SC': Georgia/South Carolina coast                                 #
#              'NC': North Carolina coast (currently unavailable to use)             #
#              'AC': Atlantic coast (EFL + GA-SC)                                    #
#              'GOM': Gulf of Mexico coast (TX + LA-MS + AL-FL)                      #
#                                                                                    #
# Last updated: 11/20/2018                                                            #
#                                                                                    #
######################################################################################

import math
import random
import sys
import numpy as np
import matplotlib as mpl
#matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
from mpl_toolkits.basemap import Basemap

# Description: Converts floats to Datetime objects
# Input: -wind_speed_times: An array of floats (which should represent a time and date
#                           for each wind speed)
# Output: -dates_and_times: An array of Datetime objects that have been converted from
#                           the floats in the wind_speed_times array
def convertToDatetime(wind_speed_times):
    dates_and_times = []
    for mt in wind_speed_times:
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
#                      is a wind speed in m/s
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
            # Make the wind speed a positive value (if necessary) and round it down to
            # the nearest integer. Then, increment the frequency of the wind speed range
            # that this value falls in.
            if not np.isnan(wind_speed):
                current_ws = int(math.floor(abs(wind_speed)))
                if current_ws < len(wind_speed_freq):
                    wind_speed_freq[current_ws] = wind_speed_freq[current_ws] + 1
                else:
                    # Optionally, disregard high wind speeds
                    current_ws = current_ws


# Description: Populates a translation speed frequency array with the frequency of
#              translation speeds within each 1 m/s interval represented in the
#              translation speed frequency array
# Input: -best_track_speeds: An array containing translation speeds (in m/s) of all
#                            tropical cyclones that were within one degree of the
#                            region of interest during north Atlantic hurricane
#                            seasons from 1979 to 2016
#        -trans_speed_freq: An array that will store the frequency of a range of
#                           translation speeds that correspond to its indices (i.e.
#                           index 0 will store the frequency of translation speeds
#                           between 0 m/s and 1 m/s; index 1 will store the frequency
#                           of translation speeds between 1 m/s and 2 m/s; etc...).
#                           When passed into the function, all entries should be 0
#                           since no frequencies have been calculated yet.
# Output: Nothing, but the trans_speed_freq array should have been modified to now
#         contain the correct frequency value for each speed range
def getBestTrackFrequencies(best_track_speeds, trans_speed_freq):
        for translation_speed in best_track_speeds:
            # Make the wind speed a positive value (if necessary) and round it down to
            # the nearest integer. Then, increment the frequency of the wind speed range
            # that this value falls in.
            if not np.isnan(translation_speed):
                current_ts = int(math.floor(abs(translation_speed)))
                if current_ts < len(trans_speed_freq):
                    trans_speed_freq[current_ts] = trans_speed_freq[current_ts] + 1
                else:
                    # Optionally, disregard high wind speeds
                    current_ts = current_ts


# Description: Obtains the wind speeds at all locations within a specified time interval
# Input: -wind_speeds: An array that stores an array of floats. Each array within
#                      this array represents a different location, while each float
#                      is a wind speed in m/s
#        -left_bound: The index that represents the beginning of the desired time interval
#                     (included in the time interval that is returned)
#        -right_bound: The index that represents the end of the desired time interval (not
#                      included in the time interval that is returned)
# Output: -wind_speed_interval: An array that stores an array of floats. It includes
#                               all locations, but only includes the wind speeds that
#                               were from the specified time period
def getWindSpeedInterval(wind_speeds, left_bound, right_bound):
    # Intialize wind_speed_interval to have same size as wind_speeds
    wind_speed_interval = [None] * len(wind_speeds)
    i = 0

    # Get data for all locations, but only during the specified time period
    for loc in wind_speeds:
        wind_speed_interval[i] = loc[left_bound:right_bound]
        i = i + 1

    return wind_speed_interval


# Description: Determines how many of the wind speeds in the wind speed
#              frequency array can be classified as stagnant flow
# Input: -wind_speed_freq: An array that will store the frequency of a range of wind
#                          speeds that correspond to its indices (i.e. index 0 will
#                          store the frequency of wind speeds between 0 m/s and 1 m/s;
#                          index 1 will store the frequency of wind speeds between
#                          1 m/s and 2 m/s; etc...).
#        -stagnant_flow: An integer signifying the upper threshold for what wind speed
#                        is considered to be stagnant flow
# Output: -num_stag_flow: The number of wind speeds in the wind_speed_freq array that
#                         are classified as stagnant flow
def calcNumStagFlow(wind_speed_freq, stagnant_flow):
    i = 0
    num_stag_flow = 0

    # Add the value for the current index in wind_speed_freq to the number of stagnant
    # flow data until the index that represents the upper threshold for stagnant
    # flow is reached
    while i < stagnant_flow:
        num_stag_flow = num_stag_flow + wind_speed_freq[i]
        i = i + 1

    return num_stag_flow


# Description: Splits the wind speed data into five different arrays, based on the
#              time of the hurricane season that the wind speed was from
# Input: -wind_speeds: An array that stores an array of floats. Each array within
#                      this array represents a different location, while each float
#                      is a wind speed in m/s
# Output: -wind_speeds_EHS: An array that stores an array of floats. Each array within
#                           this array represents a different location, while each float
#                           is a wind speed in m/s. All wind speeds in this
#                           array were from the early part of the Atlantic hurricane
#                           season (late May, June, or July)
#         -wind_speeds_MHS: Like wind_speeds_EHS, except all wind speeds in this array
#                           were from the middle part of the Atlantic hurricane
#                           season (August or September)
#         -wind_speeds_MHS_v2: Like wind_speeds_MHS, except all wind speeds in this array
#                              were from July or August
#         -wind_speeds_LHS: Like wind_speeds_EHS and wind_speeds_MHS, except all wind
#                           speeds in this array were from the late part of the
#                           Atlantic hurricane season (October or November)
#         -wind_speeds_LHS_v2: Like wind_speeds_LHS, except all wind speeds in this array
#                              were from September or October
#         -wind_speeds_June: Same as the above arrays, except all wind speeds are from June
#                            (plus the last 2 days of May)
#         -wind_speeds_July: Same as the above arrays, except all wind speeds are from July
#         -wind_speeds_Aug: Same as the above arrays, except all wind speeds are from August
#         -wind_speeds_Sep: Same as the above arrays, except all wind speeds are from September
#         -wind_speeds_Oct: Same as the above arrays, except all wind speeds are from October
#         -wind_speeds_Nov: Same as the above arrays, except all wind speeds are from November
def divideBySeason(wind_speeds):
    wind_speeds_EHS = [[None]] * len(wind_speeds)
    wind_speeds_MHS = [[None]] * len(wind_speeds)
    wind_speeds_LHS = [[None]] * len(wind_speeds)
    wind_speeds_MHS_v2 = [[None]] * len(wind_speeds)
    wind_speeds_LHS_v2 = [[None]] * len(wind_speeds)
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
        # Go through all wind speed data
        while j < len(wind_speeds[i]):
            # If data was in some month prior to August, add it to the early hurricane season
            # wind speed array
            if (j % 732) < 252:
                if wind_speeds_EHS[i] == [None]:
                    wind_speeds_EHS[i] = [wind_speeds[i][j]]
                else:
                    wind_speeds_EHS[i].append(wind_speeds[i][j])

                # Also add the data to the June or July array
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

                    if wind_speeds_MHS_v2[i] == [None]:
                        wind_speeds_MHS_v2[i] = [wind_speeds[i][j]]
                    else:
                        wind_speeds_MHS_v2[i].append(wind_speeds[i][j])

            # If data was in October or later, add it to the late hurricane season
            # wind speed array
            elif (j % 732) >= 496:
                if wind_speeds_LHS[i] == [None]:
                    wind_speeds_LHS[i] = [wind_speeds[i][j]]
                else:
                    wind_speeds_LHS[i].append(wind_speeds[i][j])

                # Also add the data to the October or November array
                # October case
                if (j % 732) < 620:
                    if wind_speeds_Oct[i] == [None]:
                        wind_speeds_Oct[i] = [wind_speeds[i][j]]
                    else:
                        wind_speeds_Oct[i].append(wind_speeds[i][j])

                    if wind_speeds_LHS_v2[i] == [None]:
                        wind_speeds_LHS_v2[i] = [wind_speeds[i][j]]
                    else:
                        wind_speeds_LHS_v2[i].append(wind_speeds[i][j])

                # November case
                else:
                    if wind_speeds_Nov[i] == [None]:
                        wind_speeds_Nov[i] = [wind_speeds[i][j]]
                    else:
                        wind_speeds_Nov[i].append(wind_speeds[i][j])

            # Otherwise, the wind speed was in August or September, so
            # add it to the mid hurricane season wind speed array
            else:
                if wind_speeds_MHS[i] == [None]:
                    wind_speeds_MHS[i] = [wind_speeds[i][j]]
                else:
                    wind_speeds_MHS[i].append(wind_speeds[i][j])

                # Also add the data to the August or September array
                # August case
                if (j % 732) < 376:
                    if wind_speeds_Aug[i] == [None]:
                        wind_speeds_Aug[i] = [wind_speeds[i][j]]
                    else:
                        wind_speeds_Aug[i].append(wind_speeds[i][j])

                    if wind_speeds_MHS_v2[i] == [None]:
                        wind_speeds_MHS_v2[i] = [wind_speeds[i][j]]
                    else:
                        wind_speeds_MHS_v2[i].append(wind_speeds[i][j])
                # September case
                else:
                    if wind_speeds_Sep[i] == [None]:
                        wind_speeds_Sep[i] = [wind_speeds[i][j]]
                    else:
                        wind_speeds_Sep[i].append(wind_speeds[i][j])

                    if wind_speeds_LHS_v2[i] == [None]:
                        wind_speeds_LHS_v2[i] = [wind_speeds[i][j]]
                    else:
                        wind_speeds_LHS_v2[i].append(wind_speeds[i][j])

            j = j + 1

        i = i + 1

    return wind_speeds_EHS, wind_speeds_MHS, wind_speeds_MHS_v2, wind_speeds_LHS, wind_speeds_LHS_v2, \
           wind_speeds_June, wind_speeds_July, wind_speeds_Aug, wind_speeds_Sep, wind_speeds_Oct, wind_speeds_Nov


# Description: Averages wind speeds over a specified time interval.
# Input: -wind_speeds: An array that stores an array of floats. Each array within
#                      this array represents a different location, while each float
#                      is a wind speed in m/s
#        -dates_and_times: An array that stores Datetime objects. Each index in the array
#                          corresponds to the time that data was obtained for
#        -time_interval: An integer representing the period of time the winds should be
#                        averaged over. A time interval of 1 would correspond to averaging
#                        over every 6 hours (this would just be the time period between
#                        data), a time interval of 2 would correspond to averaging
#                        over every 12 hours, etc. The time interval needs to be able to
#                        divide the length of the wind speed array in order for this
#                        function to work properly
# Output: -avg_wind_speeds: An array that stores an array of floats. It includes
#                           all locations, but should be a condensed version of
#                           wind_speeds where it stores the average wind speeds based
#                           on the specified time interval
#         -new_dates_and_times: An array that stores Datetime objects. Each index in the
#                               array now represents the beginning time of when the
#                               averaged wind speeds occurred
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
#                      is a wind speed in m/s
#        -dates_and_times: An array that stores Datetime objects. Each index in the array
#                          corresponds to the time that data was obtained for
#        -time_interval: An integer representing the period of time the winds should be
#                        averaged over. A time interval of 1 would correspond to averaging
#                        over every 6 hours (this would just be the time period between
#                        data), a time interval of 2 would correspond to averaging
#                        over every 12 hours, etc. The time interval needs to be able to
#                        divide the length of the wind speed array in order for this
#                        function to work properly
# Output: -avg_wind_speeds: An array that stores an array of floats. It includes
#                           all locations, but should be a condensed version of
#                           wind_speeds where it stores the average wind speeds based
#                           on the specified time interval
#         -new_dates_and_times: An array that stores Datetime objects. Each index in the
#                               array now represents the beginning time of when the
#                               averaged wind speeds were from
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
#                      is a wind speed in m/s
#        -dates_and_times: An array that stores Datetime objects. Each index in the array
#                          corresponds to the time for that data
#        -time_interval: An integer representing the period of time the winds should be
#                        averaged over. A time interval of 1 would correspond to averaging
#                        over every 6 hours (this would just be the time period between
#                        data), a time interval of 2 would correspond to averaging
#                        over every 12 hours, etc. The time interval needs to be able to
#                        divide the length of the wind speed array in order for this
#                        function to work properly
# Output: -avg_wind_speeds: An array of floats, where each float is the average wind
#                           speed between all locations and all hurricane seasons
#                           at the current time
#         -new_dates_and_times: An updated version of the dates_and_times array with certain
#                               times removed based on which wind speeds were removed
def averageWindsEachTime(wind_speeds, dates_and_times, time_interval):
    avg_wind_speeds = []
    new_dates_and_times = []

    i = 0
    # Go through each time that data was obtained for during hurricane seasons
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
            # If all wind speeds were removed, set the average to NaN
            avg_wind_speeds.append(float('nan'))

        i = i + 1

    #new_dates_and_times = dates_and_times
    # Optionally, also perform a low pass filter averaging method to smoothen the curve
    temp_wind_speeds = avg_wind_speeds
    avg_wind_speeds = []
    i = time_interval / 2
    while i < (len(temp_wind_speeds) - (time_interval / 2) + 1):
        # Update times
        #new_dates_and_times.append(dates_and_times[i])
        new_dates_and_times.append(datetime.strptime(dates_and_times[i].strftime('%m-%d %H:%M:%S'), '%m-%d %H:%M:%S'))

        # Average wind speeds
        avg_wind_speeds.append(np.sum(temp_wind_speeds[(i - (time_interval / 2)):(i + (time_interval / 2))])/time_interval)

        i = i + 1

    return avg_wind_speeds, new_dates_and_times


# Description: Averages wind speeds between all locations for the whole time period
# Input: -wind_speeds: An array that stores an array of floats. Each array within
#                      this array represents a different location, while each float
#                      is a wind speed in m/s
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
            # If all wind speeds were removed, set the average to NaN
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
        norm_wind_speed_freq.append(100 * (float(freq)/float(sum_freq)))

    return norm_wind_speed_freq


# Description: Filters out wind speeds taken during tropical cyclone landfalls or
#              near-landfalls
# Input: -wind_speeds: An array that stores an array of floats. Each array within
#                      this array represents a different location, while each float
#                      is a wind speed in m/s
#        -dates_and_times: An array that stores Datetime objects. Each index in the array
#                          corresponds to the time that a measurement was taken at
#        -locs: An array storing the coordinates of the locations where wind speed
#               data was obtained for
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

    # Loop through times and look for times that have data that need to be removed
    i = 0
    while i < len(times_to_remove):
        j = 0

        # Remove data for all locations that need to have them removed at the
        # current time
        while j < len(locs_to_remove[i]):
            # Determine the proper location index
            k = 0
            while k < len(locs):
                if np.all(locs_to_remove[i][j] == locs[k]):
                    break

                k = k + 1

            # Remove the data by setting it to NaN
            new_wind_speeds[k][dates_and_times.index(times_to_remove[i])] = float('nan')
            j = j + 1

        i = i + 1

    return new_wind_speeds


# Description: Reads in HURDAT2 best track data and populates arrays with times and
#              locations of tropical cyclones
# Input: -filename: A string containing the name of the text file that contains the
#                   hurricane best track data
#        -locs: An array storing the coordinates of the locations where wind speed
#               data was obtained for
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

           # Only take times at 0000Z, 0600Z, 1200Z, and 1800Z
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


# Description: Reads in HURDAT2 best track data and quantifies tropical cyclone translation
#              speeds in a few different ways
# Input: -filename: A string containing the name of the text file that contains the
#                   hurricane best track data
#        -locs: An array storing the coordinates of the locations where wind speed
#               data was obtained for
# Output: -best_track_speeds: An array containing translation speeds (in m/s) of all
#                             tropical cyclones that were within one degree of the region
#                             of interest during north Atlantic hurricane seasons from
#                             1979 to 2016
#         -best_track_speeds_79_97: Same as best_track_speeds except from 1979 through 1997
#         -best_track_speeds_98_16: Same as best_track_speeds except from 1998 through 2016
def getBestTrackSpeeds(filename, locs):
    # Intialize best track speed arrays
    best_track_speeds = []
    best_track_speeds_79_97 = []
    best_track_speeds_98_16 = []

    # Define the Earth's radius in kilometers
    earth_radius = 6371

    # Read file
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    # Remove any cyclone information that is not at 0000Z, 0600Z, 1200Z, or 1800Z. Also keep tropical cyclone name header
    new_lines = []
    for line in lines:
        if (line[10:14] == '0000') or (line[10:14] == '0600') or (line[10:14] == '1200') or (line[10:14] == '1800'):
            new_lines.append(line)
        elif line[0:2] == 'AL':
            new_lines.append(line)

    lines = new_lines

    # Parse data line by line
    i = 0
    while i < len(lines):
        # Data had to have been taken between May 30 and November 28. Also exclude data from 2017
        if (lines[i][0:4] != '2017') and (((int(lines[i][4:6]) > 5) and (int(lines[i][4:6]) < 11)) or (((int(lines[i][4:6]) == 5) and \
           (int(lines[i][6:8]) >= 30))) or (((int(lines[i][4:6]) == 11) and (int(lines[i][6:8]) <= 28)))) and (lines[i][0:2] != 'AL'):

           # Only consider tropical storms and hurricanes
           if (lines[i][19:21] == 'TS') or (lines[i][19:21] == 'HU'):

               # Ignore any tropical cyclones that were too far away from the region of interest
               if (float(lines[i][23:27]) < 38) and (float(lines[i][23:27]) > 20) and (float(lines[i][30:35]) < 103) and (float(lines[i][30:35]) > 73):

                   # Check if the cyclone's eye is within 1 degree of a data point
                   in_range = 0
                   for loc in locs:
                       if (abs(float(lines[i][23:27]) - abs(loc[1])) < 1) and (abs(float(lines[i][30:35]) - abs(loc[0])) < 1):
                           in_range = 1
                           break
                       # Also consider the case where the cyclone was within the desired range at the previous time step
                       # but moved out of the range at the current time step
                       elif (lines[i-1][0:2] != 'AL') and (abs(float(lines[i-1][23:27]) - abs(loc[1])) < 1) and (abs(float(lines[i-1][30:35]) - abs(loc[0])) < 1):
                           in_range = 1
                           break

                   if in_range:
                       # Check if the same tropical cyclone was a tropical storm or hurricane at the previous timestep. If that criteria is met, then
                       # calculate the speed of the tropical cyclone based on the change in location between the previous and current time step.
                       if ((lines[i-1][19:21] == 'TS') or (lines[i-1][19:21] == 'HU')):
                           # Determine the distance between consecutive eye locations in degrees
                           d_lat = abs(float(lines[i][23:27]) - float(lines[i-1][23:27]))
                           d_lon = abs(float(lines[i][30:35]) - float(lines[i-1][30:35]))

                           # Convert degrees to radians
                           d_lat = d_lat * math.pi/180
                           d_lon = d_lon * math.pi/180

                           # Use the Haversine formula to convert degrees to km
                           a = (math.sin(d_lat/2) * math.sin(d_lat/2)) + (math.cos(float(lines[i][23:27]) * math.pi/180) * \
                                math.cos(float(lines[i-1][23:27]) * math.pi/180) * math.sin(d_lon/2) * math.sin(d_lon/2))
                           dist = 2 * earth_radius * math.atan2(math.sqrt(a), math.sqrt(1-a))

                           # Calculate the speed of the tropical cyclone in the current 6 hour interval
                           speed = dist/6

                           # Convert from km/hr to m/s
                           speed = speed * 1000/3600

                           # Add the translation speed to the best_track_speed array
                           best_track_speeds.append(speed)
                           # Also add the speed to a best_track_speed array that corresponds to the 19
                           # year period that it occurred in
                           if int(lines[i][0:4]) < 1998:
                               best_track_speeds_79_97.append(speed)
                           else:
                               best_track_speeds_98_16.append(speed)

        i = i + 1

    return best_track_speeds, best_track_speeds_79_97, best_track_speeds_98_16


# Description: Computes the average DLM wind speed for each year represented in a
#              DLM wind speed array and plots a time series of how the annual mean
#              changes from 1979 to 2016. Also performs this process for each month
#              in the north Atlantic hurricane season
# Input: -wind_speeds: An array that stores an array of floats. Each array within
#                      this array represents a different location, while each float
#                      is a wind speed in m/s
#        -location_names: An array storing names of locations in the region of interest.
#                         Only used for plot titles and file names for plot images.
# Output: none
def getAnnualMeans(wind_speeds, location_names):
    annual_means = []
    annual_means_June = []
    annual_means_July = []
    annual_means_Aug = []
    annual_means_Sep = []
    annual_means_Oct = []
    annual_means_Nov = []

    # Loop through all times
    i = 0
    total_sum = 0
    total_count = 0
    month_sum = 0
    month_count = 0
    while i < len(wind_speeds[0]):
        # Loop through all locations
        j = 0
        while j < len(wind_speeds):
            # Sum all wind speeds at all locations for the current year
            if (~np.isnan(wind_speeds[j][i])):
                total_sum = total_sum + wind_speeds[j][i]
                total_count = total_count + 1
                month_sum = month_sum + wind_speeds[j][i]
                month_count = month_count + 1

            j = j + 1

        # Average wind speeds for the current year and add them to the annual mean array
        # Also calculate annual means for individual months
        if (i % 732 == 731):
            # Whole season
            annual_means.append(total_sum/total_count)
            total_sum = 0
            total_count = 0
            # November
            annual_means_Nov.append(month_sum/month_count)
            month_sum = 0
            month_count = 0
        # June
        elif (i % 732 == 127):
            annual_means_June.append(month_sum/month_count)
            month_sum = 0
            month_count = 0
        # July
        elif (i % 732 == 251):
            annual_means_July.append(month_sum/month_count)
            month_sum = 0
            month_count = 0
        # August
        elif (i % 732 == 375):
            annual_means_Aug.append(month_sum/month_count)
            month_sum = 0
            month_count = 0
        # September
        elif (i % 732 == 495):
            annual_means_Sep.append(month_sum/month_count)
            month_sum = 0
            month_count = 0
        # October
        elif (i % 732 == 619):
            annual_means_Oct.append(month_sum/month_count)
            month_sum = 0
            month_count = 0

        i = i + 1

    # Determine linear regresion lines
    years = [1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997,
             1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
    am_fit = np.polyfit(years, annual_means, 1)
    am_June_fit = np.polyfit(years, annual_means_June, 1)
    am_July_fit = np.polyfit(years, annual_means_July, 1)
    am_Aug_fit = np.polyfit(years, annual_means_Aug, 1)
    am_Sep_fit = np.polyfit(years, annual_means_Sep, 1)
    am_Oct_fit = np.polyfit(years, annual_means_Oct, 1)
    am_Nov_fit = np.polyfit(years, annual_means_Nov, 1)

    # Generate subplots of annual mean DLMSF for June, July, August, September, October, and November
    # Also add subplot for whole season
    plt.figure(1, figsize = (30,30))
    plt.rc('axes', titlesize = 30)
    plt.rc('axes', labelsize = 25)
    plt.rc('xtick', labelsize = 22)
    plt.rc('ytick', labelsize = 22)
    plt.subplot2grid((4,8), (0,2), colspan = 4)
    #plt.ylim(ymin = 0, ymax = 21)
    plt.xlim(xmin = 1978, xmax = 2017)
    plt.plot(years, annual_means, '-b', years, np.poly1d(am_fit)(years), '-k', linewidth = 2)
    #plt.text(1980, 6.25, 'Linear Regression Slope = ' + str(round(am_fit[0], 6)), fontsize = 20)
    plt.ylabel('Wind Speed (m/s)')
    plt.xlabel('Year')
    plt.title('a.) Whole Hurricane Season')

    plt.subplot(423)
    plt.rc('axes', titlesize = 30)
    plt.rc('axes', labelsize = 25)
    plt.rc('xtick', labelsize = 22)
    plt.rc('ytick', labelsize = 22)
    #plt.ylim(ymin = 0, ymax = 21)
    plt.xlim(xmin = 1978, xmax = 2017)
    plt.plot(years, annual_means_June, '-b', years, np.poly1d(am_June_fit)(years), '-k', linewidth = 2)
    #plt.text(1995, 9.5, 'Linear Regression Slope = ' + str(round(am_June_fit[0], 6)), fontsize = 20)
    plt.ylabel('Wind Speed (m/s)')
    plt.xlabel('Year')
    plt.title('b.) June')

    plt.subplot(424)
    plt.rc('axes', titlesize = 30)
    plt.rc('axes', labelsize = 25)
    plt.rc('xtick', labelsize = 22)
    plt.rc('ytick', labelsize = 22)
    #plt.ylim(ymin = 0, ymax = 21)
    plt.xlim(xmin = 1978, xmax = 2017)
    plt.plot(years, annual_means_July, '-b', years, np.poly1d(am_July_fit)(years), '-k', linewidth = 2)
    #plt.text(1980, 6, 'Linear Regression Slope = ' + str(round(am_July_fit[0], 6)), fontsize = 20)
    plt.ylabel('Wind Speed (m/s)')
    plt.xlabel('Year')
    plt.title('c.) July')

    plt.subplot(425)
    plt.rc('axes', titlesize = 30)
    plt.rc('axes', labelsize = 25)
    plt.rc('xtick', labelsize = 22)
    plt.rc('ytick', labelsize = 22)
    #plt.ylim(ymin = 0, ymax = 21)
    plt.xlim(xmin = 1978, xmax = 2017)
    plt.plot(years, annual_means_Aug, '-b', years, np.poly1d(am_Aug_fit)(years), '-k', linewidth = 2)
    #plt.text(1980, 6.2, 'Linear Regression Slope = ' + str(round(am_Aug_fit[0], 6)), fontsize = 20)
    plt.ylabel('Wind Speed (m/s)')
    plt.xlabel('Year')
    plt.title('d.) August')

    plt.subplot(426)
    plt.rc('axes', titlesize = 30)
    plt.rc('axes', labelsize = 25)
    plt.rc('xtick', labelsize = 22)
    plt.rc('ytick', labelsize = 22)
    #plt.ylim(ymin = 0, ymax = 21)
    plt.xlim(xmin = 1978, xmax = 2017)
    plt.plot(years, annual_means_Sep, '-b', years, np.poly1d(am_Sep_fit)(years), '-k', linewidth = 2)
    #plt.text(1995, 9, 'Linear Regression Slope = ' + str(round(am_Sep_fit[0], 6)), fontsize = 20)
    plt.ylabel('Wind Speed (m/s)')
    plt.xlabel('Year')
    plt.title('e.) September')

    plt.subplot(427)
    plt.rc('axes', titlesize = 30)
    plt.rc('axes', labelsize = 25)
    plt.rc('xtick', labelsize = 22)
    plt.rc('ytick', labelsize = 22)
    #plt.ylim(ymin = 0, ymax = 21)
    plt.xlim(xmin = 1978, xmax = 2017)
    plt.plot(years, annual_means_Oct, '-b', years, np.poly1d(am_Oct_fit)(years), '-k', linewidth = 2)
    #plt.text(1980, 7, 'Linear Regression Slope = ' + str(round(am_Oct_fit[0], 6)), fontsize = 20)
    plt.ylabel('Wind Speed (m/s)')
    plt.xlabel('Year')
    plt.title('f.) October')

    plt.subplot(428)
    plt.rc('axes', titlesize = 30)
    plt.rc('axes', labelsize = 25)
    plt.rc('xtick', labelsize = 22)
    plt.rc('ytick', labelsize = 22)
    #plt.ylim(ymin = 0, ymax = 21)
    plt.xlim(xmin = 1978, xmax = 2017)
    plt.plot(years, annual_means_Nov, '-b', years, np.poly1d(am_Nov_fit)(years), '-k', linewidth = 2)
    #plt.text(1980, 10, 'Linear Regression Slope = ' + str(round(am_Nov_fit[0], 6)), fontsize = 20)
    plt.ylabel('Wind Speed (m/s)')
    plt.xlabel('Year')
    plt.title('g.) November')

    plt.subplots_adjust(left = 0.125, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.25, hspace = 0.3)

    plt.savefig('Figures/Annual_Mean_Regressions_79-16' + location_names[region][0] + '.png')
    plt.show()

    # Plot all regression lines together
    '''plt.figure(1, figsize = (20,10))
    plt.rc('axes', titlesize = 30)
    plt.rc('axes', labelsize = 25)
    plt.rc('xtick', labelsize = 22)
    plt.rc('ytick', labelsize = 22)
    plt.plot(years, np.poly1d(am_fit)(years), label = 'Whole Season', linewidth = 2)
    plt.plot(years, np.poly1d(am_June_fit)(years), label = 'June')
    plt.plot(years, np.poly1d(am_July_fit)(years), label = 'July')
    plt.plot(years, np.poly1d(am_Aug_fit)(years), label = 'August')
    plt.plot(years, np.poly1d(am_Sep_fit)(years), label = 'September')
    plt.plot(years, np.poly1d(am_Oct_fit)(years), label = 'October')
    plt.plot(years, np.poly1d(am_Nov_fit)(years), label = 'November')
    plt.legend(loc = 0)
    plt.ylabel('Wind Speed (m/s)')
    plt.xlabel('Year')
    plt.title('Annual Means of Deep Layer Mean Wind Speeds During Months of the North Atlantic\nHurricane Season from 1979 through 2016 on ' +
              location_names[region][1])
    plt.savefig('Figures/Annual_Mean_Speeds/WS_79_16/Annual_Mean_All_Regression_Lines_79_16_' + location_names[region][0] + '.png')
    plt.show()'''


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
location_names['FL'] = ['FL_Coast', 'the Florida Peninsula', 'Locations on the Florida Peninsula']
location_names['NTX'] = ['NTX_Coast', 'the North Texas Coast', 'Locations Along the North Texas Coast']
location_names['STX'] = ['STX_Coast', 'the South Texas Coast', 'Locations Along the South Texas Coast']
location_names['TX'] = ['TX_Coast', 'the Texas Coast', 'Locations Along the Texas Coast']
location_names['LA-MS'] = ['LA-MS_Coast', 'Louisiana and Mississippi Coasts',
                           'Locations Along the Louisiana and Mississippi Coasts']
location_names['AL-FL'] = ['AL-FL_Pan_Coast', 'Alabama and Florida Panhandle Coasts',
                           'Locations Along the Alabama and Florida Panhandle Coasts']
location_names['WFL'] = ['West_FL_Coast', 'the West Florida Coast', 'Locations Along the West Florida Coast']
location_names['EFL'] = ['East_FL_Coast', 'the East Florida Coast', 'Locations Along the East Florida Coast']
location_names['NFL'] = ['North_FL_Coast', 'the North Florida Coast', 'Locations Along the North Florida Coast']
location_names['SFL'] = ['South_FL_Coast', 'the South Florida Coast', 'Locations Along the South Florida Coast']
location_names['GA-SC'] = ['GA-SC_Coast', 'Georgia and South Carolina Coasts',
                           'Locations Along the Georgia and South Carolina Coasts']
location_names['NC'] = ['NC_Coast', 'the North Carolina Coast', 'Locations Along the North Carolina Coast']

# Load data
data = np.load(sys.argv[1])

# Obtain the locations, times/dates, and the wind speeds from the data
locs = data['loc']
wind_speed_times = data['mydate']
wind_speeds = data['ts']

# Convert floats in the wind_speed_times array to datetime objects
dates_and_times = convertToDatetime(wind_speed_times)

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
                if locs[i][0] < -83.25: # or (locs[i][0] == -81.75 and locs[i][1] <= 28.5) or (locs[i][0] == -81 and locs[i][1] <= 26.25):
                    temp_locs.append(locs[i])
                    temp_wind_speeds.append(wind_speeds[i])
            elif region == 'FL':
                # Florida peninsula case
                if (locs[i][0] > -84 and locs[i][0] <= -79.5 and locs[i][1] <= 30.75):
                    temp_locs.append(locs[i])
                    temp_wind_speeds.append(wind_speeds[i])
            elif region == 'NTX':
                # North Texas coast case
                if locs[i][0] <= -94.5 and locs[i][1] >= 28.5:
                    temp_locs.append(locs[i])
                    temp_wind_speeds.append(wind_speeds[i])
            elif region == 'STX':
                # South Texas coast case
                if locs[i][0] <= -94.5 and locs[i][1] < 28.5:
                    temp_locs.append(locs[i])
                    temp_wind_speeds.append(wind_speeds[i])
            elif region == 'TX':
                # Texas coast case
                if locs[i][0] <= -94.5:
                    temp_locs.append(locs[i])
                    temp_wind_speeds.append(wind_speeds[i])
            elif region == 'LA-MS':
                # Louisiana/Mississippi coast case
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
            elif region == 'NFL':
                # North Florida coast case
                if (locs[i][0] > -84 and locs[i][0] <= -79.5) and (locs[i][1] <= 30.75 and locs[i][1] > 27):
                    temp_locs.append(locs[i])
                    temp_wind_speeds.append(wind_speeds[i])
            elif region == 'SFL':
                # South Florida coast case
                if (locs[i][0] > -84 and locs[i][0] <= -79.5 and locs[i][1] <= 27):
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

# Determine the maximum wind speed
max_wind_speed = int(math.ceil(np.amax(wind_speeds)))

# Or set a predefined max to disregard larger wind speeds
# Comment this line out if you don't want any maximum limit on what values to plot
max_wind_speed = 45

# Read in hurricane best track data
best_track_speeds, best_track_speeds_79_97, best_track_speeds_98_16 = getBestTrackSpeeds('best_tracks.txt', locs)
times_to_remove, locs_to_remove = readBestTracks('best_tracks.txt', locs)

# Remove DLM winds contaminated with tropical cyclone winds
wind_speeds = removeTCWinds(wind_speeds, dates_and_times, locs, times_to_remove, locs_to_remove)

# Plot deep layer mean weights
p_levels = [100, 150, 200, 250, 300, 400, 500, 700, 850, 1000]
weights = [(float(25)/float(900)) * 100, (float(50)/float(900)) * 100, (float(50)/float(900)) * 100,
           (float(50)/float(900)) * 100, (float(75)/float(900)) * 100, (float(100)/float(900)) * 100,
           (float(150)/float(900)) * 100, (float(175)/float(900)) * 100, (float(150)/float(900)) * 100,
           (float(75)/float(900)) * 100]
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(6, 6)
ax.plot(weights, p_levels, linewidth = 2)
ax.tick_params(labelsize = 12)
ax.set_xlim(0, 20)
ax.set_ylim(1000, 100)
ax.set_yscale('log')
ax.set_yticks([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_ylabel('Pressure Level (mb)', fontsize = 17)
ax.set_xlabel('Weight (%)', fontsize = 17)
#ax.set_title('Neumann (1988) Deep Layer Mean Weighting Scheme', fontsize = 17)
fig.savefig('Figures/DLM_weights.png')
plt.show()

######################################################################################
##                         All Locations, Whole Time Period                         ##
######################################################################################

# Analyze all data together, regardless of time or location

# Create an array to store the frequency of different wind speed values (at all
# locations). The zeroeth index of the array will store the frequency of wind
# speeds between 0 m/s and 1 m/s; the first index of the array will store the frequency
# of wind speeds between 1 m/s and 2 m/s; etc...
wind_speed_freq_all = [0] * max_wind_speed
# Do the same for translation speeds
trans_speed_freq_all = [0] * max_wind_speed

# Create wind speed array that will only contain data for the desired period of time
wind_speeds_79_16 = getWindSpeedInterval(wind_speeds, 0, 27816)

# Get updated time interval as well
dates_and_times_79_16 = dates_and_times[0:27816]

# Determine and plot annual mean DLM wind speeds from 1979 to 2016
getAnnualMeans(wind_speeds_79_16, location_names)

# Populate the wind speed frequency array
getFrequencies(wind_speeds_79_16, wind_speed_freq_all)
# Alternatively, populate the frequency array using best track translation speeds
getBestTrackFrequencies(best_track_speeds, trans_speed_freq_all)

current_year = 2005
current_year_index = current_year - 1979

# Average the wind speed between all locations at each time
avg_wind_speeds_79_16 = averageWindsAmongAllPoints(wind_speeds_79_16)

# Plot averaged wind speeds over the course of a year
'''plt.figure(1, figsize = (20,10))
plt.rc('axes', titlesize = 30)
plt.rc('axes', labelsize = 25)
plt.rc('xtick', labelsize = 22)
plt.rc('ytick', labelsize = 22)
plt.plot(dates_and_times_79_16[(current_year_index * 732):(732 * (current_year_index + 1))], avg_wind_speeds_79_16[(current_year_index * 732):(732 * (current_year_index + 1))])
plt.ylabel('Wind Speed (m/s)')
plt.xlabel('Time')
plt.title('Average Wind Speeds During the ' + str(current_year) + ' Hurricane Season on ' +
          location_names[region][1])
#plt.savefig('Figures/Time_Series/' + str(current_year) + '/Filtered_' + location_names[region][0]  + '_Time_Series_' + str(current_year) + '.png')
#plt.savefig('Figures/Time_Series/Yearly/Filtered_' + location_names[region][0]  + '_Time_Series_' + str(current_year) + '.png')
plt.show()'''

# Normalize wind speed frequencies
norm_wind_speed_freq_all = normalizeWindSpeeds(wind_speed_freq_all)
# And translation speed frequencies
norm_trans_speed_freq_all = normalizeWindSpeeds(trans_speed_freq_all)

# Generate a histogram to show the frequency distribution for the wind speeds at all
# specified locations from 1979 to 2017
'''plt.figure(1, figsize = (20,10))
#plt.bar(np.arange(len(wind_speed_freq_all)), wind_speed_freq_all)
plt.bar(np.arange(len(norm_wind_speed_freq_all)), norm_wind_speed_freq_all)
plt.xlim(xmax = max_wind_speed + 1)
plt.ylabel('Frequency (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nDuring ' +
          'North Atlantic Hurricane Seasons from 1979 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_Histogram.png')
plt.savefig('Figures/WS_79-16/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_Histogram.png')
plt.show()

# Generate a histogram to show the frequency distribution for the tropical cyclone
# translation speeds near the region fo interest from 1979 to 2017
plt.figure(1, figsize = (20,10))
#plt.bar(np.arange(len(trans_speed_freq_all)), trans_speed_freq_all)
plt.bar(np.arange(len(norm_trans_speed_freq_all)), norm_trans_speed_freq_all)
plt.xlim(xmax = max_wind_speed + 1)
plt.ylabel('Frequency (%)')
plt.xlabel('Translation Speed (m/s)')
plt.title('Normalized Frequency of Tropical Cyclone Translation Speeds Recorded\nnear ' + location_names[region][1] + ' During ' +
          'North Atlantic Hurricane Seasons\nfrom 1979 up to 2017')
plt.savefig('Figures/Translation_Speeds/' + location_names[region][0]  + '/Filtered_Norm_Trans_Speeds_' + location_names[region][0] + '_79-16_Histogram.png')
plt.savefig('Figures/Translation_Speeds/TS_79-16/Filtered_Norm_Trans_Speeds_' + location_names[region][0] + '_79-16_Histogram.png')
plt.show()

# Create arrays to store the frequency of different wind speed values (at all
# locations). The zeroeth index of the array will store the frequency of wind
# speeds between 0 m/s and 1 m/s; the first index of the array will store the frequency
# of wind speeds between 1 m/s and 2 m/s; etc...
wind_speed_freq_all_EHS = [0] * max_wind_speed
wind_speed_freq_all_MHS = [0] * max_wind_speed
wind_speed_freq_all_LHS = [0] * max_wind_speed
wind_speed_freq_all_MHS_v2 = [0] * max_wind_speed
wind_speed_freq_all_LHS_v2 = [0] * max_wind_speed
wind_speed_freq_all_June = [0] * max_wind_speed
wind_speed_freq_all_July = [0] * max_wind_speed
wind_speed_freq_all_Aug = [0] * max_wind_speed
wind_speed_freq_all_Sep = [0] * max_wind_speed
wind_speed_freq_all_Oct = [0] * max_wind_speed
wind_speed_freq_all_Nov = [0] * max_wind_speed

# Divide the wind speed data up by which part of the hurricane season they were from
# (early hurricane season (June, July), mid  hurricane season (August, September), and
# late hurricane season (October, November)). Also get data for each month
# individually
wind_speeds_79_16_EHS, wind_speeds_79_16_MHS, wind_speeds_79_16_MHS_v2, wind_speeds_79_16_LHS, wind_speeds_79_16_LHS_v2, \
wind_speeds_79_16_June, wind_speeds_79_16_July, wind_speeds_79_16_Aug, wind_speeds_79_16_Sep, wind_speeds_79_16_Oct, \
wind_speeds_79_16_Nov = divideBySeason(wind_speeds_79_16)

# Populate the wind speed frequency arrays for each of the three parts of the hurricane season
# (and each month)
getFrequencies(wind_speeds_79_16_EHS, wind_speed_freq_all_EHS)
getFrequencies(wind_speeds_79_16_MHS, wind_speed_freq_all_MHS)
getFrequencies(wind_speeds_79_16_LHS, wind_speed_freq_all_LHS)
getFrequencies(wind_speeds_79_16_MHS_v2, wind_speed_freq_all_MHS_v2)
getFrequencies(wind_speeds_79_16_LHS_v2, wind_speed_freq_all_LHS_v2)
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
norm_wind_speed_freq_all_MHS_v2 = normalizeWindSpeeds(wind_speed_freq_all_MHS_v2)
norm_wind_speed_freq_all_LHS_v2 = normalizeWindSpeeds(wind_speed_freq_all_LHS_v2)
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
plt.ylabel('Frequency (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nDuring ' +
          'August and September from 1979 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_MHS_Histogram.png')
plt.savefig('Figures/WS_79-16_MHS/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_MHS_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,10))
plt.bar(np.arange(len(norm_wind_speed_freq_all_MHS_v2)), norm_wind_speed_freq_all_MHS_v2)
plt.xlim(xmax = max_wind_speed + 1)
plt.ylabel('Frequency (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nDuring ' +
          'July and August from 1979 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_MHS_v2_Histogram.png')
plt.savefig('Figures/WS_79-16_MHS_v2/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_MHS_v2_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,10))
plt.bar(np.arange(len(norm_wind_speed_freq_all_Aug)), norm_wind_speed_freq_all_Aug)
plt.xlim(xmax = max_wind_speed + 1)
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
plt.ylabel('Frequency (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nDuring ' +
          'October and November from 1979 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_LHS_Histogram.png')
plt.savefig('Figures/WS_79-16_LHS/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_LHS_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,10))
plt.bar(np.arange(len(norm_wind_speed_freq_all_LHS_v2)), norm_wind_speed_freq_all_LHS_v2)
plt.xlim(xmax = max_wind_speed + 1)
plt.ylabel('Frequency (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nDuring ' +
          'September and October from 1979 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_LHS_v2_Histogram.png')
plt.savefig('Figures/WS_79-16_LHS_v2/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_LHS_v2_Histogram.png')
plt.show()

plt.figure(1, figsize = (20,10))
plt.bar(np.arange(len(norm_wind_speed_freq_all_Oct)), norm_wind_speed_freq_all_Oct)
plt.xlim(xmax = max_wind_speed + 1)
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
plt.ylabel('Frequency (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nDuring ' +
          'the Month of November from 1979 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_Nov_Histogram.png')
plt.savefig('Figures/WS_79-16_Nov/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-16_Nov_Histogram.png')
plt.show()'''

######################################################################################
##                        All Locations, Halved Time Periods                        ##
######################################################################################

# Split the data into two time periods: One from 1979 up to 1998, and the other from 1998
# up to 2017

# Create arrays to store the frequency of different wind speed values (at all locations
# for each time period. The zeroeth index of the array will store the frequency of
# wind speeds between 0 m/s and 1 m/s; the first index of the array will store the frequency
# of wind speeds between 1 m/s and 2 m/s; etc...
wind_speed_freq_79_97 = [0] * max_wind_speed
wind_speed_freq_98_16 = [0] * max_wind_speed
# Do the same for translation speeds
trans_speed_freq_79_97 = [0] * max_wind_speed
trans_speed_freq_98_16 = [0] * max_wind_speed

# Create wind speed arrays that will only contain data for the desired period of time
wind_speeds_79_97 = getWindSpeedInterval(wind_speeds, 0, 13908)
wind_speeds_98_16 = getWindSpeedInterval(wind_speeds, 13908, 27816)

# Get updated time intervals as well
dates_and_times_79_97 = dates_and_times[0:13908]
dates_and_times_98_16 = dates_and_times[13908:27816]

# Populate the wind speed frequency arrays
getFrequencies(wind_speeds_79_97, wind_speed_freq_79_97)
getFrequencies(wind_speeds_98_16, wind_speed_freq_98_16)
# And translation speed frequency arrays
getBestTrackFrequencies(best_track_speeds_79_97, trans_speed_freq_79_97)
getBestTrackFrequencies(best_track_speeds_98_16, trans_speed_freq_98_16)

# Normalize wind speed frequencies
norm_wind_speed_freq_79_97 = normalizeWindSpeeds(wind_speed_freq_79_97)
norm_wind_speed_freq_98_16 = normalizeWindSpeeds(wind_speed_freq_98_16)
# Do the same for translation speeds
norm_trans_speed_freq_79_97 = normalizeWindSpeeds(trans_speed_freq_79_97)
norm_trans_speed_freq_98_16 = normalizeWindSpeeds(trans_speed_freq_98_16)

# Average the wind speed at each time
avg_wind_speeds_79_97, dates_and_times_1y = averageWindsEachTime(wind_speeds_79_97, dates_and_times_79_97, 12)
avg_wind_speeds_98_16, dates_and_times_1y = averageWindsEachTime(wind_speeds_98_16, dates_and_times_79_97, 12)

# Plot the averaged wind speeds
fig, ax = plt.subplots(1, figsize = (20,10))
ax.tick_params(labelsize = 26)
ax.plot(dates_and_times_1y, avg_wind_speeds_79_97, 'b-', linewidth = 2, label = '1979-1997')
ax.plot(dates_and_times_1y, avg_wind_speeds_98_16, 'r-', linewidth = 2, label = '1998-2016')
ax.set_ylim(ymin = -0.3)
ax.set_ylabel('Wind Speed (m/s)', fontsize = 35)
ax.set_xlabel('Time', fontsize = 35)
#plt.title('Three Day Average Wind Speeds During Hurricane Seasons from 1979 through 1997\n' +
#          'Compared to Hurricane Seasons from 1998 through 2016 on ' + location_names[region][1])
#ax.set_title('Average Wind Speeds During North Atlantic Hurricane Seasons', fontsize = 30)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%B'))
ax.legend(loc = 2, fontsize = 30)
fig.savefig('Figures/Time_Series/Yearly/Filtered_' + location_names[region][0]  + '_Time_Series_3d_Avg_Comparison.png')
plt.show()

# Determine the difference in the two averaged time series arrays
i = 0
avg_wind_speeds_diff = [0] * len(avg_wind_speeds_79_97)
while i < len(avg_wind_speeds_79_97):
    avg_wind_speeds_diff[i] = avg_wind_speeds_98_16[i] - avg_wind_speeds_79_97[i]
    i = i + 1

# Plot the difference of the averaged wind speeds
'''plt.figure(1, figsize = (20,10))
plt.rc('axes', titlesize = 30)
plt.rc('axes', labelsize = 25)
plt.rc('xtick', labelsize = 22)
plt.rc('ytick', labelsize = 22)
plt.plot(dates_and_times_1y, avg_wind_speeds_diff)
plt.plot(dates_and_times_1y, [0] * len(dates_and_times_1y))
plt.ylabel('Wind Speed (m/s)')
plt.xlabel('Time')
plt.title('Difference Between Three Day Average Wind Speeds During Hurricane Seasons from 1979 through 1997\n' +
          'and Hurricane Seasons from 1998 through 2016 on ' + location_names[region][1])
plt.savefig('Figures/Time_Series/Yearly/Filtered_' + location_names[region][0]  + '_Time_Series_3d_Avg_Comparison_Diff.png')
plt.show()

# Generate histograms to show the frequency distribution for the wind speeds at all
# locations from 1979 up to 1998 and from 1998 up to 2017
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

# Generate histograms to show the frequency distribution for the translation speeds
# near the region of interest from 1979 up to 1998 and from 1998 up to 2017
plt.figure(1, figsize = (20,10))
plt.subplot(211)
plt.bar(np.arange(len(norm_trans_speed_freq_79_97)), norm_trans_speed_freq_79_97)
plt.xlim(xmax = max_wind_speed + 1)
#plt.ylim(ymax = 250000)
plt.ylabel('Frequency (%)')
plt.xlabel('Translation Speed (m/s)')
plt.title('Normalized Frequency of Tropical Cyclone Translation Speeds Recorded\nnear ' + location_names[region][1] + ' During ' +
          'Hurricane Seasons from 1979 up to 1998')

plt.subplot(212)
plt.bar(np.arange(len(norm_trans_speed_freq_98_16)), norm_trans_speed_freq_98_16)
plt.xlim(xmax = max_wind_speed + 1)
#plt.ylim(ymax = 250000)
plt.ylabel('Frequency (%)')
plt.xlabel('Translation Speed (m/s)')
plt.title('Normalized Frequency of Tropical Cyclone Translation Speeds Recorded\nnear ' + location_names[region][1] + 'During ' +
          'Hurricane Seasons from 1998 up to 2017')

plt.subplots_adjust(left = 0.125, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.2,
                    hspace = 0.5)
plt.savefig('Figures/Translation_Speeds/' + location_names[region][0]  + '/Filtered_Norm_Trans_Speeds_' + location_names[region][0] + '_79-97_98-16_Histogram.png')
plt.savefig('Figures/Translation_Speeds/WS_79-97_98-16/Filtered_Norm_Trans_Speeds_' + location_names[region][0] + '_79-97_98-16_Histogram.png')
plt.show()'''

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

# Do the same for transation speeds
'''fig, ax = plt.subplots(figsize = (20,15))
bar_width = 0.35
ax.bar(np.arange(len(norm_trans_speed_freq_79_97)), norm_trans_speed_freq_79_97, bar_width,
                color='b', label='1979-1997')
ax.bar(np.arange(len(norm_trans_speed_freq_98_16)) + bar_width, norm_trans_speed_freq_98_16,
                bar_width, color='r', label='1998-2016')

ax.set_xlim(xmax = max_wind_speed + 1)
ax.set_xlabel('Translation Speed (m/s)')
ax.set_ylabel('Frequency (%)')
ax.set_title('Normalized Frequency of Tropical Cyclone Translation Speeds Recorded\nnear ' + location_names[region][1] + ' During ' +
             'Hurricane Seasons from 1979 up to 1998\nvs. Hurricane Seasons from ' +
             '1998 up to 2017')

ax.legend()
fig.savefig('Figures/Translation_Speeds/' + location_names[region][0]  + '/Filtered_Norm_Trans_Speeds_' + location_names[region][0] + '_79-97_98-16_SBS_Histogram.png')
fig.savefig('Figures/Translation_Speeds/WS_79-97_98-16_SBS/Filtered_Norm_Trans_Speeds_' + location_names[region][0] + '_79-97_98-16_SBS_Histogram.png')
plt.show()'''

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
'''plt.figure(1, figsize = (20,15))
plt.bar(np.arange(len(norm_wind_speed_freq_diff)), norm_wind_speed_freq_diff)
plt.xlim(xmax = max_wind_speed + 1)
plt.ylabel('Frequency Difference (%)')
plt.xlabel('Wind Speed (m/s)')
#plt.title('Difference Between Normalized Frequency of Wind Speeds Recorded on\n' + location_names[region][1] +
#          ' During Hurricane Seasons from 1979 up to\n1998 and During Hurricane ' +
#          'Seasons from 1998 up to 2017')
plt.title('Difference Between Normalized Frequencies of Wind Speeds')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_Histogram.png')
plt.savefig('Figures/WS_Diff_79-97_98-16/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_Histogram.png')
plt.show()

# Determine differences in the translation speed frequencies between the two time periods
i = 0
trans_speed_freq_diff = [0] * len(trans_speed_freq_79_97)
norm_trans_speed_freq_diff = [0] * len(norm_trans_speed_freq_79_97)
while i < len(trans_speed_freq_79_97):
    trans_speed_freq_diff[i] = trans_speed_freq_98_16[i] - trans_speed_freq_79_97[i]
    norm_trans_speed_freq_diff[i] = norm_trans_speed_freq_98_16[i] - norm_trans_speed_freq_79_97[i]
    i = i + 1

# Create a histogram that shows the difference between the wind speed frequencies for
# the two time periods
plt.figure(1, figsize = (20,15))
plt.bar(np.arange(len(norm_trans_speed_freq_diff)), norm_trans_speed_freq_diff)
plt.xlim(xmax = max_wind_speed + 1)
plt.ylabel('Frequency Difference (%)')
plt.xlabel('Translation Speed (m/s)')
plt.title('Difference Between Normalized Frequencies of Tropical Cyclone Translation Speeds')
plt.savefig('Figures/Translation_Speeds/' + location_names[region][0]  + '/Filtered_Norm_Trans_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_Histogram.png')
plt.savefig('Figures/Translation_Speeds/WS_Diff_79-97_98-16/Filtered_Norm_Trans_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_Histogram.png')
plt.show()'''


# Create arrays to store the frequency of different wind speed values (at all
# locations). The zeroeth index of the array will store the frequency of wind
# speeds between 0 m/s and 1 m/s; the first index of the array will store the frequency
# of wind speeds between 1 m/s and 2 m/s; etc...
wind_speed_freq_79_97_EHS = [0] * max_wind_speed
wind_speed_freq_79_97_MHS = [0] * max_wind_speed
wind_speed_freq_79_97_LHS = [0] * max_wind_speed
wind_speed_freq_79_97_MHS_v2 = [0] * max_wind_speed
wind_speed_freq_79_97_LHS_v2 = [0] * max_wind_speed
wind_speed_freq_79_97_June = [0] * max_wind_speed
wind_speed_freq_79_97_July = [0] * max_wind_speed
wind_speed_freq_79_97_Aug = [0] * max_wind_speed
wind_speed_freq_79_97_Sep = [0] * max_wind_speed
wind_speed_freq_79_97_Oct = [0] * max_wind_speed
wind_speed_freq_79_97_Nov = [0] * max_wind_speed
wind_speed_freq_98_16_EHS = [0] * max_wind_speed
wind_speed_freq_98_16_MHS = [0] * max_wind_speed
wind_speed_freq_98_16_LHS = [0] * max_wind_speed
wind_speed_freq_98_16_MHS_v2 = [0] * max_wind_speed
wind_speed_freq_98_16_LHS_v2 = [0] * max_wind_speed
wind_speed_freq_98_16_June = [0] * max_wind_speed
wind_speed_freq_98_16_July = [0] * max_wind_speed
wind_speed_freq_98_16_Aug = [0] * max_wind_speed
wind_speed_freq_98_16_Sep = [0] * max_wind_speed
wind_speed_freq_98_16_Oct = [0] * max_wind_speed
wind_speed_freq_98_16_Nov = [0] * max_wind_speed

# Divide the wind speed data up by which part of the hurricane season they were from
# (early hurricane season (June, July), mid hurricane season (August, September), and
# late hurricane season (October, November)). Also get data for each month individually
wind_speeds_79_97_EHS, wind_speeds_79_97_MHS, wind_speeds_79_97_MHS_v2, wind_speeds_79_97_LHS, wind_speeds_79_97_LHS_v2, wind_speeds_79_97_June, \
wind_speeds_79_97_July, wind_speeds_79_97_Aug, wind_speeds_79_97_Sep, wind_speeds_79_97_Oct, wind_speeds_79_97_Nov = divideBySeason(wind_speeds_79_97)
wind_speeds_98_16_EHS, wind_speeds_98_16_MHS, wind_speeds_98_16_MHS_v2, wind_speeds_98_16_LHS, wind_speeds_98_16_LHS_v2, wind_speeds_98_16_June, \
wind_speeds_98_16_July, wind_speeds_98_16_Aug, wind_speeds_98_16_Sep, wind_speeds_98_16_Oct, wind_speeds_98_16_Nov = divideBySeason(wind_speeds_98_16)

# Populate the wind speed frequency arrays for each of the three parts of the hurricane
# season for each time period (and each month)
getFrequencies(wind_speeds_79_97_EHS, wind_speed_freq_79_97_EHS)
getFrequencies(wind_speeds_79_97_MHS, wind_speed_freq_79_97_MHS)
getFrequencies(wind_speeds_79_97_LHS, wind_speed_freq_79_97_LHS)
getFrequencies(wind_speeds_79_97_MHS_v2, wind_speed_freq_79_97_MHS_v2)
getFrequencies(wind_speeds_79_97_LHS_v2, wind_speed_freq_79_97_LHS_v2)
getFrequencies(wind_speeds_79_97_June, wind_speed_freq_79_97_June)
getFrequencies(wind_speeds_79_97_July, wind_speed_freq_79_97_July)
getFrequencies(wind_speeds_79_97_Aug, wind_speed_freq_79_97_Aug)
getFrequencies(wind_speeds_79_97_Sep, wind_speed_freq_79_97_Sep)
getFrequencies(wind_speeds_79_97_Oct, wind_speed_freq_79_97_Oct)
getFrequencies(wind_speeds_79_97_Nov, wind_speed_freq_79_97_Nov)
getFrequencies(wind_speeds_98_16_EHS, wind_speed_freq_98_16_EHS)
getFrequencies(wind_speeds_98_16_MHS, wind_speed_freq_98_16_MHS)
getFrequencies(wind_speeds_98_16_LHS, wind_speed_freq_98_16_LHS)
getFrequencies(wind_speeds_98_16_MHS_v2, wind_speed_freq_98_16_MHS_v2)
getFrequencies(wind_speeds_98_16_LHS_v2, wind_speed_freq_98_16_LHS_v2)
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
norm_wind_speed_freq_79_97_MHS_v2 = normalizeWindSpeeds(wind_speed_freq_79_97_MHS_v2)
norm_wind_speed_freq_98_16_MHS_v2 = normalizeWindSpeeds(wind_speed_freq_98_16_MHS_v2)
norm_wind_speed_freq_79_97_LHS_v2 = normalizeWindSpeeds(wind_speed_freq_79_97_LHS_v2)
norm_wind_speed_freq_98_16_LHS_v2 = normalizeWindSpeeds(wind_speed_freq_98_16_LHS_v2)
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
'''fig, ax = plt.subplots(figsize = (20,15))
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

fig, ax = plt.subplots(figsize = (20,15))
bar_width = 0.35
ax.bar(np.arange(len(norm_wind_speed_freq_79_97_MHS_v2)), norm_wind_speed_freq_79_97_MHS_v2, bar_width,
                color='b', label='1979-1997')
ax.bar(np.arange(len(norm_wind_speed_freq_98_16_MHS_v2)) + bar_width, norm_wind_speed_freq_98_16_MHS_v2,
                bar_width, color='r', label='1998-2016')
ax.set_xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    ax.set_ylim(ymax = 120000)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Frequency (%)')
ax.set_title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + ' in\nJuly ' +
             'and August During Hurricane Seasons from 1979 up to\n1998 vs. Hurricane ' +
             'Seasons from 1998 up to 2017')
ax.legend()
fig.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-97_98-16_MHS_v2_SBS_Histogram.png')
fig.savefig('Figures/WS_79-97_98-16_MHS_v2_SBS/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-97_98-16_MHS_v2_SBS_Histogram.png')
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

fig, ax = plt.subplots(figsize = (20,15))
bar_width = 0.35
ax.bar(np.arange(len(norm_wind_speed_freq_79_97_LHS_v2)), norm_wind_speed_freq_79_97_LHS_v2, bar_width,
                color='b', label='1979-1997')
ax.bar(np.arange(len(norm_wind_speed_freq_98_16_LHS_v2)) + bar_width, norm_wind_speed_freq_98_16_LHS_v2,
                bar_width, color='r', label='1998-2016')
ax.set_xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    ax.set_ylim(ymax = 120000)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Frequency (%)')
ax.set_title('Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + ' in\nSeptember ' +
             'and October During Hurricane Seasons from 1979 up to\n1998 vs. Hurricane ' +
             'Seasons from 1998 up to 2017')
ax.legend()
fig.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-97_98-16_LHS_v2_SBS_Histogram.png')
fig.savefig('Figures/WS_79-97_98-16_LHS_v2_SBS/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-97_98-16_LHS_v2_SBS_Histogram.png')
plt.show()'''

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
wind_speed_freq_MHS_v2_diff = [0] * len(wind_speed_freq_79_97_MHS_v2)
norm_wind_speed_freq_MHS_v2_diff = [0] * len(norm_wind_speed_freq_79_97_MHS_v2)
while i < len(wind_speed_freq_79_97_MHS_v2):
    wind_speed_freq_MHS_v2_diff[i] = wind_speed_freq_98_16_MHS_v2[i] - wind_speed_freq_79_97_MHS_v2[i]
    norm_wind_speed_freq_MHS_v2_diff[i] = norm_wind_speed_freq_98_16_MHS_v2[i] - norm_wind_speed_freq_79_97_MHS_v2[i]
    i = i + 1

i = 0
wind_speed_freq_LHS_v2_diff = [0] * len(wind_speed_freq_79_97_LHS_v2)
norm_wind_speed_freq_LHS_v2_diff = [0] * len(norm_wind_speed_freq_79_97_LHS_v2)
while i < len(wind_speed_freq_79_97_LHS_v2):
    wind_speed_freq_LHS_v2_diff[i] = wind_speed_freq_98_16_LHS_v2[i] - wind_speed_freq_79_97_LHS_v2[i]
    norm_wind_speed_freq_LHS_v2_diff[i] = norm_wind_speed_freq_98_16_LHS_v2[i] - norm_wind_speed_freq_79_97_LHS_v2[i]
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
'''plt.figure(1, figsize = (20,15))
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
plt.bar(np.arange(len(norm_wind_speed_freq_MHS_v2_diff)), norm_wind_speed_freq_MHS_v2_diff)
plt.xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    plt.ylim(ymax = 9000, ymin = -7000)
plt.ylabel('Frequency Difference (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Difference Between Normalized Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nin July ' +
          'and August During Hurricane Seasons from 1979 up to 1998\nand During Hurricane ' +
          'Seasons from 1998 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_MHS_v2_Histogram.png')
plt.savefig('Figures/WS_Diff_79-97_98-16_MHS_v2/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_MHS_v2_Histogram.png')
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
plt.bar(np.arange(len(norm_wind_speed_freq_LHS_v2_diff)), norm_wind_speed_freq_LHS_v2_diff)
plt.xlim(xmax = max_wind_speed + 1)
#if region == 'AL':
#    plt.ylim(ymax = 9000, ymin = -7000)
plt.ylabel('Frequency Difference (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('Difference Between Frequency of Wind Speeds Recorded on ' + location_names[region][1] + '\nin September ' +
          'and October During Hurricane Seasons from 1979 up to 1998\nand During Hurricane ' +
          'Seasons from 1998 up to 2017')
plt.savefig('Figures/' + location_names[region][0]  + '/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_LHS_v2_Histogram.png')
plt.savefig('Figures/WS_Diff_79-97_98-16_LHS_v2/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_Diff_Between_79-97_and_98-16_LHS_v2_Histogram.png')
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
plt.show()'''

# Generate subplots of side-by-side frequency differences for June, July, August, September, October, and November
# Also add subplot for whole season
fig, axes = plt.subplots(4, 2, figsize = (30,30))
bar_width = 0.35

axes[0,0] = plt.subplot2grid((4,8), (0,2), colspan = 4)
axes[0,0].bar(np.arange(len(norm_wind_speed_freq_79_97)), norm_wind_speed_freq_79_97, bar_width,
              color='b', label='1979-1997')
axes[0,0].bar(np.arange(len(norm_wind_speed_freq_98_16)) + bar_width, norm_wind_speed_freq_98_16,
                bar_width, color='r', label='1998-2016')
axes[0,0].set_xlim(xmax = max_wind_speed + 1)
axes[0,0].set_ylim(ymax = 17)
axes[0,0].set_xlabel('Wind Speed (m/s)')
axes[0,0].set_ylabel('Frequency (%)')
axes[0,0].set_title('a.) Whole Hurricane Season')
axes[0,0].legend(fontsize = 30)

axes[1,0].bar(np.arange(len(norm_wind_speed_freq_79_97_June)), norm_wind_speed_freq_79_97_June, bar_width,
              color='b', label='1979-1997')
axes[1,0].bar(np.arange(len(norm_wind_speed_freq_98_16_June)) + bar_width, norm_wind_speed_freq_98_16_June,
                bar_width, color='r', label='1998-2016')
axes[1,0].set_xlim(xmax = max_wind_speed + 1)
axes[1,0].set_ylim(ymax = 17)
axes[1,0].set_xlabel('Wind Speed (m/s)')
axes[1,0].set_ylabel('Frequency (%)')
axes[1,0].set_title('b.) June')
axes[1,0].legend(fontsize = 30)

axes[1,1].bar(np.arange(len(norm_wind_speed_freq_79_97_July)), norm_wind_speed_freq_79_97_July, bar_width,
              color='b', label='1979-1997')
axes[1,1].bar(np.arange(len(norm_wind_speed_freq_98_16_July)) + bar_width, norm_wind_speed_freq_98_16_July,
                bar_width, color='r', label='1998-2016')
axes[1,1].set_xlim(xmax = max_wind_speed + 1)
axes[1,1].set_ylim(ymax = 17)
axes[1,1].set_xlabel('Wind Speed (m/s)')
axes[1,1].set_ylabel('Frequency (%)')
axes[1,1].set_title('c.) July')
axes[1,1].legend(fontsize = 30)

axes[2,0].bar(np.arange(len(norm_wind_speed_freq_79_97_Aug)), norm_wind_speed_freq_79_97_Aug, bar_width,
              color='b', label='1979-1997')
axes[2,0].bar(np.arange(len(norm_wind_speed_freq_98_16_Aug)) + bar_width, norm_wind_speed_freq_98_16_Aug,
                bar_width, color='r', label='1998-2016')
axes[2,0].set_xlim(xmax = max_wind_speed + 1)
axes[2,0].set_ylim(ymax = 17)
axes[2,0].set_xlabel('Wind Speed (m/s)')
axes[2,0].set_ylabel('Frequency (%)')
axes[2,0].set_title('d.) August')
axes[2,0].legend(fontsize = 30)

axes[2,1].bar(np.arange(len(norm_wind_speed_freq_79_97_Sep)), norm_wind_speed_freq_79_97_Sep, bar_width,
              color='b', label='1979-1997')
axes[2,1].bar(np.arange(len(norm_wind_speed_freq_98_16_Sep)) + bar_width, norm_wind_speed_freq_98_16_Sep,
                bar_width, color='r', label='1998-2016')
axes[2,1].set_xlim(xmax = max_wind_speed + 1)
axes[2,1].set_ylim(ymax = 17)
axes[2,1].set_xlabel('Wind Speed (m/s)')
axes[2,1].set_ylabel('Frequency (%)')
axes[2,1].set_title('e.) September')
axes[2,1].legend(fontsize = 30)

axes[3,0].bar(np.arange(len(norm_wind_speed_freq_79_97_Oct)), norm_wind_speed_freq_79_97_Oct, bar_width,
              color='b', label='1979-1997')
axes[3,0].bar(np.arange(len(norm_wind_speed_freq_98_16_Oct)) + bar_width, norm_wind_speed_freq_98_16_Oct,
                bar_width, color='r', label='1998-2016')
axes[3,0].set_xlim(xmax = max_wind_speed + 1)
axes[3,0].set_ylim(ymax = 17)
axes[3,0].set_xlabel('Wind Speed (m/s)')
axes[3,0].set_ylabel('Frequency (%)')
axes[3,0].set_title('f.) October')
axes[3,0].legend(fontsize = 30)

axes[3,1].bar(np.arange(len(norm_wind_speed_freq_79_97_Nov)), norm_wind_speed_freq_79_97_Nov, bar_width,
              color='b', label='1979-1997')
axes[3,1].bar(np.arange(len(norm_wind_speed_freq_98_16_Nov)) + bar_width, norm_wind_speed_freq_98_16_Nov,
                bar_width, color='r', label='1998-2016')
axes[3,1].set_xlim(xmax = max_wind_speed + 1)
axes[3,1].set_ylim(ymax = 17)
axes[3,1].set_xlabel('Wind Speed (m/s)')
axes[3,1].set_ylabel('Frequency (%)')
axes[3,1].set_title('g.) November')
axes[3,1].legend(fontsize = 30)

plt.subplots_adjust(left = 0.125, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.25, hspace = 0.3)

fig.savefig('Figures/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-97_98-16_SBS_Histogram_Plots.png')
plt.show()

# Generate subplots of frequency differences for June, July, August, September, October, and November
# Also add subplot for whole season
plt.figure(1, figsize = (30,30))
plt.subplot2grid((4,8), (0,2), colspan = 4)
plt.bar(np.arange(len(norm_wind_speed_freq_diff)), norm_wind_speed_freq_diff)
plt.xlim(xmax = max_wind_speed + 1)
if region == 'AL':
    plt.ylim(ymax = 2.5, ymin = -1.7)
plt.ylabel('Frequency Difference (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('a.) Whole Hurricane Season')

plt.figure(1, figsize = (30,30))
plt.subplot(423)
plt.bar(np.arange(len(norm_wind_speed_freq_June_diff)), norm_wind_speed_freq_June_diff)
plt.xlim(xmax = max_wind_speed + 1)
if region == 'AL':
    plt.ylim(ymax = 2.5, ymin = -1.7)
plt.ylabel('Frequency Difference (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('b.) June')
#plt.suptitle('Difference Between Normalized Frequencies of Wind Speeds', fontsize = 35)

plt.subplot(424)
plt.bar(np.arange(len(norm_wind_speed_freq_July_diff)), norm_wind_speed_freq_July_diff)
plt.xlim(xmax = max_wind_speed + 1)
if region == 'AL':
    plt.ylim(ymax = 2.5, ymin = -1.7)
plt.ylabel('Frequency Difference (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('c.) July')

plt.subplot(425)
plt.bar(np.arange(len(norm_wind_speed_freq_Aug_diff)), norm_wind_speed_freq_Aug_diff)
plt.xlim(xmax = max_wind_speed + 1)
if region == 'AL':
    plt.ylim(ymax = 2.5, ymin = -1.7)
plt.ylabel('Frequency Difference (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('d.) August')

plt.subplot(426)
plt.bar(np.arange(len(norm_wind_speed_freq_Sep_diff)), norm_wind_speed_freq_Sep_diff)
plt.xlim(xmax = max_wind_speed + 1)
if region == 'AL':
    plt.ylim(ymax = 2.5, ymin = -1.7)
plt.ylabel('Frequency Difference (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('e.) September')

plt.subplot(427)
plt.bar(np.arange(len(norm_wind_speed_freq_Oct_diff)), norm_wind_speed_freq_Oct_diff)
plt.xlim(xmax = max_wind_speed + 1)
if region == 'AL':
    plt.ylim(ymax = 2.5, ymin = -1.7)
plt.ylabel('Frequency Difference (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('f.) October')

plt.subplot(428)
plt.bar(np.arange(len(norm_wind_speed_freq_Nov_diff)), norm_wind_speed_freq_Nov_diff)
plt.xlim(xmax = max_wind_speed + 1)
if region == 'AL':
    plt.ylim(ymax = 2.5, ymin = -1.7)
plt.ylabel('Frequency Difference (%)')
plt.xlabel('Wind Speed (m/s)')
plt.title('g.) November')

plt.subplots_adjust(left = 0.125, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.25, hspace = 0.3)

plt.savefig('Figures/Filtered_Norm_Wind_Speeds_' + location_names[region][0] + '_79-97_98-16_Diff_3x2_Histogram.png')
plt.show()


######################################################################################
##                                 Print Statistics                                 ##
######################################################################################

# Create a map showing the points that the data was obtained from
m = Basemap(llcrnrlon = -100, llcrnrlat = 23, urcrnrlon = -75, urcrnrlat = 37,
            projection = 'cyl', resolution ='i')#, lon_0 = -80, lat_0 = 35)
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary(fill_color='#208eed')
m.fillcontinents(color = '#2ac745', lake_color='#208eed')
m.drawparallels(np.arange(20,50,10),labels = [1,1,0,0], fontsize = 15)
m.drawmeridians(np.arange(-100,-60,10),labels = [0,0,0,1], fontsize = 15)

i = 0
for loc in locs:
    #if (loc[0] > -84 and loc[0] <= -79.5 and loc[1] <= 30.75):
    #    colorcode = '#006700'
    #elif (loc[0] == -81.75 and loc[1] > 30.75) or (loc[0] == -81 and loc[1] > 30.75) or (loc[0] == -80.25 and loc[1] > 30.75) or \
    #     (loc[0] == -79.5 and loc[1] > 30.75) or (loc[0] == -78.75 and loc[1] <= 33.75) or (loc[0] == -78 and loc[1] == 33):
    #    colorcode = '#0000ff'
    #else:
    #    colorcode = '#ff0000'

    colorcode = '#ff0000'
    m.scatter(loc[0], loc[1], 10, marker = 'o', color = colorcode, latlon = True, zorder = 10)

#plt.title('Locations of Data Points')
plt.savefig('Figures/' + location_names[region][0]  + '/Wind_Speed_Measurement_Locations_' + location_names[region][0] + '.png')
plt.savefig('Figures/Measurement_Locations/Wind_Speed_Measurement_Locations_' + location_names[region][0] + '.png')
plt.show()

# Flatten the wind speed arrays and remove any NaNs
wind_speeds_79_97_flat = [item for sublist in wind_speeds_79_97 for item in sublist]
wind_speeds_79_97_no_nan = [x for x in wind_speeds_79_97_flat if str(x) != 'nan']
wind_speeds_79_97 = np.array(wind_speeds_79_97_no_nan)
wind_speeds_98_16_flat = [item for sublist in wind_speeds_98_16 for item in sublist]
wind_speeds_98_16_no_nan = [x for x in wind_speeds_98_16_flat if str(x) != 'nan']
wind_speeds_98_16 = np.array(wind_speeds_98_16_no_nan)

# Bootstrap the data to get 95% confidence intervals using the median
print(len(wind_speeds_79_97))
sample_size = len(wind_speeds_79_97)/5
n_79_97 = wind_speeds_79_97[np.random.randint(0, len(wind_speeds_79_97), sample_size)]
n_98_16 = wind_speeds_98_16[np.random.randint(0, len(wind_speeds_98_16), sample_size)]
n_boots_79_97 = len(wind_speeds_79_97)
n_boots_98_16 = len(wind_speeds_98_16)
me_79_97 = np.zeros(n_boots_79_97)
#mn_79_97 = np.zeros(n_boots_79_97)
me_98_16 = np.zeros(n_boots_98_16)
#mn_98_16 = np.zeros(n_boots_98_16)
for i in xrange(n_boots_79_97):
    sample_79_97 = n_79_97[np.random.randint(0, sample_size, sample_size)]
    me_79_97[i] = np.median(sample_79_97)
    #mn_79_97[i] = np.nanmean(sample_79_97)
print('Done with 79-97')
for i in xrange(n_boots_98_16):
    sample_98_16 = n_98_16[np.random.randint(0, sample_size, sample_size)]
    me_98_16[i] = np.median(sample_98_16)
    #mn_98_16[i] = np.nanmean(sample_98_16)

# Compute medians and their confidence intervals
med_79_97 = np.median(me_79_97)
ci95med_79_97 = np.percentile(me_79_97, [2.5, 97.5])
#mea_79_97 = np.mean(mn_79_97)
#ci95mean_79_97 = np.percentile(mn_79_97, [2.5, 97.5])
med_98_16 = np.median(me_98_16)
ci95med_98_16 = np.percentile(me_98_16, [2.5, 97.5])
#mea_98_16 = np.mean(mn_98_16)
#ci95mean_98_16 = np.percentile(mn_98_16, [2.5, 97.5])

# Print some notable statistics about the data
print('\nTotal Number of Stagnant Flow Measurements Observed at ' + location_names[region][2] + ' from 1979 ' +
      'up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed at ' + location_names[region][2] + ' from 1979 ' +
      'up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed at ' + location_names[region][2] + ' from 1998 ' +
      'up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16, stagnant_flow)))
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' from 1979 up to 2017: ' + 
      str(np.nanmean(wind_speeds_79_16)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' from 1979 up to 1998: ' +
      str(np.nanmean(wind_speeds_79_97)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' from 1998 up to 2017: ' +
      str(np.nanmean(wind_speeds_98_16)) + ' m/s')
print('Difference: ' + str(np.nanmean(wind_speeds_98_16) - np.nanmean(wind_speeds_79_97)))
x = np.array(wind_speeds_79_16)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' from 1979 up to 2017: ' +
      str(np.median(x[~np.isnan(x)])) + ' m/s')
y = np.array(wind_speeds_79_97)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' from 1979 up to 1998: ' +
      str(np.median(y[~np.isnan(y)])) + ' m/s')
z = np.array(wind_speeds_98_16)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' from 1998 up to 2017: ' +
      str(np.median(z[~np.isnan(z)])) + ' m/s')
print('Difference: ' + str(np.median(z[~np.isnan(z)]) - np.median(y[~np.isnan(y)])))
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' from 1979 up to ' +
      '2017: ' + str(np.nanstd(wind_speeds_79_16)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' from 1979 up to ' +
      '1998: ' + str(np.nanstd(wind_speeds_79_97)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' from 1998 up to ' +
      '2017: ' + str(np.nanstd(wind_speeds_98_16)) + ' m/s')
print('Difference: ' + str(np.nanstd(wind_speeds_98_16) - np.nanstd(wind_speeds_79_97)) + '\n')
print('Confidence Intervals and Medians:')
print('79-97 CI and Median')
print(ci95med_79_97)
print(med_79_97)
print('98-16 CI and Median')
print(ci95med_98_16)
print(med_98_16)
print('CI and Median Diff')
print([ci95med_98_16[1] - ci95med_79_97[0], ci95med_98_16[0] - ci95med_79_97[1]])
print(str(med_98_16 - med_79_97) + '\n')

# Flatten the wind speed arrays and remove any NaNs
wind_speeds_79_97_flat = [item for sublist in wind_speeds_79_97_EHS for item in sublist]
wind_speeds_79_97_no_nan = [x for x in wind_speeds_79_97_flat if str(x) != 'nan']
wind_speeds_79_97_EHS = np.array(wind_speeds_79_97_no_nan)
wind_speeds_98_16_flat = [item for sublist in wind_speeds_98_16_EHS for item in sublist]
wind_speeds_98_16_no_nan = [x for x in wind_speeds_98_16_flat if str(x) != 'nan']
wind_speeds_98_16_EHS = np.array(wind_speeds_98_16_no_nan)

# Bootstrap the data to get 95% confidence intervals using the median
sample_size = len(wind_speeds_79_97_EHS)/5
n_79_97 = wind_speeds_79_97_EHS[np.random.randint(0, len(wind_speeds_79_97_EHS), sample_size)]
n_98_16 = wind_speeds_98_16_EHS[np.random.randint(0, len(wind_speeds_98_16_EHS), sample_size)]
n_boots_79_97 = len(wind_speeds_79_97_EHS)
n_boots_98_16 = len(wind_speeds_98_16_EHS)
me_79_97 = np.zeros(n_boots_79_97)
me_98_16 = np.zeros(n_boots_98_16)
for i in xrange(n_boots_79_97):
    sample_79_97 = n_79_97[np.random.randint(0, sample_size, sample_size)]
    me_79_97[i] = np.median(sample_79_97)
print('Done with 79-97')
for i in xrange(n_boots_98_16):
    sample_98_16 = n_98_16[np.random.randint(0, sample_size, sample_size)]
    me_98_16[i] = np.median(sample_98_16)

# Compute medians and their confidence intervals
med_79_97 = np.median(me_79_97)
ci95med_79_97 = np.percentile(me_79_97, [2.5, 97.5])
med_98_16 = np.median(me_98_16)
ci95med_98_16 = np.percentile(me_98_16, [2.5, 97.5])

print('Total Number of Stagnant Flow Measurements Observed in Late May, June, and July ' +
      'at ' + location_names[region][2] + ' from 1979 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all_EHS, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in Late May, June, and July ' +
      'at ' + location_names[region][2] + ' from 1979 up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97_EHS, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in Late May, June, and July ' +
      'at ' + location_names[region][2] + ' from 1998 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16_EHS, stagnant_flow)))
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in Late May, June, and July from 1979 up to 2017: ' +
      str(np.nanmean(wind_speeds_79_16_EHS)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in Late May, June, and July from 1979 up to 1998: ' +
      str(np.nanmean(wind_speeds_79_97_EHS)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in Late May, June, and July from 1998 up to 2017: ' +
      str(np.nanmean(wind_speeds_98_16_EHS)) + ' m/s')
print('Difference: ' + str(np.nanmean(wind_speeds_98_16_EHS) - np.nanmean(wind_speeds_79_97_EHS)))
x = np.array(wind_speeds_79_16_EHS)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in Late May, June, and July from 1979 up to 2017: ' +
      str(np.median(x[~np.isnan(x)])) + ' m/s')
y = np.array(wind_speeds_79_97_EHS)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in Late May, June, and July from 1979 up to 1998: ' +
      str(np.median(y[~np.isnan(y)])) + ' m/s')
z = np.array(wind_speeds_98_16_EHS)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in Late May, June, and July from 1998 up to 2017: ' +
      str(np.median(z[~np.isnan(z)])) + ' m/s')
print('Difference: ' + str(np.median(z[~np.isnan(z)]) - np.median(y[~np.isnan(y)])))
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in Late May, June, and July ' +
      'from 1979 up to 2017: ' + str(np.nanstd(wind_speeds_79_16_EHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in Late May, June, and July ' +
      'from 1979 up to 1998: ' + str(np.nanstd(wind_speeds_79_97_EHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in Late May, June, and July ' +
      'from 1998 up to 2017: ' + str(np.nanstd(wind_speeds_98_16_EHS)) + ' m/s')
print('Difference: ' + str(np.nanstd(wind_speeds_98_16_EHS) - np.nanstd(wind_speeds_79_97_EHS)) + '\n')
print('Early Season Confidence Intervals and Medians:')
print('79-97 CI and Median')
print(ci95med_79_97)
print(med_79_97)
print('98-16 CI and Median')
print(ci95med_98_16)
print(med_98_16)
print('CI and Median Diff')
print([ci95med_98_16[1] - ci95med_79_97[0], ci95med_98_16[0] - ci95med_79_97[1]])
print(str(med_98_16 - med_79_97) + '\n')

# Flatten the wind speed arrays and remove any NaNs
wind_speeds_79_97_flat = [item for sublist in wind_speeds_79_97_MHS for item in sublist]
wind_speeds_79_97_no_nan = [x for x in wind_speeds_79_97_flat if str(x) != 'nan']
wind_speeds_79_97_MHS = np.array(wind_speeds_79_97_no_nan)
wind_speeds_98_16_flat = [item for sublist in wind_speeds_98_16_MHS for item in sublist]
wind_speeds_98_16_no_nan = [x for x in wind_speeds_98_16_flat if str(x) != 'nan']
wind_speeds_98_16_MHS = np.array(wind_speeds_98_16_no_nan)

# Bootstrap the data to get 95% confidence intervals using the median
sample_size = len(wind_speeds_79_97_MHS)/5
n_79_97 = wind_speeds_79_97_MHS[np.random.randint(0, len(wind_speeds_79_97_MHS), sample_size)]
n_98_16 = wind_speeds_98_16_MHS[np.random.randint(0, len(wind_speeds_98_16_MHS), sample_size)]
n_boots_79_97 = len(wind_speeds_79_97_MHS)
n_boots_98_16 = len(wind_speeds_98_16_MHS)
me_79_97 = np.zeros(n_boots_79_97)
me_98_16 = np.zeros(n_boots_98_16)
for i in xrange(n_boots_79_97):
    sample_79_97 = n_79_97[np.random.randint(0, sample_size, sample_size)]
    me_79_97[i] = np.median(sample_79_97)
print('Done with 79-97')
for i in xrange(n_boots_98_16):
    sample_98_16 = n_98_16[np.random.randint(0, sample_size, sample_size)]
    me_98_16[i] = np.median(sample_98_16)

# Compute medians and their confidence intervals
med_79_97 = np.median(me_79_97)
ci95med_79_97 = np.percentile(me_79_97, [2.5, 97.5])
med_98_16 = np.median(me_98_16)
ci95med_98_16 = np.percentile(me_98_16, [2.5, 97.5])

print('Total Number of Stagnant Flow Measurements Observed in August and September ' +
      'at ' + location_names[region][2] + ' from 1979 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all_MHS, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in August and September ' +
      'at ' + location_names[region][2] + ' from 1979 up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97_MHS, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in August and September ' +
      'at ' + location_names[region][2] + ' from 1998 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16_MHS, stagnant_flow)))
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in August and September from 1979 up to 2017: ' +
      str(np.nanmean(wind_speeds_79_16_MHS)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in August and September from 1979 up to 1998: ' +
      str(np.nanmean(wind_speeds_79_97_MHS)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in August and September from 1998 up to 2017: ' +
      str(np.nanmean(wind_speeds_98_16_MHS)) + ' m/s')
print('Difference: ' + str(np.nanmean(wind_speeds_98_16_MHS) - np.nanmean(wind_speeds_79_97_MHS)))
x = np.array(wind_speeds_79_16_MHS)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in August and September from 1979 up to 2017: ' +
      str(np.median(x[~np.isnan(x)])) + ' m/s')
y = np.array(wind_speeds_79_97_MHS)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in August and September from 1979 up to 1998: ' +
      str(np.median(y[~np.isnan(y)])) + ' m/s')
z = np.array(wind_speeds_98_16_MHS)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in August and September from 1998 up to 2017: ' +
      str(np.median(z[~np.isnan(z)])) + ' m/s')
print('Difference: ' + str(np.median(z[~np.isnan(z)]) - np.median(y[~np.isnan(y)])))
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in August and September ' +
      'from 1979 up to 2017: ' + str(np.nanstd(wind_speeds_79_16_MHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in August and September ' +
      'from 1979 up to 1998: ' + str(np.nanstd(wind_speeds_79_97_MHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in August and September ' +
      'from 1998 up to 2017: ' + str(np.nanstd(wind_speeds_98_16_MHS)) + ' m/s')
print('Difference: ' + str(np.nanstd(wind_speeds_98_16_MHS) - np.nanstd(wind_speeds_79_97_MHS)) + '\n')
print('Mid Season Confidence Intervals and Medians:')
print('79-97 CI and Median')
print(ci95med_79_97)
print(med_79_97)
print('98-16 CI and Median')
print(ci95med_98_16)
print(med_98_16)
print('CI and Median Diff')
print([ci95med_98_16[1] - ci95med_79_97[0], ci95med_98_16[0] - ci95med_79_97[1]])
print(str(med_98_16 - med_79_97) + '\n')

# Flatten the wind speed arrays and remove any NaNs
wind_speeds_79_97_flat = [item for sublist in wind_speeds_79_97_MHS_v2 for item in sublist]
wind_speeds_79_97_no_nan = [x for x in wind_speeds_79_97_flat if str(x) != 'nan']
wind_speeds_79_97_MHS_v2 = np.array(wind_speeds_79_97_no_nan)
wind_speeds_98_16_flat = [item for sublist in wind_speeds_98_16_MHS_v2 for item in sublist]
wind_speeds_98_16_no_nan = [x for x in wind_speeds_98_16_flat if str(x) != 'nan']
wind_speeds_98_16_MHS_v2 = np.array(wind_speeds_98_16_no_nan)

# Bootstrap the data to get 95% confidence intervals using the median
sample_size = len(wind_speeds_79_97_MHS_v2)/5
n_79_97 = wind_speeds_79_97_MHS_v2[np.random.randint(0, len(wind_speeds_79_97_MHS_v2), sample_size)]
n_98_16 = wind_speeds_98_16_MHS_v2[np.random.randint(0, len(wind_speeds_98_16_MHS_v2), sample_size)]
n_boots_79_97 = len(wind_speeds_79_97_MHS_v2)
n_boots_98_16 = len(wind_speeds_98_16_MHS_v2)
me_79_97 = np.zeros(n_boots_79_97)
me_98_16 = np.zeros(n_boots_98_16)
for i in xrange(n_boots_79_97):
    sample_79_97 = n_79_97[np.random.randint(0, sample_size, sample_size)]
    me_79_97[i] = np.median(sample_79_97)
print('Done with 79-97')
for i in xrange(n_boots_98_16):
    sample_98_16 = n_98_16[np.random.randint(0, sample_size, sample_size)]
    me_98_16[i] = np.median(sample_98_16)

# Compute medians and their confidence intervals
med_79_97 = np.median(me_79_97)
ci95med_79_97 = np.percentile(me_79_97, [2.5, 97.5])
med_98_16 = np.median(me_98_16)
ci95med_98_16 = np.percentile(me_98_16, [2.5, 97.5])

print('Total Number of Stagnant Flow Measurements Observed in July and August ' +
      'at ' + location_names[region][2] + ' from 1979 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all_MHS_v2, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in July and August ' +
      'at ' + location_names[region][2] + ' from 1979 up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97_MHS_v2, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in July and August ' +
      'at ' + location_names[region][2] + ' from 1998 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16_MHS_v2, stagnant_flow)))
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in July and August from 1979 up to 2017: ' +
      str(np.nanmean(wind_speeds_79_16_MHS_v2)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in July and August from 1979 up to 1998: ' +
      str(np.nanmean(wind_speeds_79_97_MHS_v2)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in July and August from 1998 up to 2017: ' +
      str(np.nanmean(wind_speeds_98_16_MHS_v2)) + ' m/s')
print('Difference: ' + str(np.nanmean(wind_speeds_98_16_MHS_v2) - np.nanmean(wind_speeds_79_97_MHS_v2)))
x = np.array(wind_speeds_79_16_MHS_v2)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in July and August from 1979 up to 2017: ' +
      str(np.median(x[~np.isnan(x)])) + ' m/s')
y = np.array(wind_speeds_79_97_MHS_v2)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in July and August from 1979 up to 1998: ' +
      str(np.median(y[~np.isnan(y)])) + ' m/s')
z = np.array(wind_speeds_98_16_MHS_v2)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in July and August from 1998 up to 2017: ' +
      str(np.median(z[~np.isnan(z)])) + ' m/s')
print('Difference: ' + str(np.median(z[~np.isnan(z)]) - np.median(y[~np.isnan(y)])))
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in July and August ' +
      'from 1979 up to 2017: ' + str(np.nanstd(wind_speeds_79_16_MHS_v2)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in July and August ' +
      'from 1979 up to 1998: ' + str(np.nanstd(wind_speeds_79_97_MHS_v2)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in July and August ' +
      'from 1998 up to 2017: ' + str(np.nanstd(wind_speeds_98_16_MHS_v2)) + ' m/s')
print('Difference: ' + str(np.nanstd(wind_speeds_98_16_MHS_v2) - np.nanstd(wind_speeds_79_97_MHS_v2)) + '\n')
print('July/August Confidence Intervals and Medians:')
print('79-97 CI and Median')
print(ci95med_79_97)
print(med_79_97)
print('98-16 CI and Median')
print(ci95med_98_16)
print(med_98_16)
print('CI and Median Diff')
print([ci95med_98_16[1] - ci95med_79_97[0], ci95med_98_16[0] - ci95med_79_97[1]])
print(str(med_98_16 - med_79_97) + '\n')

# Flatten the wind speed arrays and remove any NaNs
wind_speeds_79_97_flat = [item for sublist in wind_speeds_79_97_LHS for item in sublist]
wind_speeds_79_97_no_nan = [x for x in wind_speeds_79_97_flat if str(x) != 'nan']
wind_speeds_79_97_LHS = np.array(wind_speeds_79_97_no_nan)
wind_speeds_98_16_flat = [item for sublist in wind_speeds_98_16_LHS for item in sublist]
wind_speeds_98_16_no_nan = [x for x in wind_speeds_98_16_flat if str(x) != 'nan']
wind_speeds_98_16_LHS = np.array(wind_speeds_98_16_no_nan)

# Bootstrap the data to get 95% confidence intervals using the median
sample_size = len(wind_speeds_79_97_LHS)/5
n_79_97 = wind_speeds_79_97_LHS[np.random.randint(0, len(wind_speeds_79_97_LHS), sample_size)]
n_98_16 = wind_speeds_98_16_LHS[np.random.randint(0, len(wind_speeds_98_16_LHS), sample_size)]
n_boots_79_97 = len(wind_speeds_79_97_LHS)
n_boots_98_16 = len(wind_speeds_98_16_LHS)
me_79_97 = np.zeros(n_boots_79_97)
me_98_16 = np.zeros(n_boots_98_16)
for i in xrange(n_boots_79_97):
    sample_79_97 = n_79_97[np.random.randint(0, sample_size, sample_size)]
    me_79_97[i] = np.median(sample_79_97)
print('Done with 79-97')
for i in xrange(n_boots_98_16):
    sample_98_16 = n_98_16[np.random.randint(0, sample_size, sample_size)]
    me_98_16[i] = np.median(sample_98_16)

# Compute medians and their confidence intervals
med_79_97 = np.median(me_79_97)
ci95med_79_97 = np.percentile(me_79_97, [2.5, 97.5])
med_98_16 = np.median(me_98_16)
ci95med_98_16 = np.percentile(me_98_16, [2.5, 97.5])

print('Total Number of Stagnant Flow Measurements Observed in October and November ' +
      'at ' + location_names[region][2] + ' from 1979 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all_LHS, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in October and November ' +
      'at ' + location_names[region][2] + ' from 1979 up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97_LHS, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in October and November ' +
      'at ' + location_names[region][2] + ' from 1998 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16_LHS, stagnant_flow)))
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in October and November from 1979 up to 2017: ' +
      str(np.nanmean(wind_speeds_79_16_LHS)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in October and November from 1979 up to 1998: ' +
      str(np.nanmean(wind_speeds_79_97_LHS)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in October and November from 1998 up to 2017: ' +
      str(np.nanmean(wind_speeds_98_16_LHS)) + ' m/s')
print('Difference: ' + str(np.nanmean(wind_speeds_98_16_LHS) - np.nanmean(wind_speeds_79_97_LHS)))
x = np.array(wind_speeds_79_16_LHS)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in October and November from 1979 up to 2017: ' +
      str(np.median(x[~np.isnan(x)])) + ' m/s')
y = np.array(wind_speeds_79_97_LHS)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in October and November from 1979 up to 1998: ' +
      str(np.median(y[~np.isnan(y)])) + ' m/s')
z = np.array(wind_speeds_98_16_LHS)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in October and November from 1998 up to 2017: ' +
      str(np.median(z[~np.isnan(z)])) + ' m/s')
print('Difference: ' + str(np.median(z[~np.isnan(z)]) - np.median(y[~np.isnan(y)])))
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in October and November ' +
      'from 1979 up to 2017: ' + str(np.nanstd(wind_speeds_79_16_LHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in October and November ' +
      'from 1979 up to 1998: ' + str(np.nanstd(wind_speeds_79_97_LHS)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in October and November ' +
      'from 1998 up to 2017: ' + str(np.nanstd(wind_speeds_98_16_LHS)) + ' m/s')
print('Difference: ' + str(np.nanstd(wind_speeds_98_16_LHS) - np.nanstd(wind_speeds_79_97_LHS)) + '\n')
print('Late Season Confidence Intervals and Medians:')
print('79-97 CI and Median')
print(ci95med_79_97)
print(med_79_97)
print('98-16 CI and Median')
print(ci95med_98_16)
print(med_98_16)
print('CI and Median Diff')
print([ci95med_98_16[1] - ci95med_79_97[0], ci95med_98_16[0] - ci95med_79_97[1]])
print(str(med_98_16 - med_79_97) + '\n')

# Flatten the wind speed arrays and remove any NaNs
wind_speeds_79_97_flat = [item for sublist in wind_speeds_79_97_LHS_v2 for item in sublist]
wind_speeds_79_97_no_nan = [x for x in wind_speeds_79_97_flat if str(x) != 'nan']
wind_speeds_79_97_LHS_v2 = np.array(wind_speeds_79_97_no_nan)
wind_speeds_98_16_flat = [item for sublist in wind_speeds_98_16_LHS_v2 for item in sublist]
wind_speeds_98_16_no_nan = [x for x in wind_speeds_98_16_flat if str(x) != 'nan']
wind_speeds_98_16_LHS_v2 = np.array(wind_speeds_98_16_no_nan)

# Bootstrap the data to get 95% confidence intervals using the median
sample_size = len(wind_speeds_79_97_LHS_v2)/5
n_79_97 = wind_speeds_79_97_LHS_v2[np.random.randint(0, len(wind_speeds_79_97_LHS_v2), sample_size)]
n_98_16 = wind_speeds_98_16_LHS_v2[np.random.randint(0, len(wind_speeds_98_16_LHS_v2), sample_size)]
n_boots_79_97 = len(wind_speeds_79_97_LHS_v2)
n_boots_98_16 = len(wind_speeds_98_16_LHS_v2)
me_79_97 = np.zeros(n_boots_79_97)
me_98_16 = np.zeros(n_boots_98_16)
for i in xrange(n_boots_79_97):
    sample_79_97 = n_79_97[np.random.randint(0, sample_size, sample_size)]
    me_79_97[i] = np.median(sample_79_97)
print('Done with 79-97')
for i in xrange(n_boots_98_16):
    sample_98_16 = n_98_16[np.random.randint(0, sample_size, sample_size)]
    me_98_16[i] = np.median(sample_98_16)

# Compute medians and their confidence intervals
med_79_97 = np.median(me_79_97)
ci95med_79_97 = np.percentile(me_79_97, [2.5, 97.5])
med_98_16 = np.median(me_98_16)
ci95med_98_16 = np.percentile(me_98_16, [2.5, 97.5])

print('Total Number of Stagnant Flow Measurements Observed in September and October ' +
      'at ' + location_names[region][2] + ' from 1979 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all_LHS_v2, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in September and October ' +
      'at ' + location_names[region][2] + ' from 1979 up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97_LHS_v2, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in September and October ' +
      'at ' + location_names[region][2] + ' from 1998 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16_LHS_v2, stagnant_flow)))
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in September and October from 1979 up to 2017: ' +
      str(np.nanmean(wind_speeds_79_16_LHS_v2)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in September and October from 1979 up to 1998: ' +
      str(np.nanmean(wind_speeds_79_97_LHS_v2)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in September and October from 1998 up to 2017: ' +
      str(np.nanmean(wind_speeds_98_16_LHS_v2)) + ' m/s')
print('Difference: ' + str(np.nanmean(wind_speeds_98_16_LHS_v2) - np.nanmean(wind_speeds_79_97_LHS_v2)))
x = np.array(wind_speeds_79_16_LHS_v2)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in September and October from 1979 up to 2017: ' +
      str(np.median(x[~np.isnan(x)])) + ' m/s')
y = np.array(wind_speeds_79_97_LHS_v2)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in September and October from 1979 up to 1998: ' +
      str(np.median(y[~np.isnan(y)])) + ' m/s')
z = np.array(wind_speeds_98_16_LHS_v2)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in September and October from 1998 up to 2017: ' +
      str(np.median(z[~np.isnan(z)])) + ' m/s')
print('Difference: ' + str(np.median(z[~np.isnan(z)]) - np.median(y[~np.isnan(y)])))
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in September and October ' +
      'from 1979 up to 2017: ' + str(np.nanstd(wind_speeds_79_16_LHS_v2)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in September and October ' +
      'from 1979 up to 1998: ' + str(np.nanstd(wind_speeds_79_97_LHS_v2)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in September and October ' +
      'from 1998 up to 2017: ' + str(np.nanstd(wind_speeds_98_16_LHS_v2)) + ' m/s')
print('Difference: ' + str(np.nanstd(wind_speeds_98_16_LHS_v2) - np.nanstd(wind_speeds_79_97_LHS_v2)) + '\n')
print('September/October Confidence Intervals and Medians:')
print('79-97 CI and Median')
print(ci95med_79_97)
print(med_79_97)
print('98-16 CI and Median')
print(ci95med_98_16)
print(med_98_16)
print('CI and Median Diff')
print([ci95med_98_16[1] - ci95med_79_97[0], ci95med_98_16[0] - ci95med_79_97[1]])
print(str(med_98_16 - med_79_97) + '\n')

# Flatten the wind speed arrays and remove any NaNs
wind_speeds_79_97_flat = [item for sublist in wind_speeds_79_97_June for item in sublist]
wind_speeds_79_97_no_nan = [x for x in wind_speeds_79_97_flat if str(x) != 'nan']
wind_speeds_79_97_June = np.array(wind_speeds_79_97_no_nan)
wind_speeds_98_16_flat = [item for sublist in wind_speeds_98_16_June for item in sublist]
wind_speeds_98_16_no_nan = [x for x in wind_speeds_98_16_flat if str(x) != 'nan']
wind_speeds_98_16_June = np.array(wind_speeds_98_16_no_nan)

# Bootstrap the data to get 95% confidence intervals using the median
sample_size = len(wind_speeds_79_97_June)/5
n_79_97 = wind_speeds_79_97_June[np.random.randint(0, len(wind_speeds_79_97_June), sample_size)]
n_98_16 = wind_speeds_98_16_June[np.random.randint(0, len(wind_speeds_98_16_June), sample_size)]
n_boots_79_97 = len(wind_speeds_79_97_June)
n_boots_98_16 = len(wind_speeds_98_16_June)
me_79_97 = np.zeros(n_boots_79_97)
me_98_16 = np.zeros(n_boots_98_16)
for i in xrange(n_boots_79_97):
    sample_79_97 = n_79_97[np.random.randint(0, sample_size, sample_size)]
    me_79_97[i] = np.median(sample_79_97)
print('Done with 79-97')
for i in xrange(n_boots_98_16):
    sample_98_16 = n_98_16[np.random.randint(0, sample_size, sample_size)]
    me_98_16[i] = np.median(sample_98_16)

# Compute medians and their confidence intervals
med_79_97 = np.median(me_79_97)
ci95med_79_97 = np.percentile(me_79_97, [2.5, 97.5])
med_98_16 = np.median(me_98_16)
ci95med_98_16 = np.percentile(me_98_16, [2.5, 97.5])

print('Total Number of Stagnant Flow Measurements Observed in June ' +
      'at ' + location_names[region][2] + ' from 1979 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all_June, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in June ' +
      'at ' + location_names[region][2] + ' from 1979 up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97_June, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in June ' +
      'at ' + location_names[region][2] + ' from 1998 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16_June, stagnant_flow)))
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in June from 1979 up to 2017: ' +
      str(np.nanmean(wind_speeds_79_16_June)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in June from 1979 up to 1998: ' +
      str(np.nanmean(wind_speeds_79_97_June)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in June from 1998 up to 2017: ' +
      str(np.nanmean(wind_speeds_98_16_June)) + ' m/s')
print('Difference: ' + str(np.nanmean(wind_speeds_98_16_June) - np.nanmean(wind_speeds_79_97_June)))
x = np.array(wind_speeds_79_16_June)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in June from 1979 up to 2017: ' +
      str(np.median(x[~np.isnan(x)])) + ' m/s')
y = np.array(wind_speeds_79_97_June)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in June from 1979 up to 1998: ' +
      str(np.median(y[~np.isnan(y)])) + ' m/s')
z = np.array(wind_speeds_98_16_June)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in June from 1998 up to 2017: ' +
      str(np.median(z[~np.isnan(z)])) + ' m/s')
print('Difference: ' + str(np.median(z[~np.isnan(z)]) - np.median(y[~np.isnan(y)])))
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in June ' +
      'from 1979 up to 2017: ' + str(np.nanstd(wind_speeds_79_16_June)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in June ' +
      'from 1979 up to 1998: ' + str(np.nanstd(wind_speeds_79_97_June)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in June ' +
      'from 1998 up to 2017: ' + str(np.nanstd(wind_speeds_98_16_June)) + ' m/s')
print('Difference: ' + str(np.nanstd(wind_speeds_98_16_June) - np.nanstd(wind_speeds_79_97_June)) + '\n')
print('June Confidence Intervals and Medians:')
print('79-97 CI and Median')
print(ci95med_79_97)
print(med_79_97)
print('98-16 CI and Median')
print(ci95med_98_16)
print(med_98_16)
print('CI and Median Diff')
print([ci95med_98_16[1] - ci95med_79_97[0], ci95med_98_16[0] - ci95med_79_97[1]])
print(str(med_98_16 - med_79_97) + '\n')

# Flatten the wind speed arrays and remove any NaNs
wind_speeds_79_97_flat = [item for sublist in wind_speeds_79_97_July for item in sublist]
wind_speeds_79_97_no_nan = [x for x in wind_speeds_79_97_flat if str(x) != 'nan']
wind_speeds_79_97_July = np.array(wind_speeds_79_97_no_nan)
wind_speeds_98_16_flat = [item for sublist in wind_speeds_98_16_July for item in sublist]
wind_speeds_98_16_no_nan = [x for x in wind_speeds_98_16_flat if str(x) != 'nan']
wind_speeds_98_16_July = np.array(wind_speeds_98_16_no_nan)

# Bootstrap the data to get 95% confidence intervals using the median
sample_size = len(wind_speeds_79_97_July)/5
n_79_97 = wind_speeds_79_97_July[np.random.randint(0, len(wind_speeds_79_97_July), sample_size)]
n_98_16 = wind_speeds_98_16_July[np.random.randint(0, len(wind_speeds_98_16_July), sample_size)]
n_boots_79_97 = len(wind_speeds_79_97_July)
n_boots_98_16 = len(wind_speeds_98_16_July)
me_79_97 = np.zeros(n_boots_79_97)
me_98_16 = np.zeros(n_boots_98_16)
for i in xrange(n_boots_79_97):
    sample_79_97 = n_79_97[np.random.randint(0, sample_size, sample_size)]
    me_79_97[i] = np.median(sample_79_97)
print('Done with 79-97')
for i in xrange(n_boots_98_16):
    sample_98_16 = n_98_16[np.random.randint(0, sample_size, sample_size)]
    me_98_16[i] = np.median(sample_98_16)

# Compute medians and their confidence intervals
med_79_97 = np.median(me_79_97)
ci95med_79_97 = np.percentile(me_79_97, [2.5, 97.5])
med_98_16 = np.median(me_98_16)
ci95med_98_16 = np.percentile(me_98_16, [2.5, 97.5])

print('Total Number of Stagnant Flow Measurements Observed in July ' +
      'at ' + location_names[region][2] + ' from 1979 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all_July, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in July ' +
      'at ' + location_names[region][2] + ' from 1979 up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97_July, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in July ' +
      'at ' + location_names[region][2] + ' from 1998 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16_July, stagnant_flow)))
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in July from 1979 up to 2017: ' +
      str(np.nanmean(wind_speeds_79_16_July)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in July from 1979 up to 1998: ' +
      str(np.nanmean(wind_speeds_79_97_July)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in July from 1998 up to 2017: ' +
      str(np.nanmean(wind_speeds_98_16_July)) + ' m/s')
print('Difference: ' + str(np.nanmean(wind_speeds_98_16_July) - np.nanmean(wind_speeds_79_97_July)))
x = np.array(wind_speeds_79_16_July)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in July from 1979 up to 2017: ' +
      str(np.median(x[~np.isnan(x)])) + ' m/s')
y = np.array(wind_speeds_79_97_July)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in July from 1979 up to 1998: ' +
      str(np.median(y[~np.isnan(y)])) + ' m/s')
z = np.array(wind_speeds_98_16_July)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in July from 1998 up to 2017: ' +
      str(np.median(z[~np.isnan(z)])) + ' m/s')
print('Difference: ' + str(np.median(z[~np.isnan(z)]) - np.median(y[~np.isnan(y)])))
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in July ' +
      'from 1979 up to 2017: ' + str(np.nanstd(wind_speeds_79_16_July)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in July ' +
      'from 1979 up to 1998: ' + str(np.nanstd(wind_speeds_79_97_July)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in July ' +
      'from 1998 up to 2017: ' + str(np.nanstd(wind_speeds_98_16_July)) + ' m/s')
print('Difference: ' + str(np.nanstd(wind_speeds_98_16_July) - np.nanstd(wind_speeds_79_97_July)) + '\n')
print('July Confidence Intervals and Medians:')
print('79-97 CI and Median')
print(ci95med_79_97)
print(med_79_97)
print('98-16 CI and Median')
print(ci95med_98_16)
print(med_98_16)
print('CI and Median Diff')
print([ci95med_98_16[1] - ci95med_79_97[0], ci95med_98_16[0] - ci95med_79_97[1]])
print(str(med_98_16 - med_79_97) + '\n')

# Flatten the wind speed arrays and remove any NaNs
wind_speeds_79_97_flat = [item for sublist in wind_speeds_79_97_Aug for item in sublist]
wind_speeds_79_97_no_nan = [x for x in wind_speeds_79_97_flat if str(x) != 'nan']
wind_speeds_79_97_Aug = np.array(wind_speeds_79_97_no_nan)
wind_speeds_98_16_flat = [item for sublist in wind_speeds_98_16_Aug for item in sublist]
wind_speeds_98_16_no_nan = [x for x in wind_speeds_98_16_flat if str(x) != 'nan']
wind_speeds_98_16_Aug = np.array(wind_speeds_98_16_no_nan)

# Bootstrap the data to get 95% confidence intervals using the median
sample_size = len(wind_speeds_79_97_Aug)/5
n_79_97 = wind_speeds_79_97_Aug[np.random.randint(0, len(wind_speeds_79_97_Aug), sample_size)]
n_98_16 = wind_speeds_98_16_Aug[np.random.randint(0, len(wind_speeds_98_16_Aug), sample_size)]
n_boots_79_97 = len(wind_speeds_79_97_Aug)
n_boots_98_16 = len(wind_speeds_98_16_Aug)
me_79_97 = np.zeros(n_boots_79_97)
me_98_16 = np.zeros(n_boots_98_16)
for i in xrange(n_boots_79_97):
    sample_79_97 = n_79_97[np.random.randint(0, sample_size, sample_size)]
    me_79_97[i] = np.median(sample_79_97)
print('Done with 79-97')
for i in xrange(n_boots_98_16):
    sample_98_16 = n_98_16[np.random.randint(0, sample_size, sample_size)]
    me_98_16[i] = np.median(sample_98_16)

# Compute medians and their confidence intervals
med_79_97 = np.median(me_79_97)
ci95med_79_97 = np.percentile(me_79_97, [2.5, 97.5])
med_98_16 = np.median(me_98_16)
ci95med_98_16 = np.percentile(me_98_16, [2.5, 97.5])

print('Total Number of Stagnant Flow Measurements Observed in August ' +
      'at ' + location_names[region][2] + ' from 1979 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all_Aug, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in August ' +
      'at ' + location_names[region][2] + ' from 1979 up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97_Aug, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in August ' +
      'at ' + location_names[region][2] + ' from 1998 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16_Aug, stagnant_flow)))
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in August from 1979 up to 2017: ' +
      str(np.nanmean(wind_speeds_79_16_Aug)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in August from 1979 up to 1998: ' +
      str(np.nanmean(wind_speeds_79_97_Aug)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in August from 1998 up to 2017: ' +
      str(np.nanmean(wind_speeds_98_16_Aug)) + ' m/s')
print('Difference: ' + str(np.nanmean(wind_speeds_98_16_Aug) - np.nanmean(wind_speeds_79_97_Aug)))
x = np.array(wind_speeds_79_16_Aug)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in August from 1979 up to 2017: ' +
      str(np.median(x[~np.isnan(x)])) + ' m/s')
y = np.array(wind_speeds_79_97_Aug)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in August from 1979 up to 1998: ' +
      str(np.median(y[~np.isnan(y)])) + ' m/s')
z = np.array(wind_speeds_98_16_Aug)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in August from 1998 up to 2017: ' +
      str(np.median(z[~np.isnan(z)])) + ' m/s')
print('Difference: ' + str(np.median(z[~np.isnan(z)]) - np.median(y[~np.isnan(y)])))
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in August ' +
      'from 1979 up to 2017: ' + str(np.nanstd(wind_speeds_79_16_Aug)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in August ' +
      'from 1979 up to 1998: ' + str(np.nanstd(wind_speeds_79_97_Aug)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in August ' +
      'from 1998 up to 2017: ' + str(np.nanstd(wind_speeds_98_16_Aug)) + ' m/s')
print('Difference: ' + str(np.nanstd(wind_speeds_98_16_Aug) - np.nanstd(wind_speeds_79_97_Aug)) + '\n')
print('August Confidence Intervals and Medians:')
print('79-97 CI and Median')
print(ci95med_79_97)
print(med_79_97)
print('98-16 CI and Median')
print(ci95med_98_16)
print(med_98_16)
print('CI and Median Diff')
print([ci95med_98_16[1] - ci95med_79_97[0], ci95med_98_16[0] - ci95med_79_97[1]])
print(str(med_98_16 - med_79_97) + '\n')

# Flatten the wind speed arrays and remove any NaNs
wind_speeds_79_97_flat = [item for sublist in wind_speeds_79_97_Sep for item in sublist]
wind_speeds_79_97_no_nan = [x for x in wind_speeds_79_97_flat if str(x) != 'nan']
wind_speeds_79_97_Sep = np.array(wind_speeds_79_97_no_nan)
wind_speeds_98_16_flat = [item for sublist in wind_speeds_98_16_Sep for item in sublist]
wind_speeds_98_16_no_nan = [x for x in wind_speeds_98_16_flat if str(x) != 'nan']
wind_speeds_98_16_Sep = np.array(wind_speeds_98_16_no_nan)

# Bootstrap the data to get 95% confidence intervals using the median
sample_size = len(wind_speeds_79_97_Sep)/5
n_79_97 = wind_speeds_79_97_Sep[np.random.randint(0, len(wind_speeds_79_97_Sep), sample_size)]
n_98_16 = wind_speeds_98_16_Sep[np.random.randint(0, len(wind_speeds_98_16_Sep), sample_size)]
n_boots_79_97 = len(wind_speeds_79_97_Sep)
n_boots_98_16 = len(wind_speeds_98_16_Sep)
me_79_97 = np.zeros(n_boots_79_97)
me_98_16 = np.zeros(n_boots_98_16)
for i in xrange(n_boots_79_97):
    sample_79_97 = n_79_97[np.random.randint(0, sample_size, sample_size)]
    me_79_97[i] = np.median(sample_79_97)
print('Done with 79-97')
for i in xrange(n_boots_98_16):
    sample_98_16 = n_98_16[np.random.randint(0, sample_size, sample_size)]
    me_98_16[i] = np.median(sample_98_16)

# Compute medians and their confidence intervals
med_79_97 = np.median(me_79_97)
ci95med_79_97 = np.percentile(me_79_97, [2.5, 97.5])
med_98_16 = np.median(me_98_16)
ci95med_98_16 = np.percentile(me_98_16, [2.5, 97.5])

print('Total Number of Stagnant Flow Measurements Observed in September ' +
      'at ' + location_names[region][2] + ' from 1979 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all_Sep, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in September ' +
      'at ' + location_names[region][2] + ' from 1979 up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97_Sep, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in September ' +
      'at ' + location_names[region][2] + ' from 1998 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16_Sep, stagnant_flow)))
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in September from 1979 up to 2017: ' +
      str(np.nanmean(wind_speeds_79_16_Sep)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in September from 1979 up to 1998: ' +
      str(np.nanmean(wind_speeds_79_97_Sep)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in September from 1998 up to 2017: ' +
      str(np.nanmean(wind_speeds_98_16_Sep)) + ' m/s')
print('Difference: ' + str(np.nanmean(wind_speeds_98_16_Sep) - np.nanmean(wind_speeds_79_97_Sep)))
x = np.array(wind_speeds_79_16_Sep)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in September from 1979 up to 2017: ' +
      str(np.median(x[~np.isnan(x)])) + ' m/s')
y = np.array(wind_speeds_79_97_Sep)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in September from 1979 up to 1998: ' +
      str(np.median(y[~np.isnan(y)])) + ' m/s')
z = np.array(wind_speeds_98_16_Sep)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in September from 1998 up to 2017: ' +
      str(np.median(z[~np.isnan(z)])) + ' m/s')
print('Difference: ' + str(np.median(z[~np.isnan(z)]) - np.median(y[~np.isnan(y)])))
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in September ' +
      'from 1979 up to 2017: ' + str(np.nanstd(wind_speeds_79_16_Sep)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in September ' +
      'from 1979 up to 1998: ' + str(np.nanstd(wind_speeds_79_97_Sep)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in September ' +
      'from 1998 up to 2017: ' + str(np.nanstd(wind_speeds_98_16_Sep)) + ' m/s')
print('Difference: ' + str(np.nanstd(wind_speeds_98_16_Sep) - np.nanstd(wind_speeds_79_97_Sep)) + '\n')
print('September Confidence Intervals and Medians:')
print('79-97 CI and Median')
print(ci95med_79_97)
print(med_79_97)
print('98-16 CI and Median')
print(ci95med_98_16)
print(med_98_16)
print('CI and Median Diff')
print([ci95med_98_16[1] - ci95med_79_97[0], ci95med_98_16[0] - ci95med_79_97[1]])
print(str(med_98_16 - med_79_97) + '\n')

# Flatten the wind speed arrays and remove any NaNs
wind_speeds_79_97_flat = [item for sublist in wind_speeds_79_97_Oct for item in sublist]
wind_speeds_79_97_no_nan = [x for x in wind_speeds_79_97_flat if str(x) != 'nan']
wind_speeds_79_97_Oct = np.array(wind_speeds_79_97_no_nan)
wind_speeds_98_16_flat = [item for sublist in wind_speeds_98_16_Oct for item in sublist]
wind_speeds_98_16_no_nan = [x for x in wind_speeds_98_16_flat if str(x) != 'nan']
wind_speeds_98_16_Oct = np.array(wind_speeds_98_16_no_nan)

# Bootstrap the data to get 95% confidence intervals using the median
sample_size = len(wind_speeds_79_97_Oct)/5
n_79_97 = wind_speeds_79_97_Oct[np.random.randint(0, len(wind_speeds_79_97_Oct), sample_size)]
n_98_16 = wind_speeds_98_16_Oct[np.random.randint(0, len(wind_speeds_98_16_Oct), sample_size)]
n_boots_79_97 = len(wind_speeds_79_97_Oct)
n_boots_98_16 = len(wind_speeds_98_16_Oct)
me_79_97 = np.zeros(n_boots_79_97)
me_98_16 = np.zeros(n_boots_98_16)
for i in xrange(n_boots_79_97):
    sample_79_97 = n_79_97[np.random.randint(0, sample_size, sample_size)]
    me_79_97[i] = np.median(sample_79_97)
print('Done with 79-97')
for i in xrange(n_boots_98_16):
    sample_98_16 = n_98_16[np.random.randint(0, sample_size, sample_size)]
    me_98_16[i] = np.median(sample_98_16)

# Compute medians and their confidence intervals
med_79_97 = np.median(me_79_97)
ci95med_79_97 = np.percentile(me_79_97, [2.5, 97.5])
med_98_16 = np.median(me_98_16)
ci95med_98_16 = np.percentile(me_98_16, [2.5, 97.5])

print('Total Number of Stagnant Flow Measurements Observed in October ' +
      'at ' + location_names[region][2] + ' from 1979 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all_Oct, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in October ' +
      'at ' + location_names[region][2] + ' from 1979 up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97_Oct, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in October ' +
      'at ' + location_names[region][2] + ' from 1998 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16_Oct, stagnant_flow)))
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in October from 1979 up to 2017: ' +
      str(np.nanmean(wind_speeds_79_16_Oct)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in October from 1979 up to 1998: ' +
      str(np.nanmean(wind_speeds_79_97_Oct)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in October from 1998 up to 2017: ' +
      str(np.nanmean(wind_speeds_98_16_Oct)) + ' m/s')
print('Difference: ' + str(np.nanmean(wind_speeds_98_16_Oct) - np.nanmean(wind_speeds_79_97_Oct)))
x = np.array(wind_speeds_79_16_Oct)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in October from 1979 up to 2017: ' +
      str(np.median(x[~np.isnan(x)])) + ' m/s')
y = np.array(wind_speeds_79_97_Oct)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in October from 1979 up to 1998: ' +
      str(np.median(y[~np.isnan(y)])) + ' m/s')
z = np.array(wind_speeds_98_16_Oct)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in October from 1998 up to 2017: ' +
      str(np.median(z[~np.isnan(z)])) + ' m/s')
print('Difference: ' + str(np.median(z[~np.isnan(z)]) - np.median(y[~np.isnan(y)])))
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in October ' +
      'from 1979 up to 2017: ' + str(np.nanstd(wind_speeds_79_16_Oct)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in October ' +
      'from 1979 up to 1998: ' + str(np.nanstd(wind_speeds_79_97_Oct)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in October ' +
      'from 1998 up to 2017: ' + str(np.nanstd(wind_speeds_98_16_Oct)) + ' m/s')
print('Difference: ' + str(np.nanstd(wind_speeds_98_16_Oct) - np.nanstd(wind_speeds_79_97_Oct)) + '\n')
print('October Confidence Intervals and Medians:')
print('79-97 CI and Median')
print(ci95med_79_97)
print(med_79_97)
print('98-16 CI and Median')
print(ci95med_98_16)
print(med_98_16)
print('CI and Median Diff')
print([ci95med_98_16[1] - ci95med_79_97[0], ci95med_98_16[0] - ci95med_79_97[1]])
print(str(med_98_16 - med_79_97) + '\n')

# Flatten the wind speed arrays and remove any NaNs
wind_speeds_79_97_flat = [item for sublist in wind_speeds_79_97_Nov for item in sublist]
wind_speeds_79_97_no_nan = [x for x in wind_speeds_79_97_flat if str(x) != 'nan']
wind_speeds_79_97_Nov = np.array(wind_speeds_79_97_no_nan)
wind_speeds_98_16_flat = [item for sublist in wind_speeds_98_16_Nov for item in sublist]
wind_speeds_98_16_no_nan = [x for x in wind_speeds_98_16_flat if str(x) != 'nan']
wind_speeds_98_16_Nov = np.array(wind_speeds_98_16_no_nan)

# Bootstrap the data to get 95% confidence intervals using the median
sample_size = len(wind_speeds_79_97_Nov)/5
n_79_97 = wind_speeds_79_97_Nov[np.random.randint(0, len(wind_speeds_79_97_Nov), sample_size)]
n_98_16 = wind_speeds_98_16_Nov[np.random.randint(0, len(wind_speeds_98_16_Nov), sample_size)]
n_boots_79_97 = len(wind_speeds_79_97_Nov)
n_boots_98_16 = len(wind_speeds_98_16_Nov)
me_79_97 = np.zeros(n_boots_79_97)
me_98_16 = np.zeros(n_boots_98_16)
for i in xrange(n_boots_79_97):
    sample_79_97 = n_79_97[np.random.randint(0, sample_size, sample_size)]
    me_79_97[i] = np.median(sample_79_97)
print('Done with 79-97')
for i in xrange(n_boots_98_16):
    sample_98_16 = n_98_16[np.random.randint(0, sample_size, sample_size)]
    me_98_16[i] = np.median(sample_98_16)

# Compute medians and their confidence intervals
med_79_97 = np.median(me_79_97)
ci95med_79_97 = np.percentile(me_79_97, [2.5, 97.5])
med_98_16 = np.median(me_98_16)
ci95med_98_16 = np.percentile(me_98_16, [2.5, 97.5])

print('Total Number of Stagnant Flow Measurements Observed in November ' +
      'at ' + location_names[region][2] + ' from 1979 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_all_Nov, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in November ' +
      'at ' + location_names[region][2] + ' from 1979 up to 1998: ' + str(calcNumStagFlow(wind_speed_freq_79_97_Nov, stagnant_flow)))
print('Total Number of Stagnant Flow Measurements Observed in November ' +
      'at ' + location_names[region][2] + ' from 1998 up to 2017: ' + str(calcNumStagFlow(wind_speed_freq_98_16_Nov, stagnant_flow)))
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in November from 1979 up to 2017: ' +
      str(np.nanmean(wind_speeds_79_16_Nov)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in November from 1979 up to 1998: ' +
      str(np.nanmean(wind_speeds_79_97_Nov)) + ' m/s')
print('Mean Observed Wind Speed for ' + location_names[region][2] + ' in November from 1998 up to 2017: ' +
      str(np.nanmean(wind_speeds_98_16_Nov)) + ' m/s')
print('Difference: ' + str(np.nanmean(wind_speeds_98_16_Nov) - np.nanmean(wind_speeds_79_97_Nov)))
x = np.array(wind_speeds_79_16_Nov)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in November from 1979 up to 2017: ' +
      str(np.median(x[~np.isnan(x)])) + ' m/s')
y = np.array(wind_speeds_79_97_Nov)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in November from 1979 up to 1998: ' +
      str(np.median(y[~np.isnan(y)])) + ' m/s')
z = np.array(wind_speeds_98_16_Nov)
print('Median Observed Wind Speed for ' + location_names[region][2] + ' in November from 1998 up to 2017: ' +
      str(np.median(z[~np.isnan(z)])) + ' m/s')
print('Difference: ' + str(np.median(z[~np.isnan(z)]) - np.median(y[~np.isnan(y)])))
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in November ' +
      'from 1979 up to 2017: ' + str(np.nanstd(wind_speeds_79_16_Nov)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in November ' +
      'from 1979 up to 1998: ' + str(np.nanstd(wind_speeds_79_97_Nov)) + ' m/s')
print('Standard Deviation of Observed Wind Speed for ' + location_names[region][2] + ' in November ' +
      'from 1998 up to 2017: ' + str(np.nanstd(wind_speeds_98_16_Nov)) + ' m/s')
print('Difference: ' + str(np.nanstd(wind_speeds_98_16_Nov) - np.nanstd(wind_speeds_79_97_Nov)) + '\n')
print('November Confidence Intervals and Medians:')
print('79-97 CI and Median')
print(ci95med_79_97)
print(med_79_97)
print('98-16 CI and Median')
print(ci95med_98_16)
print(med_98_16)
print('CI and Median Diff')
print([ci95med_98_16[1] - ci95med_79_97[0], ci95med_98_16[0] - ci95med_79_97[1]])
print(str(med_98_16 - med_79_97) + '\n')
