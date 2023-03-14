import time
from math import ceil

import tzlocal
from datetime import datetime


def time_cost_calc(map_timestamp, periodicities, current_time=time.time()):
    """
    Calculates the cost based on the timestamp (unix time) for a map.
    Args:
        periodicities: the significant frequencies present in the maps for an environment
        map_timestamp: the timestamp of the map
        current_time (optional): Current time,to be provided in case the cost is to be calculated w.r.t to some
        other time instead of current time.
    Returns:
        time_cost
    """
    periodicities.sort()
    N = len(periodicities)
    time_difference = current_time - map_timestamp

    similarity_score = 0
    for periodicity in periodicities:
        # TODO: weights to be decided for each periodicity
        similarity_score += ceil(time_difference / periodicity) - time_difference / periodicity

    similarity_score = similarity_score/N

    return similarity_score


def suitable_timestamps(frequencies):
    # TODO: PROBABLY USELESS
    """
    Calculates the times of the maps that should be considered based on the system's current time
    Args:
        frequencies: The significant periodicities in the data obtained by fourier transform (in seconds)
    Returns:
        A list of tuples (lower_limit, upper_limit, time_cost) containing the suitable times to consider to select maps.
        In the tuple, lower_limit and upper_limit is the range of time which should be considered,
        time_cost is the cost calculation based on time difference.
    """
    current_time = time.time()
    frequencies_in_hours = [frequency // 3600 for frequency in frequencies]
    periodicities_count = len(frequencies)
    timestamps_to_consider = []
    past_count = 3

    for i in range(periodicities_count):
        for count in range(1, past_count + 1):
            # margin calculation
            margin = 0.02 * (frequencies[i] - 3600) + 900  # straight line on the basis of 15 minutes for 1 hour, 1 week for 1 year

            lower_limit = current_time - count * frequencies[i] - margin
            upper_limit = current_time - count * frequencies[i] + margin

            # # cost calculation
            # time_cost = time_cost_calc(current_time, (lower_limit + upper_limit) / 2)

            local_timezone = tzlocal.get_localzone()  # get pytz timezone
            lower_limit_local = datetime.fromtimestamp(lower_limit, local_timezone).strftime('%Y-%m-%d %H:%M:%S')
            upper_limit_local = datetime.fromtimestamp(upper_limit, local_timezone).strftime('%Y-%m-%d %H:%M:%S')

            # print("____")
            # print(lower_limit_local)
            # print(upper_limit_local)

            timestamps_to_consider.append((lower_limit, upper_limit, lower_limit_local, upper_limit_local))

    return timestamps_to_consider
