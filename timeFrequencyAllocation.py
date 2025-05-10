import numpy as np
import userGrouping
import utils
import timeSlotAllocation

UNALLOCATED_SUBCHANNEL = -1  # Introduced constant for clarity


def determine_subchannel_rates(user, time_slot, subchannel):
    """
    Determines the rate of data transmission for a given user and subchannel in
    a specified time slot. The function calculates the frequency of the
    subchannel and determines the transmission rate based on that frequency.

    :param user: The user for whom the transmission rate is calculated.
    :type user: Any
    :param time_slot: The time slot during which the transmission is happening.
    :type time_slot: Any
    :param subchannel: A numerical identifier for the subchannel being used.
    :type subchannel: int
    :return: The calculated rate of data transmission for the user on the given
             subchannel and time slot.
    :rtype: Any
    """
    starting_freq = utils.CARRIER_FREQUENCY - (utils.BANDWIDTH // utils.SUBCHANNEL_NUMBER)
    subchannel_freq = starting_freq + 2 * 10 ** 6 * subchannel + 10 ** 6
    return timeSlotAllocation.calculate_rate(user, time_slot, subchannel_freq)


def allocate_subchannels(aggregate_users, user_time_slot_beam_mapping, allocated_subchannels):
    """
    Allocates subchannels to users based on priority, ensuring deadlines are met where possible.
    Prioritizes users by weight and deadlines, then allocates resources using rate calculations
    per time-slot and subchannel. Unallocated subchannels remain available for future assignment.

    :param aggregate_users: List of AggregateUser objects, each containing a group of users to be processed.
    :type aggregate_users: list[AggregateUser]
    :param user_time_slot_beam_mapping: Mapping of user IDs to their scheduled time slots and beam identifiers.
    :type user_time_slot_beam_mapping: dict[int, list[tuple[int, int]]]
    :param allocated_subchannels: A 3D array tracking subchannel allocations, indexed by beam, time slot, and subchannel.
    :type allocated_subchannels: numpy.ndarray
    :return: None
    :rtype: NoneType
    """
    for au in aggregate_users:
        high_priority_users = [u for u in au.usr if u.weight >= utils.USER_WEIGHT_THRESHOLD]
        low_priority_users = [u for u in au.usr if u.weight < utils.USER_WEIGHT_THRESHOLD]

        high_priority_users.sort(key=lambda u: (-u.weight, u.deadline))
        low_priority_users.sort(key=lambda u: (-u.weight, u.deadline))
        user_priority_queue = high_priority_users + low_priority_users

        print(f"Aggregate User Group Id: {au.id}, Users: {[u.id for u in au.usr]}")

        for user in user_priority_queue:
            user_rates = np.zeros((utils.TOTAL_SLOTS, utils.SUBCHANNEL_NUMBER))
            for t in range(utils.TOTAL_SLOTS):
                for f in range(utils.SUBCHANNEL_NUMBER):
                    user_rates[t, f] = determine_subchannel_rates(user, t, f)

            max_rate_index = np.unravel_index(np.argmax(user_rates, axis=None), user_rates.shape)
            if np.ceil(user.size * 1000 / user_rates[max_rate_index]) > user.deadline:
                # Cannot allocate resources, deadline cannot be met
                print(
                    f"User {user.id} cannot be scheduled: Size: {user.size * 1000}, Deadline: {user.deadline}, Max rate: {user_rates[max_rate_index]}")
                continue

            for time_slot, beam in user_time_slot_beam_mapping[user.id]:
                max_subchannel = np.argmax(user_rates[time_slot])
                while user.size > 0:
                    if allocated_subchannels[beam, time_slot, max_subchannel] == UNALLOCATED_SUBCHANNEL:
                        allocated_subchannels[beam, time_slot, max_subchannel] = user.id
                        user.size -= user_rates[time_slot, max_subchannel] / 1000
                        user_rates[time_slot, max_subchannel] = 0
                        break
                    else:
                        user_rates[time_slot, max_subchannel] = 0
                        max_subchannel = np.argmax(user_rates[time_slot])
                        if np.all(user_rates[time_slot] == 0):
                            break

def report_failed_users(aggregate_users):
    """
    Generates and prints a report on failed users based on their usage size.

    Identifies users with a usage size greater than zero from the provided
    aggregate users' data, calculates the number of such users, and logs
    their IDs and their remaining rates.

    :param aggregate_users: List of aggregate user objects. Each aggregate
        user object should have a "usr" attribute containing a list of user
        objects. Each user object should have "id" and "size" attributes.
    :type aggregate_users: list
    :return: None
    """
    failed_users = [user for au in aggregate_users for user in au.usr if user.size > 0]
    print(f"{len(failed_users)} Failed users: {[user.id for user in failed_users]}")
    print(f"Remaining rates: {[float(user.size) for user in failed_users]}")

def initialize_timeFrequencyStructures(time_slot_au_mapping):
    """
    Initialize the data structures required for managing time-frequency allocations
    and mappings between users, time slots, and beams. This function processes the
    given mapping of aggregate users for each time slot and builds a user-centric
    mapping structure, allowing efficient indexing by user ID. Additionally, it
    sets up a data structure for tracking subchannel allocation status across beams
    and time slots.

    :param time_slot_au_mapping: A mapping of time slots to aggregate user objects.
        Each entry represents a list of beams, where each beam contains an aggregate
        user object that holds user IDs.
    :type time_slot_au_mapping: list[list[AggregateUser]]

    :return: None
    """
    user_time_slot_beam_mapping = [[] for _ in range(utils.USER_NUMBER)]
    for time_slot in range(utils.TOTAL_SLOTS):
        for beam, aggregate_user in enumerate(time_slot_au_mapping[time_slot]):
            for gu in aggregate_user.usr:
                user_time_slot_beam_mapping[gu.id].append((time_slot, beam))

    allocated_subchannels = np.full((utils.TOTAL_BEAM_NUMBER, utils.TOTAL_SLOTS, utils.SUBCHANNEL_NUMBER),
                                    UNALLOCATED_SUBCHANNEL)
    return user_time_slot_beam_mapping, allocated_subchannels