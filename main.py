import numpy as np
import userGrouping
import utils
import timeSlotAllocation
import timeFrequencyAllocation
from systemUtility import compute_user_utilities


def format_allocation_log(allocation_log):
    """
    Format and return the allocation log in a readable string format.

    :param allocation_log: List of tuples containing allocation information
    :return: Formatted string representation of the allocation log
    """


    result = "\nResource Allocation Log:\n" + "=" * 80 + "\n"

    # Sort by beam, time slot, and subchannel
    sorted_log = sorted(allocation_log, key=lambda x: (x[2], x[1], x[3]))

    current_beam = -1
    current_time_slot = -1

    for entry in sorted_log:
        user_id, time_slot, beam_id, subchannel_id, rate = entry

        if beam_id != current_beam:
            result += f"\nBeam {beam_id}:\n" + "-" * 80 + "\n"
            current_beam = beam_id
            current_time_slot = -1

        if time_slot != current_time_slot:
            result += f"\nTime Slot {time_slot}:\n"
            result += "User ID | Subchannel ID | Rate (Mbps)\n"
            result += "-" * 40 + "\n"
            current_time_slot = time_slot

        result += f"{user_id:7d} | {subchannel_id:12d} | {rate:10.2f}\n"

    return result



def main():
    """
    Main function to orchestrate the simulation process for user grouping, time slot allocation,
    and time-frequency resource distribution. It manages the sequential invocation of various
    modules to perform the operations, reporting progress and results while ensuring proper
    processing of user group data and resource allocation.

    This function follows the following steps:
    1. Group the users and create aggregate user instances.
    2. Allocate initial and residual time slots to aggregated users.
    3. Print and log time slot assignments.
    4. Initialize time-frequency resource structures.
    5. Allocate subchannels based on the aggregated user data.
    6. Report any failed user allocations.
    7. Print summary information about the simulation.

    :return: None
    """
    # Main logic starts here
    user_groups, virtual_centers = userGrouping.group_users()
    user_groups2 = user_groups
    aggregate_users = timeSlotAllocation.create_aggregate_users(user_groups)

    if utils.LOG_LEVEL >= 1:
        users = []
        for au in aggregate_users:
            au.print_user()
            users.extend(au.usr)
        print("#"*80)
        users.sort(key=lambda x: x.id)
        for user in users:
            user.print_user()
        print("#" * 80)

    au_time_slot_mapping, time_slot_au_mapping, epsilon, epsilon_used = timeSlotAllocation.initial_user_time_slot_assignment(
        aggregate_users)
    au_time_slot_mapping, time_slot_au_mapping, epsilon, epsilon_used = timeSlotAllocation.residual_time_slot_assignment(
        aggregate_users, au_time_slot_mapping, time_slot_au_mapping, epsilon, epsilon_used)

    timeSlotAllocation.print_time_slot_assignments(time_slot_au_mapping,au_time_slot_mapping, epsilon, epsilon_used)

    user_time_slot_beam_mapping, allocated_subchannels = timeFrequencyAllocation.initialize_timeFrequencyStructures(time_slot_au_mapping)

    timeFrequencyAllocation.allocate_subchannels(aggregate_users, user_time_slot_beam_mapping, allocated_subchannels)



    timeFrequencyAllocation.print_allocation_summary(allocated_subchannels, user_time_slot_beam_mapping,users)







    total_utility, per_user_utilities = compute_user_utilities(users, timeFrequencyAllocation.allocation_log)
    print("System Utility:", total_utility)

    for utility, user_id in per_user_utilities:
        print(f"User {user_id} Utility: {utility:.2f}")

    timeFrequencyAllocation.report_failed_users(aggregate_users)

    users2 = []
    for au in user_groups2:
        users2.extend(au)
    print("User Size Finally",users[1].size )
    print("User Size Initially",users2[1].size )

    # if utils.LOG_LEVEL >= 1:
    #     print(format_allocation_log(timeFrequencyAllocation.allocation_log))

    print("Simulation completed.")




if __name__ == "__main__":
    main()