import numpy as np
import userGrouping
import utils
import timeSlotAllocation
import timeFrequencyAllocation


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
    aggregate_users = timeSlotAllocation.create_aggregate_users(user_groups)

    for au in aggregate_users:
        au.print_user()

    au_time_slot_mapping, time_slot_au_mapping, epsilon, epsilon_used = timeSlotAllocation.initial_user_time_slot_assignment(
        aggregate_users)
    au_time_slot_mapping, time_slot_au_mapping, epsilon, epsilon_used = timeSlotAllocation.residual_time_slot_assignment(
        aggregate_users, au_time_slot_mapping, time_slot_au_mapping, epsilon, epsilon_used)

    timeSlotAllocation.print_time_slot_assignments(time_slot_au_mapping, epsilon, epsilon_used)

    user_time_slot_beam_mapping, allocated_subchannels = timeFrequencyAllocation.initialize_timeFrequencyStructures(time_slot_au_mapping)

    timeFrequencyAllocation.allocate_subchannels(aggregate_users, user_time_slot_beam_mapping, allocated_subchannels)

    for beam in range(utils.TOTAL_BEAM_NUMBER):
        print(f"Allocated subchannels for beam {beam}: \n", allocated_subchannels[beam])

    timeFrequencyAllocation.report_failed_users(aggregate_users)


    print("Simulation completed.")



if __name__ == "__main__":
    main()