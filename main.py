import numpy as np
import userGrouping
import utils
import timeSlotAllocation
import timeFrequencyAllocation
import systemUtility






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



    systemUtility.usernumber_sweep()

    systemUtility.transmitpower_bandwidth_lambda_sweep()

    systemUtility.transmitpower_usernumber_sweep()

    systemUtility.bandwidth_radius_lambda_sweep()

    systemUtility.radius_sweep()








if __name__ == "__main__":
    main()