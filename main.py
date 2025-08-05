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
   utils.reset()






   systemUtility.usernumber_lambda_sweep()

   systemUtility.transmitpower_bandwidth_lambda_sweep()

   systemUtility.transmitpower_usernumber_sweep()

   systemUtility.bandwidth_radius_lambda_sweep()


   systemUtility.radius_vs_utility(100, 800, 50)


   systemUtility.carrierFreq_vs_utility(10, 50, 2)


   systemUtility.usernumber_vs_utility(20,600,10)


   systemUtility.transmitpower_vs_utility(20,100,2)


   systemUtility.bandwidth_vs_utility(10,200,2)


   systemUtility.bandwidth_vs_rate(10,200,5)


   systemUtility.transmitpower_vs_rate(10,60,5)


   systemUtility.tx_antenna_gains_vs_rate(15,60,5)


   systemUtility.rx_antenna_gains_vs_rate(15, 60, 5)


   systemUtility.sat_altitude_vs_rate(500,2500,50)


   systemUtility.carrier_freq_vs_rate(10,50,2)


















if __name__ == "__main__":
   main()



