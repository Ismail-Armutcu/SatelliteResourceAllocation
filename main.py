import numpy as np
import userGrouping
import utils
import timeSlotAllocation
import timeFrequencyAllocation
import systemUtility



def calculate_rate():
    f = utils.CARRIER_FREQUENCY  # Hz (Ku-band downlink)
    distance = utils.SATELLITE_ALTITUDE * 1000  # meters (LEO orbit)
    p = utils.TRANSMIT_POWER  # Watts (per beam transmit power)
    g_t = 10 ** (utils.TX_ANTENNA_GAIN/10)  # satellite antenna gain (40 dBi)
    g_r = 10 ** (utils.RX_ANTENNA_GAIN/10)  # user terminal gain (40 dBi)
    B = utils.BANDWIDTH  # Hz (100 MHz bandwidth per user)
    N_0 = 10 ** (-174/10)  # W/Hz (noise spectral density)
    c = 3e8  # m/s (speed of light)
    L = (c / (4 * np.pi * f * distance)) ** 2 # free space path loss
    x = (p * g_t * g_r * L) / (N_0 * B)
    rate = B * np.log2(1+x)
    return rate / 1e6 #return rate in Mbps


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

    # systemUtility.transmitpower_bandwidth_lambda_sweep()
    #
    # systemUtility.transmitpower_usernumber_sweep()
    #
    # systemUtility.bandwidth_radius_lambda_sweep()

    # systemUtility.radius_sweep()
    #
    # systemUtility.carrierFreq_sweep()









if __name__ == "__main__":
    main()