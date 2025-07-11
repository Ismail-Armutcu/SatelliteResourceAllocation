import numpy as np
import matplotlib.pyplot as plt
import utils
import timeSlotAllocation
import timeFrequencyAllocation
import userGrouping

def compute_user_utilities(users, allocation_log, lambda_1=utils.LAMBDA_1, lambda_2=0.3):
    """
    Calculate system utility based on successful transmissions and energy consumption.

    :param users: list of User objects
    :param allocation_log: list of tuples (user_id, time_slot, beam, subchannel, rate_kbps)
    :param lambda_1: Weight for reward
    :param lambda_2: Weight for energy
    :param slot_duration: Time duration per slot in seconds
    :return: system_utility, detailed_user_utilities
    """
    user_info = {user.id: {'size': user.size, 'weight': user.weight, 'deadline': user.deadline} for user in users}
    user_data_sent = {uid: 0 for uid in user_info}
    user_energy = {uid: 0 for uid in user_info}
    user_completion_time = {}

    # Constants
    power_watt = utils.TRANSMIT_POWER  # Assumed constant transmit power

    for uid, t, beam, sc, rate in allocation_log:
        if uid not in user_data_sent:
            continue
        sent_this_slot = rate * utils.TIME_SLOT_DURATION / 1000  # in bits
        user_data_sent[uid] += sent_this_slot
        user_energy[uid] += power_watt *  utils.TIME_SLOT_DURATION / 1000 # Joules
        if user_data_sent[uid] >= user_info[uid]['size'] and uid not in user_completion_time:
            user_completion_time[uid] = t

    # Determine max/min energy used for normalization
    energy_vals = list(user_energy.values())
    E_max, E_min = max(energy_vals), min(energy_vals)

    user_utilities = []
    total_utility = 0.0
    for uid in user_info:
        deadline = user_info[uid]['deadline']
        size = user_info[uid]['size']
        weight = user_info[uid]['weight']

        completed = user_data_sent[uid] >= size
        completed_in_time = completed and user_completion_time.get(uid, float('inf')) <= deadline
        alpha_n = 1 if completed_in_time else 0
        eta_n = weight * alpha_n

        E_n = (E_max - user_energy[uid]) / (E_max - E_min + 1e-8) if E_max > E_min else 1
        U_n = utils.LAMBDA_1 * eta_n - lambda_2 * E_n
        user_utilities.append((U_n,uid))
        total_utility+=U_n

    if utils.LOG_LEVEL >= 1:
        print("\nAllocation Log:")
        print("User ID | Time Slot | Beam | Subchannel | Rate (Mbps) | Completion Time | Deadline | Status")
        print("-" * 80)
    for entry in allocation_log:
        uid, t, beam, sc, rate = entry
        if uid not in user_info:
            continue

        completion_status = "Completed" if uid in user_completion_time and user_completion_time[uid] <= user_info[uid]['deadline'] else "Failed"
        if utils.LOG_LEVEL >= 2:
            print(f"{uid:7d} | {t:9d} | {beam:4d} | {sc:10d} | {rate:11.2f} | {user_completion_time.get(uid, 'N/A'):15} | {user_info[uid]['deadline']:8d} | {completion_status}")

        # Set failed flag for users who did not complete in time
        if completion_status == "Failed":
            for user in users:
                if user.id == uid:
                    user.is_failed = True
                    break
    return total_utility, user_utilities

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


def calculate_utility(userNumber=100, transmitPower=40, lambda_1=0.6, bandwidth=20e6, rngSeed=42,
                      beamRadius=3.5 * 100) -> tuple[float, float]:
    # Main logic starts here
    utils.USER_NUMBER = userNumber
    utils.TRANSMIT_POWER = transmitPower
    utils.LAMBDA_1 = lambda_1
    utils.BANDWIDTH = bandwidth
    utils.RNG_SEED = rngSeed
    utils.BEAM_RADIUS = beamRadius
    user_groups, virtual_centers = userGrouping.group_users()
    user_groups2 = user_groups
    aggregate_users = timeSlotAllocation.create_aggregate_users(user_groups)
    users = []
    for au in aggregate_users:
        if utils.LOG_LEVEL >= 1:
            au.print_user()
        users.extend(au.usr)
    if utils.LOG_LEVEL >= 1:
        print("#" * 80)
        users.sort(key=lambda x: x.id)
        for user in users:
            user.print_user()
        print("#" * 80)
    au_time_slot_mapping, time_slot_au_mapping, epsilon, epsilon_used = timeSlotAllocation.initial_user_time_slot_assignment(
        aggregate_users)
    au_time_slot_mapping, time_slot_au_mapping, epsilon, epsilon_used = timeSlotAllocation.residual_time_slot_assignment(
        aggregate_users, au_time_slot_mapping, time_slot_au_mapping, epsilon, epsilon_used)
    timeSlotAllocation.print_time_slot_assignments(time_slot_au_mapping, au_time_slot_mapping, epsilon, epsilon_used)
    user_time_slot_beam_mapping, allocated_subchannels = timeFrequencyAllocation.initialize_timeFrequencyStructures(
        time_slot_au_mapping)
    timeFrequencyAllocation.allocate_subchannels(aggregate_users, user_time_slot_beam_mapping, allocated_subchannels)

    timeFrequencyAllocation.print_allocation_summary(allocated_subchannels, user_time_slot_beam_mapping, users)
    beam_sum = 0
    for beam in range(utils.TOTAL_BEAM_NUMBER):
        for time in range(utils.TOTAL_SLOTS):
            for element in allocated_subchannels[beam][time]:
                if element != -1:
                    beam_sum += 1
    allocation_efficiency = beam_sum / (utils.TOTAL_BEAM_NUMBER * utils.TOTAL_SLOTS * utils.SUBCHANNEL_NUMBER) * 100

    total_utility, per_user_utilities = compute_user_utilities(users, timeFrequencyAllocation.allocation_log)
    if utils.LOG_LEVEL >= 1:
        for utility, user_id in per_user_utilities:
            print(f"User {user_id} Utility: {utility:.2f}")
    timeFrequencyAllocation.report_failed_users(aggregate_users)

    if utils.LOG_LEVEL >= 1:
        print(format_allocation_log(timeFrequencyAllocation.allocation_log))
    if utils.LOG_LEVEL >= 1:
        print("Simulation completed.")
        print(f"Simulation Parameters: ")
        print(f"User Number: {utils.USER_NUMBER}, Transmit Power: {utils.TRANSMIT_POWER} W, "
              f"Lambda 1: {utils.LAMBDA_1}, Bandwidth: {utils.BANDWIDTH / 1e6} MHz, RNG Seed: {utils.RNG_SEED} beamRadius: {beamRadius} km")
        print("System Utility:", total_utility)
    timeFrequencyAllocation.allocation_log = [] # Clear the allocation log for next run

    return total_utility, allocation_efficiency


def usernumber_sweep():
    userNumberSweepListUtility = []
    userNumberSweepListAllocEfficiency = []
    for lambda_1 in np.arange(0.5, 0.79, 0.1):
        userNumberSweepList_lambda_utility = []
        userNumberSweepList_lambda_alloc_efficiency = []
        for userNumber in range(20, 80, 5):
            utility, allocation_efficiency = calculate_utility(userNumber=userNumber, lambda_1=float(lambda_1))
            userNumberSweepList_lambda_alloc_efficiency.append(allocation_efficiency)
            userNumberSweepList_lambda_utility.append(utility)
        userNumberSweepListAllocEfficiency.append(userNumberSweepList_lambda_alloc_efficiency)
        userNumberSweepListUtility.append(userNumberSweepList_lambda_utility)
    print("User Number Sweep Results:")
    for i in userNumberSweepListUtility:
        print(i)
    print("User Number Sweep Results (Allocation Efficiency):")
    for i in userNumberSweepListAllocEfficiency:
        print(i)
    # Plot user number sweep results
    plt.figure(figsize=(10, 6))
    x = range(20, 80, 5)
    lambda_values = np.arange(0.5, 0.79, 0.1)
    for idx, data in enumerate(userNumberSweepListUtility):
        plt.plot(x, data, marker='o', label=f'λ = {lambda_values[idx]:.1f}')
    plt.xlabel('Number of Users')
    plt.ylabel('Utility')
    plt.title('System Utility Function vs Number of Users')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    x = range(20, 80, 5)
    lambda_values = np.arange(0.5, 0.79, 0.1)
    for idx, data in enumerate(userNumberSweepListAllocEfficiency):
        plt.plot(x, data, marker='o', label=f'λ = {lambda_values[idx]:.1f}')
    plt.xlabel('Number of Users')
    plt.ylabel('Allocation Efficiency (%)')
    plt.title('Allocation Efficiency vs Number of Users')
    plt.grid(True)
    plt.legend()
    plt.show()



def transmitpower_bandwidth_lambda_sweep():
    transmitPower_bandwidth_lambda_SweepList_Utility = []
    transmitPower_bandwidth_lambda_SweepList_Alloc_Efficiency = []
    transmit_power_range = np.arange(20, 42, 2)
    for lambda_1 in np.arange(0.6, 0.79, 0.1):
        for bandwidth in np.arange(20e6, 40e6, 10e6):
            transmitPowerSweepList_temp_utility = []
            transmitPowerSweepList_temp_alloc_efficiency = []
            for transmitPower in transmit_power_range:
                utility, allocation_efficiency = calculate_utility(transmitPower=transmitPower, lambda_1=lambda_1, bandwidth=bandwidth)
                transmitPowerSweepList_temp_utility.append(utility)
                transmitPowerSweepList_temp_alloc_efficiency.append(allocation_efficiency)
            transmitPower_bandwidth_lambda_SweepList_Utility.append(transmitPowerSweepList_temp_utility)
            transmitPower_bandwidth_lambda_SweepList_Alloc_Efficiency.append(transmitPowerSweepList_temp_alloc_efficiency)
    print("transmitPower_bandwidth_lambda Sweep Results:")
    for i in transmitPower_bandwidth_lambda_SweepList_Utility:
        print(i)
    # Plot utility vs transmit power for specified combinations

    combinations = [
        ('B=20MHz, λ=0.6'),
        ('B=30MHz, λ=0.6'),
        ('B=20MHz, λ=0.7'),
        ('B=30MHz, λ=0.7')
    ]
    x = np.arange(20, 42, 2)
    plt.figure(figsize=(10, 6))
    for idx, result in enumerate(transmitPower_bandwidth_lambda_SweepList_Utility):
        plt.plot(x, result, marker='o', label=combinations[idx])
    plt.xlabel('Transmit Power (W)')
    plt.ylabel('System Utility')
    plt.title('System Utility vs Transmit Power for Different Bandwidth and λ Values')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    for idx, result in enumerate(transmitPower_bandwidth_lambda_SweepList_Alloc_Efficiency):
        plt.plot(x, result, marker='o', label=combinations[idx])
    plt.xlabel('Transmit Power (W)')
    plt.ylabel('Allocation Efficiency (%)')
    plt.title('Allocation Efficiency (%) vs Transmit Power for Different Bandwidth and λ Values')
    plt.grid(True)
    plt.legend()
    plt.show()


def transmitpower_usernumber_sweep():
    transmitPower_UserNumber_SweepList_utility = []
    transmitPower_UserNumber_SweepList_alloc_efficiency = []
    for userNumber in np.arange(40, 70, 10):
        transmitPowerSweepList_temp_utility = []
        transmitPowerSweepList_temp_alloc_efficiency = []
        for transmitPower in range(20, 42, 2):
            utility, allocation_efficiency = calculate_utility(transmitPower=transmitPower, userNumber=userNumber)
            transmitPowerSweepList_temp_utility.append(utility)
            transmitPowerSweepList_temp_alloc_efficiency.append(allocation_efficiency)
        transmitPower_UserNumber_SweepList_utility.append(transmitPowerSweepList_temp_utility)
        transmitPower_UserNumber_SweepList_alloc_efficiency.append(transmitPowerSweepList_temp_alloc_efficiency)
    print("transmitPower_userNumber Sweep Results:")
    for i in transmitPower_UserNumber_SweepList_utility:
        print(i)

    combinations = [
        ('N = 40'),
        ('N = 50'),
        ('N = 60'),
        ('N = 70')
    ]
    x = np.arange(20, 42, 2)
    plt.figure(figsize=(10, 6))
    for idx, result in enumerate(transmitPower_UserNumber_SweepList_utility):
        plt.plot(x, result, marker='o', label=combinations[idx])
    plt.xlabel('Transmit Power (W)')
    plt.ylabel('System Utility')
    plt.title('System Utility vs Transmit Power for Different User Numbers and λ1 = 0.6')
    plt.grid(True)
    plt.legend()
    plt.show()


    plt.figure(figsize=(10, 6))
    for idx, result in enumerate(transmitPower_UserNumber_SweepList_alloc_efficiency):
        plt.plot(x, result, marker='o', label=combinations[idx])
    plt.xlabel('Transmit Power (W)')
    plt.ylabel('Allocation Efficiency (%)')
    plt.title('Allocation Efficiency (%) vs Transmit Power for Different User Numbers and λ1 = 0.6')
    plt.grid(True)
    plt.legend()
    plt.show()


def bandwidth_radius_lambda_sweep():
    bandwidth_radius_lamdba_SweepList_utility = []
    bandwidth_radius_lamdba_SweepList_alloc_efficiency = []
    for radius in np.arange(150, 170, 15):
        for lambda_1 in np.arange(0.5, 0.7, 0.1):
            bandwidth_radius_lamdba_SweepList_Temp_utility = []
            bandwidth_radius_lamdba_SweepList_Temp_alloc_efficiency = []
            for bandwidth in np.arange(15e6, 40e6, 5e6):
                utility, allocation_efficiency = calculate_utility(beamRadius=radius, bandwidth=bandwidth, lambda_1=lambda_1)
                bandwidth_radius_lamdba_SweepList_Temp_utility.append(utility)
                bandwidth_radius_lamdba_SweepList_Temp_alloc_efficiency.append(allocation_efficiency)
            bandwidth_radius_lamdba_SweepList_utility.append(bandwidth_radius_lamdba_SweepList_Temp_utility)
            bandwidth_radius_lamdba_SweepList_alloc_efficiency.append(bandwidth_radius_lamdba_SweepList_Temp_alloc_efficiency)
    print("bandwidth_radius_lambda Sweep Results:")
    for i in bandwidth_radius_lamdba_SweepList_utility:
        print(i)

    combinations = [
        ('r=150km, λ=0.5'),
        ('r=150km, λ=0.6'),
        ('r=165km, λ=0.5'),
        ('r=165km, λ=0.6')
    ]
    x = np.arange(15, 40, 5)

    plt.figure(figsize=(10, 6))
    for idx, result in enumerate(bandwidth_radius_lamdba_SweepList_utility):
        plt.plot(x, result, marker='o', label=combinations[idx])
    plt.xlabel('Bandwidth (B) Mhz')
    plt.ylabel('System Utility')
    plt.title('System Utility vs Bandwidth')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    for idx, result in enumerate(bandwidth_radius_lamdba_SweepList_alloc_efficiency):
        plt.plot(x, result, marker='o', label=combinations[idx])
    plt.xlabel('Bandwidth (B) Mhz')
    plt.ylabel('Allocation Efficiency (%)')
    plt.title('Allocation Efficiency (%) vs Bandwidth')
    plt.grid(True)
    plt.legend()
    plt.show()