
import numpy as np
import utils

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
            # Skip or log a warning when an allocation log entry references an unknown user
            if utils.LOG_LEVEL >= 1:
                print(f"Warning: User id {uid} in allocation_log but missing from users list -- skipping this record.")
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
        U_n = lambda_1 * eta_n - lambda_2 * E_n
        user_utilities.append((U_n,uid))
        total_utility+=U_n



    if utils.LOG_LEVEL >= 1:
        print("\nAllocation Log:")
        print("User ID | Time Slot | Beam | Subchannel | Rate (Mbps) | Completion Time | Deadline | Status")
        print("-" * 80)
    for entry in allocation_log:
        uid, t, beam, sc, rate = entry
        completion_status = "Completed" if uid in user_completion_time and user_completion_time[uid] <= user_info[uid][
            'deadline'] else "Failed"
        if utils.LOG_LEVEL >= 1:
            print(f"{uid:7d} | {t:9d} | {beam:4d} | {sc:10d} | {rate:11.2f} | {user_completion_time.get(uid, 'N/A'):15} | {user_info[uid]['deadline']:8d} | {completion_status}")

        # Set failed flag for users who did not complete in time
        if completion_status == "Failed":
            for user in users:
                if user.id == uid:
                    user.is_failed = True
                    break
    return total_utility, user_utilities
