import numpy as np
import userGrouping
import utils
import timeSlotAllocation


def calculate_subchannel_rates(au, t, f):
    starting_freq = utils.CARRIER_FREQUENCY - (utils.BANDWIDTH // utils.SUBCHANNEL_NUMBER)
    subchannel_freq = starting_freq +  2 * 10**6 * f + 10**6
    rate = timeSlotAllocation.calculate_rate(au, t, subchannel_freq)
    return rate

user_groups, virtual_centers = userGrouping.group_users()
aggregate_users = timeSlotAllocation.create_aggregate_users(user_groups)

[au.print_user() for au in aggregate_users]


au_time_slot_mapping, time_slot_au_mapping, epsilon, epsilon_used = timeSlotAllocation.initial_user_time_slot_assignment(aggregate_users)

au_time_slot_mapping, time_slot_au_mapping, epsilon, epsilon_used = timeSlotAllocation.residual_time_slot_assignment(aggregate_users, au_time_slot_mapping, time_slot_au_mapping, epsilon, epsilon_used)

user_time_slot_beam_mapping = [[] for _ in range(utils.USER_NUMBER)]

for t in range(utils.TOTAL_SLOTS):
    aggregate_users_in_slot = time_slot_au_mapping[t]
    for beam,aggregate_user in enumerate(aggregate_users_in_slot):
        for gu in aggregate_user.usr:
            user_time_slot_beam_mapping[gu.id].append((t,beam))


for time_slot in sorted(time_slot_au_mapping.keys()):
    print(f"Time slot {time_slot} is assigned to AU {[au.id for au in time_slot_au_mapping[time_slot]]}")

print("epsilon", epsilon)
print("epsilon_used: ", epsilon_used)

allocated_subchannels = np.full((utils.TOTAL_BEAM_NUMBER, utils.TOTAL_SLOTS, utils.SUBCHANNEL_NUMBER), -1)

for au in aggregate_users:
    for gu in au.usr:
        gu.print_user()


for au in aggregate_users:
    high_priority_users = [u for u in au.usr if u.weight >= utils.USER_WEIGHT_THRESHOLD]
    low_priority_users = [u for u in au.usr if u.weight < utils.USER_WEIGHT_THRESHOLD]

    high_priority_users.sort(key=lambda u: (-u.weight, u.deadline))

    low_priority_users.sort(key=lambda u: (-u.weight, u.deadline))
    
    user_priority_queue = high_priority_users + low_priority_users
    print(f"Aggregate User Group Id: {au.id} users in group {[gu.id for gu in au.usr] }")


    for gu in user_priority_queue:
        gu_rates = np.zeros((utils.TOTAL_SLOTS, utils.SUBCHANNEL_NUMBER))

        for t in range(utils.TOTAL_SLOTS):
            for f in range(utils.SUBCHANNEL_NUMBER):
                gu_rates[t, f] = calculate_subchannel_rates(gu, t, f)


        max_index = np.unravel_index(np.argmax(gu_rates, axis=None), gu_rates.shape)
        if np.ceil(gu.size*1000 / gu_rates[max_index]) > gu.deadline:
            # do not allocate resources to gu, deadline cannot be met cannot be
            print(f"gu.id {gu.id} gu.size {gu.size * 1000} gu.deadline {gu.deadline} gu max rate {gu_rates[max_index]} cannot be met")
            continue
        else:
            for t, beam in user_time_slot_beam_mapping[gu.id]:

                max_subchannel_index = np.argmax(gu_rates[t])  # Find index of max element in gu_rates[t]

                if allocated_subchannels[beam,t, max_subchannel_index] == -1 and gu.size >= 0:
                    allocated_subchannels[beam, t, max_subchannel_index] = gu.id
                    gu.size -= gu_rates[t, max_subchannel_index]/1000
                    gu_rates[t, max_subchannel_index] = 0
                elif allocated_subchannels[beam,t, max_subchannel_index] != -1 and gu.size >= 0:
                    for f in range(utils.SUBCHANNEL_NUMBER):
                        gu_rates[t, max_subchannel_index] = 0
                        max_subchannel_index = np.argmax(gu_rates[t])
                        if allocated_subchannels[beam,t, max_subchannel_index] == -1:
                            allocated_subchannels[beam, t, max_subchannel_index] = gu.id
                            gu.size -= gu_rates[t, max_subchannel_index]/1000
                            break
                        if np.all(gu_rates[t] == 0):
                            break






print("allocated_subchannels for beam0: ", allocated_subchannels[0])
print("allocated_subchannels for beam1: ", allocated_subchannels[1])
print("allocated_subchannels for beam2: ", allocated_subchannels[2])
print("allocated_subchannels for beam3: ", allocated_subchannels[3])


failed_users = []

for au in aggregate_users:
    for gu in au.usr:
        if gu.size > 0:
            failed_users.append(gu)


print(f"{len(failed_users)} Failed users: {[gu.id for gu in failed_users]}")
print(f"rates {[gu.size for gu in failed_users]}")


