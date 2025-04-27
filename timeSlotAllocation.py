import numpy as np
from scipy.optimize import linear_sum_assignment
import userGrouping
import utils


class AggregateUser:
    def __init__(self, gid, users):
        self.id   = gid
        self.usr  = users
        self.Sbar = sum(u.weight * u.size for u in users)
        self.Tbar = max(u.deadline for u in users)
        self.x  = np.mean([u.x for u in users])
        self.y  = np.mean([u.y for u in users])
    def print_user(self):
        print("Aggregate User Group Id: " ,self.id ," x_coord: " ,self.x ,"y_coord: " ,self.y,
              " Sbar: ",self.Sbar," Tbar: ",self.Tbar, f"Total User: {len(self.usr)} Users {[u.id for u in self.usr]}" )



def create_aggregate_users(user_groups):
    aggregate_users = []
    for i, user_set in enumerate(user_groups):
        aggregate_users.append(AggregateUser(i, user_set))
    return aggregate_users


def calculate_rate(vau, t, f = utils.CARRIER_FREQUENCY):
    satellite_x = satellite_y = (utils.USER_AREA / utils.TOTAL_SLOTS) * t
    # satellite is moving from [0,0] to [max_x,max_y] in a diagonal line.
    distance = np.sqrt((vau.x - satellite_x) ** 2 + (vau.y - satellite_y) ** 2 + utils.SATELLITE_ALTITUDE ** 2) * 1000
    g_t = 1000  # satellite transmit antenna gain in linear scale assuming 0.4 degree beamwidth
    g_r = 31.62  # user receiver antenna gain in linear scale
    p = 40  # satellite transmit power
    N_0 = 3.98 * 10 ** -18 # Noise spectral density -174 dbw/Hz
    c = 3 * 10 ** 8 # speed of light
    L = (c / (4 * np.pi * f * distance)) ** 2 # free space path loss
    x = (p * g_t * g_r * L) / (N_0 * utils.BANDWIDTH)
    rate = utils.BANDWIDTH * np.log(1+x)
    return rate

def needs_assignment(au_m : AggregateUser, t, epsilon, epsilon_used, au_mapping):

    return epsilon_used[au_m.id] < epsilon[au_m.id] and au_m.Tbar >= t and t not in au_mapping[au_m.id]


def initial_user_time_slot_assignment(aggregate_users):
    # proportional fairness
    N0 = utils.TOTAL_BEAM_NUMBER * utils.TOTAL_SLOTS
    S = np.array([au.Sbar for au in aggregate_users])
    epsilon = np.round(N0 * S / S.sum()).astype(int)  # beam‑slot quota per group
    epsilon_used = np.zeros(len(epsilon))
    a_m = np.ceil(epsilon / utils.TOTAL_BEAM_NUMBER).astype(int)

    print("epsilon: ", epsilon)
    print("a_m: ", a_m)

    # virtualise aggregate users
    VAUs = []
    for m, au in enumerate(aggregate_users):
        VAUsm = []
        for j in range(a_m[m]):
            VAUs.append((m, j))  # tuple (group_id, copy)

    # K-M Assignment algorithm
    C = np.zeros((len(VAUs), utils.TOTAL_SLOTS))
    for i, (group_id, _) in enumerate(VAUs):
        au = aggregate_users[group_id]
        for t in range(utils.TOTAL_SLOTS):
            rate = calculate_rate(au, t)
            C[i, t] = -rate  # Negative for Hungarian (minimize cost)

    row_idx, col_idx = linear_sum_assignment(C)
    print("row_idx: ", row_idx)
    print("col_idx: ", col_idx)

    # Interpret results:
    au_time_slot_mapping = {}
    time_slot_au_mapping = {}

    for r, c in zip(row_idx, col_idx):
        AU_id, _ = VAUs[r]
        assigned_slot = c
        if c not in time_slot_au_mapping:
            time_slot_au_mapping[c] = []
        time_slot_au_mapping[c].append(aggregate_users[AU_id])

        if AU_id not in au_time_slot_mapping:
            au_time_slot_mapping[AU_id] = []
        au_time_slot_mapping[AU_id].append(int(assigned_slot))

    print("Initial assignment:")
    for AU_id in au_time_slot_mapping:
        epsilon_used[AU_id] =(int(len(au_time_slot_mapping[AU_id])))
        print(f"AU {AU_id} is assigned to slots {au_time_slot_mapping[AU_id]}")

    return au_time_slot_mapping, time_slot_au_mapping, epsilon, epsilon_used


def residual_time_slot_assignment(aggregate_users, au_time_slot_mapping, time_slot_au_mapping, epsilon, epsilon_used):
    print("epsilon", epsilon)
    print("epsilon_used: ", epsilon_used)

    for t in range(utils.TOTAL_SLOTS):
        already_assigned = time_slot_au_mapping[t]

        free_beams = utils.TOTAL_BEAM_NUMBER - 1
        candidates = []
        for au_m in aggregate_users:
            if au_m not in already_assigned and needs_assignment(au_m, t, epsilon, epsilon_used, au_time_slot_mapping):
                candidates.append(au_m)

        candidates = [au_m for au_m in aggregate_users if
                      au_m.id not in already_assigned and needs_assignment(au_m, t, epsilon, epsilon_used,
                                                                           au_time_slot_mapping)]
        candidates.sort(key=lambda m: calculate_rate(m, t), reverse=True)

        for m in candidates[:free_beams]:
            epsilon_used[m.id] += 1
            time_slot_au_mapping[t].append(m)
            au_time_slot_mapping[m.id].append(t)

    return  au_time_slot_mapping,time_slot_au_mapping, epsilon, epsilon_used



# user_groups, virtual_centers = userGrouping.group_users()
# aggregate_users = create_aggregate_users(user_groups)
#
# [au.print_user() for au in aggregate_users]
#
#
# au_time_slot_mapping, time_slot_au_mapping, epsilon, epsilon_used = initial_user_time_slot_assignment(aggregate_users)
#
# au_time_slot_mapping, time_slot_au_mapping, epsilon, epsilon_used = residual_time_slot_assignment(aggregate_users, au_time_slot_mapping, time_slot_au_mapping, epsilon, epsilon_used)
#
#
#
# for time_slot in sorted(time_slot_au_mapping.keys()):
#     print(f"Time slot {time_slot} is assigned to AU {[au.id for au in time_slot_au_mapping[time_slot]]}")
#
# print("epsilon", epsilon)
# print("epsilon_used: ", epsilon_used)