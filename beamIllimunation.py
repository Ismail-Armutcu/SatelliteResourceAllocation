import numpy as np
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
    aggreate_users = []
    for i, user_set in enumerate(user_groups):
        aggreate_users.append(AggregateUser(i, user_set))
    return aggreate_users



user_groups, virtual_centers = userGrouping.group_users()
aggregate_users = create_aggregate_users(user_groups)

[au.print_user() for au in aggregate_users]

# proportional fairness
N0     = utils.TOTAL_BEAM_NUMBER*utils.TOTAL_SLOTS
S      = np.array([au.Sbar for au in aggregate_users])
epsilon   = np.round(N0 * S / S.sum()).astype(int)   # beam‑slot quota per group

print("epsilon: ",epsilon)

# virtualise aggregate users

VAUs = []
for m, au in enumerate(aggregate_users):
    a_m = int(np.ceil(epsilon[m] / utils.TOTAL_BEAM_NUMBER))
    for j in range(a_m):
        VAUs.append((m, j))       # tuple (group_id, copy)

# calculate the SNR and Rate based on the parameters from the paper
