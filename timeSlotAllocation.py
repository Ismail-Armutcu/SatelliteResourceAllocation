import numpy as np
from scipy.optimize import linear_sum_assignment
import userGrouping
import utils


class AggregateUser:
    """
    Represents an aggregate user group comprising multiple individual users.

    The purpose of this class is to aggregate data from multiple users, calculate
    statistical and aggregated metrics based on their individual attributes, and
    provide access to relevant collective information about the group. This can be
    helpful for tasks that require handling groups of users collectively rather than
    individually.

    :ivar id: Unique group identifier for the aggregate user group.
    :ivar usr: List of users that belong to this aggregate user group.
    :ivar Sbar: Weighted sum of product of `weight` and `size` for all users in the group.
    :ivar Tbar: Maximum deadline among all users in the group.
    :ivar x: Mean x-coordinate of all users in the group.
    :ivar y: Mean y-coordinate of all users in the group.
    """
    def __init__(self, gid, users):
        """
        Initializes an AggregateUser instance with the provided group ID and users.

        :param gid: The unique identifier for this aggregate user group.
        :type gid: int
        :param users: A list of User instances that belong to this aggregate user group.
        :type users: list[User]
        """
        self.id   = gid
        self.usr  = users
        self.Sbar = sum(u.weight * u.size for u in users)
        self.Tbar = max(u.deadline for u in users)
        self.x  = np.mean([u.x for u in users])
        self.y  = np.mean([u.y for u in users])
        for u in users:
            u.group_id = gid

    def print_user(self):
        print(f"Aggregate User Group Id: {self.id} x_coord: {self.x:.2f} y_coord:  {self.y:.2f} Sbar(kbits): {self.Sbar/1e3:.2f}  Tbar: {self.Tbar}"
              f"Total User: {len(self.usr)} Users {[u.id for u in self.usr]}" )



def create_aggregate_users(user_groups):
    """
    Creates a list of `AggregateUser` instances from the provided groups of users.

    This function iterates through the given list of user groups, creates an
    `AggregateUser` instance for each group, and adds the created instance
    to a list. The `AggregateUser` is initialized with the index of the group
    in the list and the corresponding group of users.

    :param user_groups: The list of user groups where each group is a collection
        of users to be aggregated.
    :type user_groups: list
    :return: A list of `AggregateUser` instances created from the given user
        groups.
    :rtype: list
    """
    aggregate_users = []
    for i, user_set in enumerate(user_groups):
        aggregate_users.append(AggregateUser(i, user_set))
    return aggregate_users


def calculate_rate(vau, t, f = utils.CARRIER_FREQUENCY):
    """
    Calculate the data transmission rate between a satellite and a user.

    This function computes the data rate (bits per second) based on the free-space
    path loss model and factors such as user location, satellite properties, and
    transmission settings.

    The computation assumes the satellite moves diagonally from the origin to the
    topmost corner of a defined area at a given time instance. It calculates the
    distance between the satellite and the user, applies the free-space path loss
    formula, and evaluates the achievable data rate using the Shannon-Hartley
    capacity theorem.

    :param vau: A user entity object. It contains the user's current position,
        specifically with attributes `x` and `y` for coordinates.
    :type vau: object
    :param t: The time at which the satellite's diagonal position is calculated,
        affecting its coordinates relative to the user.
    :type t: float
    :param f: The carrier frequency in Hertz used for transmission. Defaults to the
        value specified by `utils.CARRIER_FREQUENCY`.
    :type f: float, optional
    :return: The achievable data rate (bits per second) for the communication link
        between the satellite and the user at the given time.
    :rtype: float
    """
    satellite_x = satellite_y = (utils.USER_AREA / utils.TOTAL_SLOTS) * t
    # satellite is moving from [0,0] to [max_x,max_y] in a diagonal line.
    distance = np.sqrt((vau.x - satellite_x) ** 2 + (vau.y - satellite_y) ** 2 + utils.SATELLITE_ALTITUDE ** 2) * 1000
    g_t = 1000  # satellite transmit antenna gain in linear scale assuming 0.4 degree beamwidth
    g_r = 31.62  # user receiver antenna gain in linear scale
    p = utils.TRANSMIT_POWER  # satellite transmit power
    N_0 = 3.98 * 10 ** -18 # Noise spectral density -174 dbw/Hz
    c = 3 * 10 ** 8 # speed of light
    L = (c / (4 * np.pi * f * distance)) ** 2 # free space path loss
    x = (p * g_t * g_r * L) / (N_0 * utils.BANDWIDTH)
    rate = utils.BANDWIDTH * np.log(1+x)
    if utils.LOG_LEVEL >= 2:
        print(f"Satellite position at time {t}: ({satellite_x:.2f}, {satellite_y:.2f}), User position: ({vau.x:.2f}, {vau.y:.2f}), Distance: {distance:.2f} m, Rate: {rate * utils.RATE_SCALING_FACTOR /1e6:.2f} Mbps")
    return rate * utils.RATE_SCALING_FACTOR

def needs_assignment(au_m : AggregateUser, t, epsilon, epsilon_used, au_mapping):
    """
    Determine whether an AggregateUser needs assignment based on specific conditions.

    The function evaluates whether a given AggregateUser (`au_m`) requires assignment at
    time `t`. This determination is based on whether `au_m`'s usage of `epsilon` is below
    its allowable threshold, whether its `Tbar` value meets or exceeds the current time `t`,
    and whether the given time `t` is not already mapped in the user's assignment mappings.

    :param au_m: The user instance of AggregateUser to be evaluated.
    :type au_m: AggregateUser
    :param t: The current time for evaluation.
    :type t: int
    :param epsilon: A dictionary maintaining the epsilon threshold for each user indexed by their ID.
    :type epsilon: dict[int, float]
    :param epsilon_used: A dictionary mapping the used epsilon values for each user indexed by their ID.
    :type epsilon_used: dict[int, float]
    :param au_mapping: A dictionary storing time mappings for each user indexed by their ID.
    :type au_mapping: dict[int, set]
    :return: Whether the AggregateUser needs assignment at time `t` based on the criteria.
    :rtype: bool
    """
    return epsilon_used[au_m.id] < epsilon[au_m.id] and au_m.Tbar >= t and t not in au_mapping[au_m.id]


def initial_user_time_slot_assignment(aggregate_users):
    """
    Assigns initial time slots for aggregate users using proportional fairness and optimal assignment
    algorithms. This method ensures a fair allocation of beam-slot quotas to user groups based on
    their demands, followed by virtualizing users and performing cost minimization using the
    Kuhn-Munkres (Hungarian) algorithm.

    :param aggregate_users: List of aggregate user objects, each containing necessary attributes required
        for slot assignment such as 'Sbar' for their proportional demands.
    :type aggregate_users: List[AggregateUserType]

    :return: A tuple containing:
                - `au_time_slot_mapping`: A dictionary mapping aggregate user IDs to their assigned
                  time slots.
                - `time_slot_au_mapping`: A dictionary mapping time slots to aggregate user objects that
                  are assigned to those slots.
                - `epsilon`: An array representing the initial calculated beam-slot quotas per group.
                - `epsilon_used`: An array representing the final used beam-slot quotas after assignment.
    :rtype: Tuple[Dict[int, List[int]], Dict[int, List[AggregateUserType]], np.ndarray, np.ndarray]
    """
    # proportional fairness
    N0 = utils.TOTAL_BEAM_NUMBER * utils.TOTAL_SLOTS
    S = np.array([au.Sbar for au in aggregate_users])
    epsilon = np.round(N0 * S / S.sum()).astype(int)  # beam‑slot quota per group
    epsilon_used = np.zeros(len(epsilon))
    a_m = np.ceil(epsilon / utils.TOTAL_BEAM_NUMBER).astype(int)

    if utils.LOG_LEVEL >= 1:
        print("Initial time slot assignment started...")
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
    if utils.LOG_LEVEL >= 2:
        print("Cost matrix C:")
        print(C)
        print("Row indices (VAUs): ", row_idx)
        print("Column indices (Time Slots): ", col_idx)

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

    if utils.LOG_LEVEL >= 1:
        print("Initial assignment:")
    for AU_id in au_time_slot_mapping:
        epsilon_used[AU_id] =(int(len(au_time_slot_mapping[AU_id])))
        if utils.LOG_LEVEL >= 1:
            print(f"AU {AU_id} is assigned to slots {au_time_slot_mapping[AU_id]}")
    return au_time_slot_mapping, time_slot_au_mapping, epsilon, epsilon_used


def residual_time_slot_assignment(aggregate_users, au_time_slot_mapping, time_slot_au_mapping, epsilon, epsilon_used):
    """
    Assigns residual time slots to eligible Aggregate Users (AUs) according to specific conditions and constraints.
    """
    if utils.LOG_LEVEL >= 1:
        print("#" * 80)
        print("Residual time slot assignment started...")
        print("epsilon", epsilon)
        print("epsilon_used: ", epsilon_used)

    # Initialize mappings for any missing users
    for au in aggregate_users:
        if au.id not in au_time_slot_mapping:
            au_time_slot_mapping[au.id] = []
        # if au.id not in epsilon_used:
        #     epsilon_used[au.id] = 0
        # if au.id not in epsilon:
        #     epsilon[au.id] = 1.0  # Default epsilon value, adjust as needed

    for t in range(utils.TOTAL_SLOTS):
        already_assigned = time_slot_au_mapping[t]

        free_beams = utils.TOTAL_BEAM_NUMBER - 1
        candidates = []
        for au_m in aggregate_users:
            if au_m not in already_assigned and needs_assignment(au_m, t, epsilon, epsilon_used, au_time_slot_mapping):
                candidates.append(au_m)

        candidates.sort(key=lambda m: calculate_rate(m, t), reverse=True)

        for m in candidates[:free_beams]:
            epsilon_used[m.id] += 1
            time_slot_au_mapping[t].append(m)
            au_time_slot_mapping[m.id].append(t)
    if utils.LOG_LEVEL >= 1:
        print("#" * 80)
        print("Residual time slot assignment completed...")
        print("epsilon", epsilon)
        print("epsilon_used: ", epsilon_used)

    return au_time_slot_mapping, time_slot_au_mapping, epsilon, epsilon_used

def print_time_slot_assignments(time_slot_au_mapping,au_time_slot_mapping, epsilon, epsilon_used):
    """
    Prints the assignment of time slots to access units (AUs) along with epsilon values.

    This function takes a mapping of time slots to AUs and prints the assignments in
    sorted order of the time slots. Additionally, it displays the provided epsilon
    value and the amount of epsilon used.

    :param time_slot_au_mapping: Dictionary mapping time slots to sets or lists of
        AUs, where each AU has an `id` attribute.
    :type time_slot_au_mapping: dict
    :param epsilon: The total allowable epsilon value, generally a numerical type.
    :type epsilon: float
    :param epsilon_used: The portion of epsilon utilized, generally a numerical type.
    :type epsilon_used: float
    :return: None
    """
    if utils.LOG_LEVEL >= 1:
        print("#" * 80)
        print("Time slot to Aggregate User Assignments:")

        for time_slot, users in sorted(time_slot_au_mapping.items()):
            sorted_users = sorted(users, key=lambda x: x.id)
            print(f"Time slot {time_slot} is assigned to AUs {[au.id for au in sorted_users]}")

        print("#" * 80)
        print("Aggregate User to Time Slot Assignments:")
        for au_id, slots in sorted(au_time_slot_mapping.items()):
            sorted_slots = sorted(slots)
            print(f"AU {au_id} is assigned to time slots {sorted_slots}")
        print("#" * 80)
        print("epsilon:", epsilon)
        print("epsilon_used:", epsilon_used)
        print("#" * 80)
