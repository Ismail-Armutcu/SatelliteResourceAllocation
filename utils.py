import numpy as np

import utils

USER_NUMBER = 50
USER_AREA = 10*100 #km
TOTAL_SLOTS = 30
BEAM_RADIUS = 3.5 *100 #km # radius of beam in km
USER_SWITCHING_THRESHOLD_HEAVY = 8
USER_SWITCHING_THRESHOLD_LIGHT= 2
TOTAL_BEAM_NUMBER = 4
SATELLITE_ALTITUDE = 2000 #km
CARRIER_FREQUENCY = 20 * 10 ** 9 # carrier frequency 20 Ghz
BANDWIDTH = 20 * 10 ** 6 #bandwidth 20 Mhz
USER_WEIGHT_THRESHOLD = 5
SUBCHANNEL_NUMBER = 10
PACKET_SIZE = (5*1e3, 10*1e3) #bit
TIME_SLOT_DURATION = 10 # ms
RATE_SCALING_FACTOR = 1
TRANSMIT_POWER = 40  # Watts
LAMBDA_1 = 0.5  # Weight for reward in utility calculation
RNG_SEED = 42
LOG_LEVEL = 0 # 0: no logs, 1: basic logs, 2: detailed logs
PLOT_LEVEL = 0  # 0: disable plots, 1: enable plots

class User:
    """
    Represents a user with coordinates, data size, weight, and a deadline slot.

    The User class is intended to encapsulate all necessary information about a
    user, such as their unique identifier, spatial coordinates, data size, weight,
    and deadline requirements. It provides basic functionalities such as
    retrieving user coordinates and printing user details.

    :ivar id: A unique identifier for the user.
    :type id: Any
    :ivar x: The x-coordinate of the user.
    :type x: float
    :ivar y: The y-coordinate of the user.
    :type y: float
    :ivar size: The data size associated with the user.
    :type size: float
    :ivar weight: The priority weight of the user.
    :type weight: float
    :ivar deadline: The deadline slot for the user.
    :type deadline: int
    """
    def __init__(self, uid, x, y,
                 size_mb: float,
                 weight: float,
                 deadline_slot: int):
        self.id   = uid
        self.x, self.y = float(x), float(y)
        self.size = size_mb          # s_n
        self.weight = weight         # w_n
        self.deadline = deadline_slot  # T_max_n
        self.group_id = -1  # For grouping users, if needed
        self.is_failed = False  # Flag to indicate if the user failed to meet their deadline

    def coords(self):
        return np.array([self.x, self.y])

    def print_user(self):
        print(f"User Id:  {self.id} Group Id {self.group_id}  x_coord:  {self.x:.2f} , y_coord:  {self.y:.2f}  data_size(kbits): {self.size/1e3:.2f},  weight: {self.weight} deadline: {self.deadline}")


def generate_users():
    """
    Create a list[User] with random coords, traffic sizes, weights & deadlines.
    Parameters
    ----------
    n_users : int
    area_side_km : float         # users uniform in [0, area_side_km]^2
    slot_count : int             # total T used in your sim
    size_range : tuple           # MB
    weight_vals : tuple/list     # discrete set or (low,high) if floats
    """
    n_users = USER_NUMBER
    area_side_km = USER_AREA
    slot_count = TOTAL_SLOTS
    rng = None

    rng = np.random.default_rng(utils.RNG_SEED) if rng is None else rng

    coords = rng.random((n_users, 2)) * area_side_km

    # weights: pick from {1,2,3 ... 10} or make them continuous
    weights = rng.integers(1, 10, n_users)

    # sizes in kbit (use .uniform if you prefer log‑normal etc.)
    sizes = rng.uniform(*PACKET_SIZE, n_users)
    # Generate deadlines ensuring they're feasible
    deadlines = rng.integers(15, TOTAL_SLOTS-1, n_users)

    for i in range(n_users):
        user_size = rng.uniform(*PACKET_SIZE)
        user_deadline = rng.integers(15, TOTAL_SLOTS - 1)

    users = [User(i,
                  coords[i, 0], coords[i, 1],
                  sizes[i],
                  weights[i],
                  int(deadlines[i]))
             for i in range(n_users)]
    return users
