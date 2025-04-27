import numpy as np

USER_NUMBER = 100
USER_AREA = 10*100 #km
TOTAL_SLOTS = 30
BEAM_RADIUS = 3.5 *100
USER_SWITCHING_THRESHOLD_HEAVY = 8
USER_SWITCHING_THRESHOLD_LIGHT= 2
TOTAL_BEAM_NUMBER = 4
RATE_PER_USER = 163 #kbps based on paper
SATELLITE_ALTITUDE = 2000 #km
CARRIER_FREQUENCY = 20 * 10 ** 9 # carrier frequency 20 Ghz
BANDWIDTH = 20 * 10 ** 6 #bandwidth 20 Mhz
USER_WEIGHT_THRESHOLD = 5
SUBCHANNEL_NUMBER = 10

class User:
    def __init__(self, uid, x, y,
                 size_mb: float,
                 weight: float,
                 deadline_slot: int):
        self.id   = uid
        self.x, self.y = float(x), float(y)
        self.size = size_mb          # s_n
        self.weight = weight         # w_n
        self.deadline = deadline_slot  # T_max_n

    def coords(self):
        return np.array([self.x, self.y])

    def print_user(self):
        print("User Id: " ,self.id ," x_coord: " ,self.x ," y_coord: " ,self.y, " data_size: ",self.size, " weight: ",self.weight," deadline: ",self.deadline, "")

# def create_users(user_number):
#     rng = np.random.default_rng()
#     user_coords = rng.random((user_number, 2)) * 10  # 30 random points in 2-D, coordinates in range (0,10]
#     user_list = []
#     for i in range(user_number):
#         user_list.append(User(i ,user_coords[i][0] ,user_coords[i][1]))
#     return user_list

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
    tau = 1.0  #slot length (s) – only matters for plotting
    size_range = (100, 1500) # kbits
    rng = None


    rng = np.random.default_rng() if rng is None else rng

    coords = rng.random((n_users, 2)) * area_side_km

    # sizes in kbit (use .uniform if you prefer log‑normal etc.)
    sizes = rng.uniform(*size_range, n_users)

    # weights: pick from {1,2,3 ... 10} or make them continuous
    weights = rng.integers(1, 10, n_users)

    # deadline slots: at least 5 slots after start and 2 before end
    deadlines = rng.integers(5, slot_count-1, n_users)

    users = [User(i,
                  coords[i,0], coords[i,1],
                 sizes[i],
                  weights[i],
                  int(deadlines[i]))
             for i in range(n_users)]
    return users
