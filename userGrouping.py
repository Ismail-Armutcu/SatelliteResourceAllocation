from scipy.spatial import ConvexHull
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import numpy as np

USER_NUMBER = 100
BEAM_RADIUS = 3
USER_SWITCHING_THRESHOLD_HEAVY = 8
USER_SWITCHING_THRESHOLD_LIGHT= 2

class User:
    def __init__(self, user_id, x, y):
        self.id = user_id
        self.x = x
        self.y = y

    def coords(self):
        return np.array([self.x, self.y])

    def matches_coords(self, coord):
        """
        Returns True if the given coordinate matches this user's coordinates.
        """
        return np.allclose(coord, self.coords())
    def print_user(self):
        print("User Id: ",self.id," x_coord: ",self.x," y_coord: ",self.y)

def create_users(user_number):
    rng = np.random.default_rng()
    user_coords = rng.random((user_number, 2)) * 10  # 30 random points in 2-D, coordinates in range (0,10]
    user_list = []
    for i in range(user_number):
        user_list.append(User(i,user_coords[i][0],user_coords[i][1]))
    return user_list

def plot_users(user_list):
    x_coords = [user.x for user in user_list]
    y_coords = [user.y for user in user_list]
    # Create a scatter plot
    plt.scatter(x_coords, y_coords, color='blue', label='Users')
    # Add labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('User Coordinates')

    # Annotate points with user IDs
    for user in user_list:
        plt.text(user.x, user.y, f"{user.id}", fontsize=9, ha='right')

    user_coordinates = np.array([[user.x, user.y] for user in user_list])

    hull = ConvexHull(user_coordinates)
    boundary_users = []
    internal_users = []
    for user in user_list:
        if user.id in hull.vertices:
            boundary_users.append(user)
        else:
            internal_users.append(user)

    for simplex in hull.simplices:
        plt.plot(user_coordinates[simplex, 0], user_coordinates[simplex, 1], 'r-',
                 label='Convex Hull' if 'Convex Hull' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.legend(loc = 'upper right')
    plt.grid(True)
    plt.show()
    return plt


def plot_users2(user_list, user_set_Im, virtual_center_x, virtual_center_y):
    """
    Plots all users, their convex hull, and highlights specific users within a circular region.

    :param user_list: List of all user objects. Each user has `x` and `y` coordinates and an `id`.
    :param user_set_Im: Subset of users (list) to be highlighted with the circular boundary.
    :param virtual_center_x: X-coordinate of the center of the circular region.
    :param virtual_center_y: Y-coordinate of the center of the circular region.
    """

    # Extract coordinates of all users
    user_coordinates = np.array([[user.x, user.y] for user in user_list])

    # Begin plotting all users
    plt.figure(figsize=(10, 10))

    # Plot all users as points
    plt.scatter(user_coordinates[:, 0], user_coordinates[:, 1], c='blue', label='All Users')
    # Annotate points with user IDs
    for user in user_list:
        plt.text(user.x, user.y+0.2, f"{user.id}", fontsize=15, ha='center')

    # Compute and plot convex hull
    if len(user_coordinates) > 2:  # ConvexHull needs at least 3 points
        hull = ConvexHull(user_coordinates)
        for simplex in hull.simplices:
            plt.plot(user_coordinates[simplex, 0], user_coordinates[simplex, 1], 'k-')
        plt.plot(user_coordinates[hull.vertices, 0], user_coordinates[hull.vertices, 1], 'orange', linewidth=2,
                 label="Convex Hull")

    # Plot specific user_set_Im within circular boundary
    user_set_Im_coords = np.array([[user.x, user.y] for user in user_set_Im])
    plt.scatter(user_set_Im_coords[:, 0], user_set_Im_coords[:, 1], c='red', label="Users in Im")

    # Plot circular boundary
    circle = plt.Circle((virtual_center_x, virtual_center_y), BEAM_RADIUS, color='green', alpha=0.3,
                        label="Beam Radius")
    plt.text(virtual_center_x + 0.1, virtual_center_y + 0.1, "Center", fontsize=10, color='red')
    plt.gca().add_patch(circle)
    plt.scatter(virtual_center_x, virtual_center_y, c='green',marker = 'x')

    # Add labels and legend
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("User Positions and Convex Hull")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()

def plot_users3(user_list, user_groups, virtual_center_list):
    """
    Plots all users, their convex hull, and highlights specific users groups within a circular region.

    :param user_list: List of all user objects. Each user has `x` and `y` coordinates and an `id`.
    :param user_groups: Subset of users (list) to be highlighted with the circular boundary.
    :param virtual_center_list: X-Y coordinates of the centers of the circular regions.
      """

    # Extract coordinates of all users
    user_coordinates = np.array([[user.x, user.y] for user in user_list])

    # Begin plotting all users
    plt.figure(figsize=(10, 10))
    x_coords = [user.x for user in user_list]
    y_coords = [user.y for user in user_list]

    # Plot all users as points
    plt.scatter(x_coords, y_coords, c='blue', label='All Users')
    # Annotate points with user IDs
    for user in user_list:
        plt.text(user.x, user.y+0.2, f"{user.id}", fontsize=15, ha='center')

    # Compute and plot convex hull
    if len(user_coordinates) > 2:  # ConvexHull needs at least 3 points
        hull = ConvexHull(user_coordinates)
        for simplex in hull.simplices:
            plt.plot(user_coordinates[simplex, 0], user_coordinates[simplex, 1], 'k-')
        vertices = list(hull.vertices)
        vertices.append(vertices[0])
        plt.plot(user_coordinates[vertices, 0], user_coordinates[vertices, 1], 'orange', linewidth=2,
                 label="Convex Hull")
    for user_sets in user_groups:
        # Plot specific user_set_Im within circular boundary
        user_set_Im_coords = np.array([[user.x, user.y] for user in user_sets])

        virtual_center_index = user_groups.index(user_sets)
        virtual_center_x, virtual_center_y = virtual_center_list[virtual_center_index]
        # Plot circular boundary
        if virtual_center_index == 1:
            plt.scatter(user_set_Im_coords[:, 0], user_set_Im_coords[:, 1], c='red', label="User Group")
            circle = plt.Circle((virtual_center_x, virtual_center_y), BEAM_RADIUS, color='green', alpha=0.3,
                                label="Beam Radius")
        else:
            plt.scatter(user_set_Im_coords[:, 0], user_set_Im_coords[:, 1], c='red')
            circle = plt.Circle((virtual_center_x, virtual_center_y), BEAM_RADIUS, color='green', alpha=0.3)
        plt.text(virtual_center_x + 0.1, virtual_center_y + 0.1, "Center", fontsize=10, color='red')
        plt.gca().add_patch(circle)
        plt.scatter(virtual_center_x, virtual_center_y, c='green',marker = 'x')

    # Add labels and legend
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("User Positions and Convex Hull")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()

def find_mimimum_circle_from_three_user(GUn1,GUn2,GUn3):
    if GUn1 == None or GUn2 == None or GUn3 == None:
        print("One of the users is None!!!!")
    A = np.array([GUn1.x, GUn1.y])
    B = np.array([GUn2.x, GUn2.y])
    C = np.array([GUn3.x, GUn3.y])

    # Calculate circumcenter
    D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
    if np.abs(D) < 1e-10:  # Handling degenerate case if the points are collinear
        print("The points are collinear; cannot form a circle.")
        return

    ux = ((A[0] ** 2 + A[1] ** 2) * (B[1] - C[1]) +
          (B[0] ** 2 + B[1] ** 2) * (C[1] - A[1]) +
          (C[0] ** 2 + C[1] ** 2) * (A[1] - B[1])) / D
    uy = ((A[0] ** 2 + A[1] ** 2) * (C[0] - B[0]) +
          (B[0] ** 2 + B[1] ** 2) * (A[0] - C[0]) +
          (C[0] ** 2 + C[1] ** 2) * (B[0] - A[0])) / D

    # Calculate radius
    new_center = np.array([ux, uy])
    new_radius = np.sqrt((new_center[0] - A[0]) ** 2 + (new_center[1] - A[1]) ** 2)

    return new_center, new_radius

def minimum_enclosing_circle_based_group_position_update(user_set_Im,user_set_Imc,boundary_users,internal_users):

    # a. Select two users from Im
    max_distance = 0
    farthest_users = (None, None)
    GUn1 = None
    GUn2 = None

    center_x = None
    center_y = None
    radius = None
    if len(user_set_Im) < 2:
        print("Not enough users in Im to form a circle.")
        return radius, center_x, center_y

    for i in range(len(user_set_Im)-1):
        for j in range(i + 1, len(user_set_Im)):
            user1, user2 = user_set_Im[i], user_set_Im[j]
            distance = np.sqrt((user1.x - user2.x) ** 2 + (user1.y - user2.y) ** 2)
            if distance > max_distance:
                max_distance = distance
                GUn1, GUn2 = (user1, user2)

    print(f"Farthest users are User {GUn1.id} and User {GUn2.id} with distance {max_distance:.2f}.")

    # b. Form an Initial Circle
    center_x = (GUn1.x + GUn2.x) / 2
    center_y = (GUn1.y + GUn2.y) / 2
    radius = max_distance / 2
    print(f"Initial circle center: ({center_x:.2f}, {center_y:.2f}) with radius: {radius:.2f}")

    ungrouped_users = boundary_users + internal_users + user_set_Imc
    temp_user_set = []

    for user in list(ungrouped_users):  # Create a copy of the list to avoid issues during iteration
        distance_to_center = np.sqrt((user.x - center_x) ** 2 + (user.y - center_y) ** 2)
        if distance_to_center <= radius:
            temp_user_set.append(user)  # Add user to the group
    if len(temp_user_set) != 0:
        for user in temp_user_set:
            if user in boundary_users:
                boundary_users.remove(user)
            if user in internal_users:
                internal_users.remove(user)
            if user in user_set_Imc:
                user_set_Imc.remove(user)
            user_set_Im.append(user)
            print("User with Id: ",user.id," is appended to Im")
            print(len(temp_user_set)," Users are added to Im !!!!!####***ALGO COMPLETED!!!!!####***")
        return radius, center_x, center_y
    else:
        # c. Select an Ungrouped GU from Imc
        for user in list(user_set_Imc):
            GUn3 = None
            distance_to_center = np.sqrt((user.x - center_x) ** 2 + (user.y - center_y) ** 2)
            if distance_to_center > BEAM_RADIUS:
                GUn3 = user

            if GUn3 is not None:
                # Form a circle that covers GUn1, GUn2, and GUn3
                user_set_Imc.remove(GUn3)
                new_center,new_radius = find_mimimum_circle_from_three_user(GUn1,GUn2, GUn3)
                GUn1.print_user()
                GUn2.print_user()
                GUn3.print_user()
                print(f"New minimum circle center: ({new_center[0]:.2f}, {new_center[1]:.2f}) with radius: {new_radius:.2f}")
                radius = new_radius
                center_x = new_center[0]
                center_y = new_center[1]

                if new_radius <= BEAM_RADIUS:
                    user_set_Im.append(GUn3)
                    print("User with Id: ", GUn3.id, "is appended to Im")
                if len(user_set_Imc) == 0:
                    print("!!!!!####***ALGO COMPLETED!!!!!####***")
                    break

    return radius, center_x, center_y


def create_grouping(boundary_users, internal_users, distance_weights, distance_matrix_var):
    # d. Select the Ungrouped Boundary User with the Highest Distance Degree to initiate user grouping
    user_set_Im = []  # Initialize list of user groups
    user_set_Imc = []
    boundary_user_ids = [user.id for user in boundary_users]
    boundary_weights = [(user_id, distance_weights[user_id]) for user_id in boundary_user_ids]
    max_boundary_user = max(boundary_weights, key=lambda x: x[1])
    max_user_id, max_distance = max_boundary_user
    print(f"Boundary User with ID {max_user_id} has the highest distance degree: {max_distance}")

    # Remove user with ID max_user_id from boundary_users

    GUn0 = None
    for user in boundary_users:
        if user.id == max_user_id:
            GUn0 = user
            break

    # Remove the user from boundary_users
    boundary_users.remove(GUn0)
    # Add the user to user_set_Im
    user_set_Im.append(GUn0)
    print("User with Id: ",GUn0.id," is added to Im")


    # e. Determine the users in Im

    for GUn in list(boundary_users):
        if distance_matrix_var[GUn0.id, GUn.id] <= BEAM_RADIUS:
            user_set_Im.append(GUn)
            print("User With Id: ",GUn.id," is appended to Im")
            boundary_users.remove(GUn)

    # f. Determine Imc (the users that fall into region (r,2r]
    for GUn in boundary_users:
        if BEAM_RADIUS < distance_matrix_var[GUn0.id, GUn.id] <=2*BEAM_RADIUS:
            user_set_Imc.append(GUn) ## append to the last element

    print("Users in Im Initally")
    [user.print_user() for user in user_set_Im]
    print("Users in Imc Initally")
    [user.print_user() for user in user_set_Imc]
    print("Users in boundary_users Initally")
    [user.print_user() for user in boundary_users]
    print("Users in internal_users Initally")
    [user.print_user() for user in internal_users]

    virtual_center_x = np.mean([user.x for user in user_set_Im])
    virtual_center_y = np.mean([user.y for user in user_set_Im])

    for user in list(internal_users):
        distance_to_center = np.sqrt((user.x - virtual_center_x) ** 2 + (user.y - virtual_center_y) ** 2)
        if distance_to_center <= BEAM_RADIUS:
            user_set_Im.append(user)
            print("User with Id: ", user.id, " is appended to Im")
            internal_users.remove(user)


    # g. Move Cm to Cover as Many Candidate Users as Possible
    Cm_radius,Cm_x,Cm_y= minimum_enclosing_circle_based_group_position_update(user_set_Im, user_set_Imc, boundary_users,internal_users)


    if Cm_radius != None:
        for user in boundary_users:
            if user in list(user_set_Imc):
                distance_to_center = np.sqrt((user.x - Cm_x) ** 2 + (user.y - Cm_y) ** 2)
                if distance_to_center <= Cm_radius:
                    user_set_Im.append(user)
                    print("User with Id: ",user.id," is appended to Im")
                    user_set_Imc.remove(user)
                    boundary_users.remove(user)

    virtual_center_x = np.mean([user.x for user in user_set_Im])
    virtual_center_y = np.mean([user.y for user in user_set_Im])

    for user in list(internal_users):
        distance_to_center = np.sqrt((user.x - virtual_center_x) ** 2 + (user.y - virtual_center_y) ** 2)
        if distance_to_center <= BEAM_RADIUS:
            user_set_Im.append(user)
            print("User with Id: ", user.id, " is appended to Im")
            internal_users.remove(user)



    print("Users in Im Finally")
    [user.print_user() for user in user_set_Im]
    print("Users in Imc Finally")
    [user.print_user() for user in user_set_Imc]
    print("Users in boundary_users Finally")
    [user.print_user() for user in boundary_users]
    print("Users in internal_users Finally")
    [user.print_user() for user in internal_users]

    return user_set_Im,virtual_center_x,virtual_center_y


def group_ungrouped_internal_users(ungrouped_users,distance_weights,distance_matrix_var,user_groups,virtual_centers):
    while len(ungrouped_users) != 0:
        user_set_Im = []
        user_ids = [user.id for user in ungrouped_users]
        user_weights = [(user_id, distance_weights[user_id]) for user_id in user_ids]
        most_distant_user = min(user_weights, key=lambda x: x[1])
        max_user_id, max_distance = most_distant_user
        print(f"Boundary User with ID {max_user_id} has the highest distance degree: {max_distance}")

        least_distant_user = None
        for user in ungrouped_users:
            if user.id == max_user_id:
                least_distant_user = user
                ungrouped_users.remove(user)
                user_set_Im.append(user)
                break

        for user in list(ungrouped_users):
            if distance_matrix_var[least_distant_user.id, user.id] < BEAM_RADIUS:
                user_set_Im.append(user)
                ungrouped_users.remove(user)

        virtual_center_x = np.mean([user.x for user in user_set_Im])
        virtual_center_y = np.mean([user.y for user in user_set_Im])

        user_groups.append(user_set_Im)
        virtual_centers.append(np.array([virtual_center_x, virtual_center_y]))


# a. Initialization


boundary_users = []
internal_users = []

user_list = create_users(USER_NUMBER) # The list of all users
user_coordinates = np.array([[user.x, user.y] for user in user_list])

# b. Determine Boundary and Internal Users
hull = ConvexHull(user_coordinates)
for user in user_list:
    if user.id in hull.vertices:
        boundary_users.append(user)
    else:
        internal_users.append(user)

# c. Evaluate the Distance Degrees of GUs
distance_matrix_var = distance_matrix(user_coordinates,user_coordinates)
distance_weights = np.sum(distance_matrix_var, axis=1)

user_groups = []
virtual_centers = []
while True:
    if len(boundary_users) == 0:
        break
    user_set_1,virtual_center_x,virtual_center_y = create_grouping(boundary_users, internal_users, distance_weights, distance_matrix_var)
    user_groups.append(user_set_1)
    virtual_centers.append(np.array([virtual_center_x,virtual_center_y]))
    #plot_users2(user_list,user_set_1,virtual_center_x,virtual_center_y)

plot_users3(user_list,user_groups,virtual_centers)

if len(internal_users) != 0:
    print("!!!!*****####Internal Users Will be Grouped Seperately!!!!*****####")
    ungrouped_users = internal_users
    internal_users = []
    group_ungrouped_internal_users(ungrouped_users,distance_weights,distance_matrix_var,user_groups,virtual_centers)
    plot_users3(user_list, user_groups, virtual_centers)

print("User Groups")
for i, user_set in enumerate(user_groups):
    print(f"User Group {i + 1} With {len(user_set)} Users:")
    [user.print_user() for user in user_set]

# Create an array of the lengths of each user group
user_group_lengths_before = [len(user_set) for user_set in user_groups]


for i in range(len(user_groups) - 1):
    candidate_target_groups = []
    for j in range(i + 1, len(user_groups)):
        if (len(user_groups[i]) >= USER_SWITCHING_THRESHOLD_HEAVY and
                len(user_groups[j]) <= USER_SWITCHING_THRESHOLD_LIGHT):
            for user in user_groups[i]:
                distance = np.sqrt((user.x - virtual_centers[j][0]) ** 2 + (user.y - virtual_centers[j][1]) ** 2)
                if distance <= BEAM_RADIUS:
                    candidate_target_groups.append((j, user_groups[j]))
                    break

    for candidate in candidate_target_groups:
        print(candidate[0])
        for candidate_user in candidate[1]:
            print("\t", candidate_user.id)
                    
    if len(candidate_target_groups) == 0:
        print("No candidate target groups found.")
        continue
    elif len(candidate_target_groups) == 1:
        # select the only candidate
        candidate_switching_users = []
        [target_group_id,target_group] = candidate_target_groups.pop()
        for switching_user in user_groups[i]:
            distance = np.sqrt((switching_user.x - virtual_centers[target_group_id][0]) ** 2 + (switching_user.y - virtual_centers[target_group_id][1]) ** 2)
            if distance <= BEAM_RADIUS:
                candidate_switching_users.append(switching_user)
        l_max = (len(user_groups[i]) - len(target_group)) // 2
        l_switch = max(1, min(l_max, len(candidate_switching_users)))
        print(
            f"Only one candidate target group found. l_max:{l_max}, l_switch:{l_switch} "
            f"size_user_group:{len(user_groups[i])} "
            f"size_target_group:{len(target_group)} size_Switching_users:{len(candidate_switching_users)}")
        candidate_switching_users = candidate_switching_users[:l_switch]
        for switching_user in candidate_switching_users:
            user_groups[i].remove(switching_user)
            target_group.append(switching_user)
            print(f"Switched User {switching_user.id} from User Group {i + 1} to User Group {target_group_id + 1}")



    elif len(candidate_target_groups) > 1:
        # Find the candidate group ID with the smallest number of elements
        candidate_switching_users = []
        candidate_target_groups.sort(key=lambda c: len(c[1]))
        [target_group_id,target_group] = candidate_target_groups[0]
        print(f"Selected Target Group ID: {target_group_id}")
        for switching_user in user_groups[i]:
            distance = np.sqrt((switching_user.x - virtual_centers[target_group_id][0]) ** 2 + (
                        switching_user.y - virtual_centers[target_group_id][1]) ** 2)
            if distance <= BEAM_RADIUS:
                candidate_switching_users.append(switching_user)
        l_max = (len(user_groups[i]) - len(target_group)) // 2
        l_switch = max(1, min(l_max, len(candidate_switching_users)))
        print(
            f"Multiple target group found. l_max:{l_max}, l_switch:{l_switch} size_user_group:{len(user_groups[i])} "
            f"size_target_group:{len(target_group)} size_Switching_users:{len(candidate_switching_users)}")

        candidate_switching_users = candidate_switching_users[:l_switch]
        for switching_user in candidate_switching_users:
            user_groups[i].remove(switching_user)
            target_group.append(switching_user)
            print(f"Switched User {switching_user.id} from User Group {i + 1} to User Group {target_group_id + 1}")
        


new_virtual_centers = []
for user_group in user_groups:
    virtual_center_x = np.mean([user.x for user in user_group])
    virtual_center_y = np.mean([user.y for user in user_group])
    new_virtual_centers.append(np.array([virtual_center_x, virtual_center_y]))

plot_users3(user_list,user_groups,new_virtual_centers)
user_group_lengths_after = [len(user_set) for user_set in user_groups]
print(f"Lengths of user groups before switching: {user_group_lengths_before}")
print(f"Lengths of user groups after switching: {user_group_lengths_after}")


# Plot user group lengths before and after switching
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(user_group_lengths_before) + 1), user_group_lengths_before, label="Before Switching", marker='x')
plt.plot(range(1, len(user_group_lengths_after) + 1), user_group_lengths_after, label="After Switching", marker='o')
plt.xlabel("User Group Index")
plt.ylabel("Number of Users")
plt.title("User Group Sizes Before and After Switching")
plt.legend()
plt.grid(True)
plt.show()











