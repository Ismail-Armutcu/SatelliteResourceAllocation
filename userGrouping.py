from scipy.spatial import ConvexHull
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import numpy as np
import utils
import datetime


def get_timestamp_filename(prefix="user_groups_plot"):
    """Generate a filename with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.png"


def plot_users(user_list, user_groups, virtual_center_list, plotName="user_groups_plot"):
    """
    Plots all users, their convex hull, and highlights specific users groups within a circular region.

    :param user_list: List of all user objects. Each user has `x` and `y` coordinates and an `id`.
    :param user_groups: Subset of users (list) to be highlighted with the circular boundary.
    :param virtual_center_list: X-Y coordinates of the centers of the circular regions.
      """
    if utils.PLOT_LEVEL >= 1:
        # Extract coordinates of all users
        user_coordinates = np.array([[user.x, user.y] for user in user_list])

        # Begin plotting all users
        plt.figure(figsize=(10, 10))
        x_coords = [user.x for user in user_list]
        y_coords = [user.y for user in user_list]

        # Plot all users as points
        plt.scatter(x_coords, y_coords, c='blue', label='All Users')
        # Annotate points with user IDs
        # for user in user_list:
        #     plt.text(user.x, user.y+0.2, f"{user.id}", fontsize=15, ha='center')

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
                circle = plt.Circle((virtual_center_x, virtual_center_y), utils.BEAM_RADIUS, color='green', alpha=0.3,
                                    label="Beam Radius")
            else:
                plt.scatter(user_set_Im_coords[:, 0], user_set_Im_coords[:, 1], c='red')
                circle = plt.Circle((virtual_center_x, virtual_center_y), utils.BEAM_RADIUS, color='green', alpha=0.3)
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



        filename = get_timestamp_filename(plotName)
        plt.savefig(   "plots/"+filename, dpi=300, bbox_inches='tight')
        plt.show()

def find_minimum_circle_from_three_user(GUn1, GUn2, GUn3):
    """
    Finds the minimum circle that encompasses three given points.

    This function calculates the circumcenter and circumradius of the circle
    that passes through the three specified user locations. It uses deterministic
    geometry to compute the center and radius. If the provided points are
    collinear, a circle cannot be formed, and the function handles this
    degenerated case by displaying an appropriate message and returning None.

    :param GUn1: The first user represented as a point object with x and y
                 coordinates.
    :param GUn2: The second user represented as a point object with x and y
                 coordinates.
    :param GUn3: The third user represented as a point object with x and y
                 coordinates.
    :returns: A tuple containing the coordinates of the center of the circle
              (numpy array) and the radius of the circle (float). Returns None
              if the points are collinear.
    """
    if utils.LOG_LEVEL >= 2:
        if GUn1 == None or GUn2 == None or GUn3 == None:
            print("One of the users is None!!!!")
    A = np.array([GUn1.x, GUn1.y])
    B = np.array([GUn2.x, GUn2.y])
    C = np.array([GUn3.x, GUn3.y])

    # Calculate circum center
    D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
    
    if np.abs(D) < 1e-10:  # Handling degenerate case if the points are collinear
        if utils.LOG_LEVEL >= 2:
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
    """
    Computes the minimum enclosing circle around a group of users and updates group positions.
    This method orchestrates identifying the farthest pair among a subset of users, forms an
    initial circle, and iteratively adjusts the circle's bounds to accommodate ungrouped users.
    The algorithm updates positional groups as appropriate to refine the enclosing circle
    construct until a termination condition is reached or no further updates are possible.

    :param user_set_Im:
        List of users currently in group Im.
    :param user_set_Imc:
        List of users candidate for inclusion in Im.
    :param boundary_users:
        Users on the boundary of a previous group formation.
    :param internal_users:
        Users fully enclosed in previous group formations.
    :return:
        A tuple containing (radius, center_x, center_y) of the minimum enclosing circle.
    """
    # a. Select two users from I_m
    max_distance = 0
    farthest_users = (None, None)
    GUn1 = None
    GUn2 = None

    center_x = None
    center_y = None
    radius = None
    if len(user_set_Im) < 2:
        if utils.LOG_LEVEL >= 2:
            print("Not enough users in Im to form a circle.")
        return radius, center_x, center_y

    for i in range(len(user_set_Im)-1):
        for j in range(i + 1, len(user_set_Im)):
            user1, user2 = user_set_Im[i], user_set_Im[j]
            distance = np.sqrt((user1.x - user2.x) ** 2 + (user1.y - user2.y) ** 2)
            if distance > max_distance:
                max_distance = distance
                GUn1, GUn2 = (user1, user2)


    # b. Form an Initial Circle
    center_x = (GUn1.x + GUn2.x) / 2
    center_y = (GUn1.y + GUn2.y) / 2
    radius = max_distance / 2

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
            if utils.LOG_LEVEL >= 2:
                print(len(temp_user_set)," Users are added to Im !!!!!####***ALGO COMPLETED!!!!!####***")
        return radius, center_x, center_y
    else:
        # c. Select an Ungrouped GU from Imc
        for user in list(user_set_Imc):
            GUn3 = None
            distance_to_center = np.sqrt((user.x - center_x) ** 2 + (user.y - center_y) ** 2)
            if distance_to_center > utils.BEAM_RADIUS:
                GUn3 = user

            if GUn3 is not None:
                # Form a circle that covers GUn1, GUn2, and GUn3
                user_set_Imc.remove(GUn3)
                new_center,new_radius = find_minimum_circle_from_three_user(GUn1, GUn2, GUn3)
                radius = new_radius
                center_x = new_center[0]
                center_y = new_center[1]

                if new_radius <= utils.BEAM_RADIUS:
                    user_set_Im.append(GUn3)
                if len(user_set_Imc) == 0:
                    break

    return radius, center_x, center_y


def create_grouping(boundary_users, internal_users, distance_weights, distance_matrix_var):
    """
    Create user groupings based on boundary and internal users, as well as a distance matrix.
    This function calculates central positions for user groupings, then groups users based on
    their proximity to these central positions using specified distance criteria. Boundary users
    and internal users are processed iteratively to form and refine groups, ensuring optimal
    coverage of user clusters.

    :param boundary_users: A list of boundary user objects that need to be grouped.
    :param internal_users: A list of internal user objects that are considered for inclusion in groups.
    :param distance_weights: A dictionary where keys are user IDs and values represent their
                             associated distance weights.
    :param distance_matrix_var: A numpy matrix (or similar structure) representing distances
                                between all user pairs.
    :return: A tuple containing the list of grouped users (`user_set_Im`), the x-coordinate
             of their virtual center (`virtual_center_x`), and the y-coordinate of their
             virtual center (`virtual_center_y`).
    """
    # d. Select the Ungrouped Boundary User with the Highest Distance Degree to initiate user grouping
    user_set_Im = []  # Initialize list of user groups
    user_set_Imc = []
    boundary_user_ids = [user.id for user in boundary_users]
    boundary_weights = [(user_id, distance_weights[user_id]) for user_id in boundary_user_ids]
    max_boundary_user = max(boundary_weights, key=lambda x: x[1])
    max_user_id, max_distance = max_boundary_user

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

    # e. Determine the users in Im

    for GUn in list(boundary_users):
        if distance_matrix_var[GUn0.id, GUn.id] <= utils.BEAM_RADIUS:
            user_set_Im.append(GUn)
            boundary_users.remove(GUn)

    # f. Determine Imc (the users that fall into region (r,2r]
    for GUn in boundary_users:
        if utils.BEAM_RADIUS < distance_matrix_var[GUn0.id, GUn.id] <=2*utils.BEAM_RADIUS:
            user_set_Imc.append(GUn) ## append to the last element


    virtual_center_x = np.mean([user.x for user in user_set_Im])
    virtual_center_y = np.mean([user.y for user in user_set_Im])

    for user in list(internal_users):
        distance_to_center = np.sqrt((user.x - virtual_center_x) ** 2 + (user.y - virtual_center_y) ** 2)
        if distance_to_center <= utils.BEAM_RADIUS:
            user_set_Im.append(user)
            internal_users.remove(user)


    # g. Move Cm to Cover as Many Candidate Users as Possible
    Cm_radius,Cm_x,Cm_y= minimum_enclosing_circle_based_group_position_update(user_set_Im, user_set_Imc, boundary_users,internal_users)


    if Cm_radius != None:
        for user in boundary_users:
            if user in list(user_set_Imc):
                distance_to_center = np.sqrt((user.x - Cm_x) ** 2 + (user.y - Cm_y) ** 2)
                if distance_to_center <= Cm_radius:
                    user_set_Im.append(user)
                    user_set_Imc.remove(user)
                    boundary_users.remove(user)

    virtual_center_x = np.mean([user.x for user in user_set_Im])
    virtual_center_y = np.mean([user.y for user in user_set_Im])

    for user in list(internal_users):
        distance_to_center = np.sqrt((user.x - virtual_center_x) ** 2 + (user.y - virtual_center_y) ** 2)
        if distance_to_center <= utils.BEAM_RADIUS:
            user_set_Im.append(user)
            internal_users.remove(user)


    return user_set_Im,virtual_center_x,virtual_center_y


def group_ungrouped_internal_users(ungrouped_users,distance_weights,distance_matrix_var,user_groups,virtual_centers):
    """
    Groups ungrouped users into clusters based on a distance threshold and updates the
    user groups and their corresponding virtual centers. The grouping process iteratively
    identifies the most distant user in the current ungrouped users and forms a group with
    users that fall within a predefined radius from this user.

    :param ungrouped_users:
        List of users that are not yet grouped. Each user in the list is expected
        to have attributes `id`, `x`, and `y` indicating their unique identifier
        and spatial coordinates, respectively.
    :param distance_weights:
        Dictionary where the keys are user IDs and the values are their
        associated distance weights. This determines which user has the
        maximum distance among the ungrouped.
    :param distance_matrix_var:
        Two-dimensional matrix (or equivalent dictionary) that stores pairwise
        distances between user IDs. The matrix is used to evaluate how close
        any two users are to each other.
    :param user_groups:
        List of lists where each sublist contains grouped users. This parameter
        is modified in place to include newly formed groups during the function execution.
    :param virtual_centers:
        List of numpy arrays representing the calculated virtual centers for
        each user group. A virtual center is computed as the mean of the x and
        y coordinates of users in a specific group. This parameter is modified
        in place to include newly computed centers for the groups formed.

    :return:
        None. The input parameters `user_groups` and `virtual_centers` are modified
        in place to update the grouped users and their respective virtual centers.
    """
    while len(ungrouped_users) != 0:
        user_set_Im = []
        user_ids = [user.id for user in ungrouped_users]
        user_weights = [(user_id, distance_weights[user_id]) for user_id in user_ids]
        most_distant_user = min(user_weights, key=lambda x: x[1])
        max_user_id, max_distance = most_distant_user

        least_distant_user = None
        for user in ungrouped_users:
            if user.id == max_user_id:
                least_distant_user = user
                ungrouped_users.remove(user)
                user_set_Im.append(user)
                break

        for user in list(ungrouped_users):
            if distance_matrix_var[least_distant_user.id, user.id] < utils.BEAM_RADIUS:
                user_set_Im.append(user)
                ungrouped_users.remove(user)

        virtual_center_x = np.mean([user.x for user in user_set_Im])
        virtual_center_y = np.mean([user.y for user in user_set_Im])

        user_groups.append(user_set_Im)
        virtual_centers.append(np.array([virtual_center_x, virtual_center_y]))

def perform_user_switching(user_groups, virtual_centers):
    """
    Performs user switching among groups based on distance thresholds, beam radii, and group size
    constraints. This function modifies the given user groups by identifying users that can be
    moved to target groups. The process aims to balance user groups when certain thresholds are
    met and returns updated virtual center positions for all user groups.

    :param user_groups: A list where each element is a list of users representing a user group. Each user
        in the group has attributes `x` and `y` representing their coordinates.
    :type user_groups: list[list[User]]
    :param virtual_centers: A list of 2D coordinates representing the initial virtual centers for
        each user group. Each element is a numpy array with two values `[x, y]`.
    :type virtual_centers: list[numpy.ndarray]
    :return: A list of updated virtual centers, each being a numpy array `[x, y]`, representing the
        new average positions of users in each group after potential switching.
    :rtype: list[numpy.ndarray]
    """
    for i in range(len(user_groups) - 1):
        candidate_target_groups = []
        for j in range(i + 1, len(user_groups)):
            if (len(user_groups[i]) >= utils.USER_SWITCHING_THRESHOLD_HEAVY and
                    len(user_groups[j]) <= utils.USER_SWITCHING_THRESHOLD_LIGHT):
                for user in user_groups[i]:
                    distance = np.sqrt((user.x - virtual_centers[j][0]) ** 2 + (user.y - virtual_centers[j][1]) ** 2)
                    if distance <= utils.BEAM_RADIUS:
                        candidate_target_groups.append((j, user_groups[j]))
                        break


        if len(candidate_target_groups) == 0:
            if utils.LOG_LEVEL >= 2:
                print("No candidate target groups found.")
            continue
        elif len(candidate_target_groups) == 1:
            # select the only candidate
            candidate_switching_users = []
            [target_group_id, target_group] = candidate_target_groups.pop()
            for switching_user in user_groups[i]:
                distance = np.sqrt((switching_user.x - virtual_centers[target_group_id][0]) ** 2 + (
                            switching_user.y - virtual_centers[target_group_id][1]) ** 2)
                if distance <= utils.BEAM_RADIUS:
                    candidate_switching_users.append(switching_user)
            l_max = (len(user_groups[i]) - len(target_group)) // 2
            l_switch = max(1, min(l_max, len(candidate_switching_users)))
            candidate_switching_users = candidate_switching_users[:l_switch]
            for switching_user in candidate_switching_users:
                user_groups[i].remove(switching_user)
                target_group.append(switching_user)




        elif len(candidate_target_groups) > 1:
            # Find the candidate group ID with the smallest number of elements
            candidate_switching_users = []
            candidate_target_groups.sort(key=lambda c: len(c[1]))
            [target_group_id, target_group] = candidate_target_groups[0]

            for switching_user in user_groups[i]:
                distance = np.sqrt((switching_user.x - virtual_centers[target_group_id][0]) ** 2 + (
                        switching_user.y - virtual_centers[target_group_id][1]) ** 2)
                if distance <= utils.BEAM_RADIUS:
                    candidate_switching_users.append(switching_user)
            l_max = (len(user_groups[i]) - len(target_group)) // 2
            l_switch = max(1, min(l_max, len(candidate_switching_users)))


            candidate_switching_users = candidate_switching_users[:l_switch]
            for switching_user in candidate_switching_users:
                user_groups[i].remove(switching_user)
                target_group.append(switching_user)


    new_virtual_centers = []
    for user_group in user_groups:
        virtual_center_x = np.mean([user.x for user in user_group])
        virtual_center_y = np.mean([user.y for user in user_group])
        new_virtual_centers.append(np.array([virtual_center_x, virtual_center_y]))
    return  new_virtual_centers


def group_users():
    """
    Groups users based on their spatial coordinates into boundary users and internal users,
    and further clusters all users into groups using their calculated distance weights and
    connectivity matrix. The function refines grouped users by performing user-switching
    to optimize group configurations and results are visualized in different stages.

    Summary:
    1. Identifies boundary and internal users using a convex hull on spatial coordinates.
    2. Groups users iteratively, creating boundary-centered groups with virtual centers.
    3. Groups internal users separately if any remain ungrouped after step 2.
    4. Refines the user groups by allowing users to switch between groups for optimization.
    5. Visualizes the size of groups before and after the switching process.
    6. Plots intermediate and final group configurations along with their group centers.

    :param boundary_users: List of boundary users identified surrounding internal users.
    :param internal_users: List of users classified as internal within the boundary.
    :param user_list: List of all user objects defined by their coordinates and properties.
    :param user_coordinates: NumPy array of user spatial coordinates of shape (n, 2), where
        n is the number of users.

    :return: A tuple of two elements:
        - user_groups: List of lists, where each inner list represents a grouped set of users.
        - new_virtual_centers: List of NumPy arrays, where each array contains the coordinates
          (x, y) of the virtual group centers after switching.
    """
    # a. Initialization
    boundary_users = []
    internal_users = []
    user_list = utils.generate_users()  # The list of all users
    user_coordinates = np.array([[user.x, user.y] for user in user_list])
    # b. Determine Boundary and Internal Users
    hull = ConvexHull(user_coordinates)
    for user in user_list:
        if user.id in hull.vertices:
            boundary_users.append(user)
        else:
            internal_users.append(user)
    # c. Evaluate the Distance Degrees of GUs
    distance_matrix_var = distance_matrix(user_coordinates, user_coordinates)
    distance_weights = np.sum(distance_matrix_var, axis=1)
    user_groups = []
    virtual_centers = []
    while True:
        if len(boundary_users) == 0:
            break
        user_set_1, virtual_center_x, virtual_center_y = create_grouping(boundary_users, internal_users,
                                                                         distance_weights, distance_matrix_var)
        user_groups.append(user_set_1)
        virtual_centers.append(np.array([virtual_center_x, virtual_center_y]))

    plot_users(user_list, user_groups, virtual_centers,"Inıtial User Grouping Before Switching")
    if len(internal_users) != 0:
        if utils.LOG_LEVEL >= 2:
            print("!!!!*****####Internal Users Will be Grouped Separately!!!!*****####")
        ungrouped_users = internal_users
        internal_users = []
        group_ungrouped_internal_users(ungrouped_users, distance_weights, distance_matrix_var, user_groups,
                                       virtual_centers)
        plot_users(user_list, user_groups, virtual_centers,"Internal User Grouping Before Switching")

    # Create an array of the lengths of each user group
    user_group_lengths_before = [len(user_set) for user_set in user_groups]
    new_virtual_centers = perform_user_switching(user_groups, virtual_centers)
    plot_users(user_list, user_groups, new_virtual_centers,"User Grouping After Switching")
    user_group_lengths_after = [len(user_set) for user_set in user_groups]
    if utils.LOG_LEVEL >= 1:
        print(f"Lengths of user groups before switching: {user_group_lengths_before}")
        print(f"Lengths of user groups after switching: {user_group_lengths_after}")
    # Plot user group lengths before and after switching
    if utils.PLOT_LEVEL >=1:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(user_group_lengths_before) + 1), user_group_lengths_before, label="Before Switching",
                 marker='x')
        plt.plot(range(1, len(user_group_lengths_after) + 1), user_group_lengths_after, label="After Switching", marker='o')
        plt.xlabel("User Group Index")
        plt.ylabel("Number of Users")
        plt.title("User Group Sizes Before and After Switching")
        plt.legend()
        plt.grid(True)
        filename = get_timestamp_filename("group_sizes_plot")
        plt.savefig("plots/"+filename, dpi=300, bbox_inches='tight')
        plt.show()

    return user_groups, new_virtual_centers
