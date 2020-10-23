from math import cos, pi, radians, sqrt, acos, degrees
from matplotlib import pyplot as plt
import numpy as np


def get_precision_points(lower_limit: float, upper_limit: float, no_pp: int):
    """"
    We will be finding the precision points using chebyshev spacing
    :param lower_limit: The lower limit for teh range
    :param upper_limit: The upper limit for the range
    :param no_pp; The number of precision precision points
    :return precision_points: The precision points determined
    """
    precision_points = []
    for i in range(1, no_pp + 1, 1):
        precision_points.append((0.5 * (lower_limit + upper_limit)) - (
                    0.5 * (upper_limit - lower_limit) * cos((pi * (2 * i - 1)) / (2 * no_pp))))
    return precision_points


def get_theta_4(theta2: list):
    """"
    Computes the output angles when given the input angles
    output_angle = 65 + (0.43 * input_angle)
    :param theta2: Input angles
    :return theta4: Output angles
    """
    theta4 = []
    for i in theta2:
        m = 65 + (0.43 * i)
        theta4.append(m)
    return theta4


def convert_angles_to_radians(angles: list):
    """"
    Converts a list of angles from degrees to radians
    :param angles: The list of angles
    :return new_angles: The converted angles
    """

    new_angles = []
    for i in angles:
        new_angles.append(radians(i))
    return new_angles


def compute_freudensteins_constants(input_angles: list, output_angles: list):
    """
    We will use the Freudenstein’s equation. It relate the input to output as a function
    of the size of the linkages. For a given input φ we can use this equation to solve for the
    output ψ
    Equation: k1(cos ψ) − k2(cos φ) + k3 = cos(φ − ψ)
    :param input_angles: The input angles
    :param output_angles: The output angles
    :return constants: The k1, k2, k3
    """
    # Changes the angles to radians as the function math.cos accepts only radians
    input_angles = convert_angles_to_radians(input_angles)
    output_angles = convert_angles_to_radians(output_angles)
    # Define coefficient matrices as numpy arrays
    A = np.array([
        [cos(output_angles[0]),
         -1 * cos(input_angles[0]),
         1
         ],
        [cos(output_angles[1]),
         -1 * cos(input_angles[1]),
         1
         ],
        [cos(output_angles[2]),
         -1 * cos(input_angles[2]),
         1]])
    # Define results matrices as numpy arrays
    B = np.array([
        cos(input_angles[0] - output_angles[0]),
        cos(input_angles[1] - output_angles[1]),
        cos(input_angles[2] - output_angles[2])])
    # Use numpy’s linear algebra solve function to solve the system
    constants = np.linalg.solve(A, B)
    return constants


def lengths_of_links(constants: list):
    """
    Compute the lengths of the links
    :param list: The constants
    k1 = d / a; a = d / k1
    k2 = d / c; c = d / k2
    k3 = (a*a - b*b + c*c + d*d) / 2ac; b = squareroot(a^2 + c^2 + d^2 - 2ack3)
    """
    d = 410
    a = d / constants[0]
    c = d / constants[1]
    b = sqrt((a * a) + (c * c) + (d * d) - (2 * a * c * constants[2]))
    return a, b, c, d


def get_transmission_angles(a, b, c, d, lower_limit, upper_limit, steps):
    """
    Computes the transmission angles based on the input_angles and length of links
    :param a: Crank, input link
    :param b: Coupler
    :param c: Rocker, output link
    :param d: Fixed link
    :param lower_limit: The input angle lower limit
    :param upper_limit: The input angle upper limit
    :param steps: The incremental steps
    :return transmission_angles
    """
    transmission_angles = []
    for i in range(lower_limit, upper_limit, steps):
        m = degrees(acos(((b * b + c * c) - (a * a + d * d) + (2 * a * d * cos(radians(i)))) / (2 * b * c)))
        transmission_angles.append(m)
    return transmission_angles


def plot_transmission_angles_and_input_angles(transmission_angles: list, input_angles: list):
    """
    Plots the transmission_angles vs input_angles angles
    :param transmission_angles: The transmission angles
    :param input_angles: The input angles
    """
    plt.xlabel("Input angles")
    plt.ylabel("Transmission angles")
    plt.title("Input angles vs Transmission angles")
    plt.plot(input_angles, transmission_angles, color='red', linewidth=1.0, )
    plt.show()


if __name__ == "__main__":
    theta2 = get_precision_points(15, 165, 3)
    theta4 = get_theta_4(theta2)
    constants = compute_freudensteins_constants(theta2, theta4)
    print("The constants are:\nk1: {}\nk2: {}\nk3: {}\n".format(constants[0], constants[1], constants[2]))
    a, b, c, d = lengths_of_links(constants)
    print(
        "The lenghts of links are:\nInput link: {}\nCoupler: {}\nOutput link: {}\nFixed link: {}\n".format(a, b, c, d))
    transmission_angles = get_transmission_angles(a, b, c, d, 15, 165, 5)
    print("The transmission angles for the range of inputs are:\n{}\n".format(transmission_angles))
    plot_transmission_angles_and_input_angles(transmission_angles, [i for i in range(15, 165, 5)])
