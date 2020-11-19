from math import cos, pi, radians, sqrt, acos, degrees, atan, sin
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


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


def lengths_of_links(constants: list):
    """
    Compute the lengths of the links
    k1 = d / a; a = d / k1
    k2 = d / c; c = d / k2
    k3 = (a*a - b*b + c*c + d*d) / 2ac; b = squareroot(a^2 + c^2 + d^2 - 2ack3)
    :param list: The constants
    :return a, b, c, d: The link lengths
    """
    d = 180
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
    plt.plot(input_angles, transmission_angles, color='red', linewidth=1.0)
    plt.show()


def compute_least_square_method(theta2: list, theta4: list):
    """
    The angles θ and φ are specified for a position. If θ i and φ i are the angles for ith
    position, then Freudenstein’s equation may be written as
    k 1 cos φ i − k 2 cos θ i + k 3 − cos( θ i − φ i ) = 0
    Let e be the error which is defined as
    e = ∑ [ k 1 cos φ i − k 2 cos θ i + k 3 − cos ( θ i − φ i )] 2
    For e to be minimum, the partial derivatives of e with respect to k 1 , k 2 , k 3 separately must
    be equal to zero, i.e.
    :param theta2: Theta2
    :param theta4: Theta4
    :return constants: The constants
    """
    a, b, c, d, e, f, g, h = 0,0,0,0,0,0,0,0
    for i in range(0, len(theta2), 1):
        a = a + (cos(theta4[i]) * cos(theta4[i]))
        b = b + (cos(theta2[i]) * cos(theta4[i]))
        c = c + (cos(theta4[i]))
        d = d + (cos(theta2[i] - theta4[i]) * cos(theta4[i]))
        e = e + (cos(theta2[i]) * cos(theta2[i]))
        f = f + (cos(theta2[i]))
        g = g + (cos(theta2[i] - theta4[i]) * cos(theta2[i]))
        h = h + (cos(theta2[i] - theta4[i]))
        # Define coefficient matrices as numpy arrays
    A = np.array([
        [a, -1 * b, c],
        [b, -1 * e, f],
        [c, -1 * f, len(theta2)]])
    # Define results matrices as numpy arrays
    B = np.array([d, g , h])
    # Use numpy’s linear algebra solve function to solve the system
    constants = np.linalg.solve(A, B)
    return constants

def calculate_structural_errors(input_angles: list, output_angles: list, constants: list):
    """"
    Calculates the structural errors from the input, output and constants
    :param input_angles: The input angles
    :param output_angles: The output angles
    :param constants: The constants
    :return structural_errors: The structural errors
    """
    input_angles = convert_angles_to_radians(input_angles)
    output_angles = convert_angles_to_radians(output_angles)
    structural_errors = []
    for i in range(len(input_angles)):
        structural_errors.append((constants[0] * cos(output_angles[i])) - (constants[1] * cos(input_angles[i])) + constants[2] - cos(input_angles[i] - output_angles[i]))

    return structural_errors

def compute_output_angles(input_angles: list, constants: list):
    """
    Based on the given set of input and output angles applies
    simple linear regression for the whole range of input angles
    :param input_angles: The input angles
    :return output_angles: The output angles
    """
    # x = np.array([40, 45, 50, 55, 60]).reshape((-1, 1))
    # y = np.array([70, 76, 83, 91, 100])
    # model = LinearRegression()
    # model.fit(x,y) #actually produces the linear eqn for the data
    
    # # predicting the test set results
    # output_angles = model.predict(np.array(input_angles).reshape((-1,1)))
    output_angles = []
    for i in input_angles:
        A = sin(radians(i))
        B = cos(radians(i)) - constants[0]
        C = constants[2] - constants[1] * cos(radians(i))
        output_angles.append(round(degrees(2 * atan(((A - sqrt(A*A + B*B - C*C)) / (B + C)))), 1))
    print('INFO: We use the second set with negative as it corresponds with the input output values given in the question')
    return output_angles

def plot_structural_errors(errors, input_angles):
    """
    Plots the errors against input angles
    :param errors: The errors
    :param input_angles: The input angles
    """
    from scipy.interpolate import splrep, splev
    bspl = splrep(input_angles, errors, s=10)
    poly_y = splev(input_angles, bspl)
    plt.xlabel("Input angles")
    plt.ylabel("Structural errors")
    plt.title("Structural errors vs input angles")
    plt.plot(input_angles, poly_y, color='red', linewidth=1.0, label='smoothen')
    plt.plot(input_angles, errors, color='blue', linewidth=1.0, label='real errors')
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    # Question a
    theta2 = convert_angles_to_radians([40, 45, 50, 55, 60])
    theta4 = convert_angles_to_radians([70, 76, 83, 91, 100])
    constants = compute_least_square_method(theta2, theta4)
    print("The constants are:\nk1: {}\nk2: {}\nk3: {}\n".format(constants[0], constants[1], constants[2]))
    a, b, c, d = lengths_of_links(constants)
    print(
        "The lenghts of links based on Least Square Method are:\nInput link: {}\nCoupler: {}\nOutput link: {}\nFixed link: {}\n".format(a, b, c, d))

    # Question b
    input_angles = range(40, 60, 1)
    transmission_angles = get_transmission_angles(a, b, c, d, 40, 60, 1)
    plot_transmission_angles_and_input_angles(transmission_angles, input_angles)


    # Question c
    output_angles = compute_output_angles(input_angles, constants)
    errors = calculate_structural_errors(input_angles, output_angles, constants)
    plot_structural_errors(errors, input_angles)
