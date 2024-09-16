import pickle
import numpy as np
from scipy.optimize import curve_fit
import logging

logger = logging.getLogger(__name__)


class Calibration:
    """
    A class to handle calibration of EBT films, including fitting a calibration curve
    to NetOD and dose data, saving the calibration to disk, and loading it when needed.

    Attributes:
        ds (np.ndarray): List of doses (in Gy).
        nods (np.ndarray): List of corresponding NetOD values.
        lot (str): Lot number for the batch of EBT films.
        date (str): Calibration date.
        channel (str): Color channel used for calibration (default: 'RED').
        fitparams (tuple): Parameters of the fitted calibration curve (a, b, c).
        fitstr (str): A string representation of the fitted calibration equation.

    Example:
        Creating and using a Calibration object:

        >>> doses = [0, 1, 2, 3, 4, 5]
        >>> netODs = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        >>> calib = Calibration(doses, netODs, lot="12345678", channel="RED")
        >>> calib.save()
        >>> loaded_calib = Calibration.load("ebt_calibration_lot12345678_RED.pkl")
        >>> dose = loaded_calib.dose(0.5)
        >>> print(f"Dose for NetOD 0.5: {dose} Gy")
    """

    def __init__(self, ds, nods, lot="00000000", date="", channel='RED', guess=(20, 40, 3)):
        """
        Initializes the Calibration object by fitting a curve to the provided dose and NetOD data.

        Args:
            ds (list or np.ndarray): List of doses (in Gy).
            nods (list or np.ndarray): List of corresponding NetOD values.
            lot (str, optional): Lot number for the EBT film batch. Defaults to "00000000".
            date (str, optional): Calibration date. Defaults to an empty string.
            channel (str, optional): Color channel used for calibration. Defaults to 'RED'.
            guess (tuple, optional): Initial guess for the curve fitting parameters (a, b, c). Defaults to (20, 40, 3).
        """
        self.ds = np.asarray(ds)  # List of doses
        self.nods = np.asarray(nods)  # List of NetOD values
        self.lot = lot  # Lot number (string, may start with 0)
        self.date = date  # Calibration date
        self.channel = channel  # Color channel used

        # Fit the calibration curve to the data (NetOD vs. Dose)
        self.fitparams, _ = curve_fit(self.func, nods, ds, p0=guess)
        self.fitstr = f"Fit Dw = {self.fitparams[0]:.3f} * netOD + {self.fitparams[1]:.3f} * netOD^{self.fitparams[2]:.3f}"

    def save(self, filename=None):
        """
        Saves the calibration object to disk using pickle.

        Args:
            filename (str, optional): Filename to save the calibration object. Defaults to
                                      'ebt_calibration_lot{lot_number}_{channel}.pkl'.

        Note:
            The object is saved using pickle with the highest protocol available (`pickle.HIGHEST_PROTOCOL`).
        """
        if not filename:
            filename = f"ebt_calibration_lot{self.lot}_{self.channel}.pkl"
        with open(filename, 'wb') as f:
            logging.debug(f"Saving calibration to {filename}")
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def dose(self, netOD):
        """
        Calculates the dose (in Gy) for a given NetOD value using the fitted calibration curve.

        Args:
            netOD (float or np.ndarray): The NetOD value(s) to convert to dose. Can be a scalar or an array.

        Returns:
            float or np.ndarray: The corresponding dose in Gy.
        """
        netOD = np.asarray(netOD)
        return self.func(netOD, *self.fitparams)

    def plot(self, netODmin=0, netODmax=1.1, save=None):
        """
        Plots the calibration curve based on the fitted parameters.

        Args:
            netODmin (float, optional): The minimum NetOD value to plot. Defaults to 0.
            netODmax (float, optional): The maximum NetOD value to plot. Defaults to 2.0.
            save (str, optional): If provided, saves the plot to the specified file. Otherwise, displays the plot.
        """
        import matplotlib.pyplot as plt
        # Generate a range of NetOD values for the plot
        netOD_values = np.linspace(netODmin, netODmax, 100)
        dose_values = self.dose(netOD_values)

        # Create the plot
        plt.figure(figsize=(6, 4))
        plt.plot(self.nods, self.ds, 'ro',
                 label="Data points (NetOD vs. Dose)")
        plt.plot(netOD_values, dose_values, 'b-',
                 label=f"Calibration curve\n{self.fitstr}")
        plt.xlabel("Net Optical Density (NetOD)")
        plt.ylabel("Dose (Gy)")
        plt.title(f"Calibration Curve - Lot {self.lot}")
        plt.legend(loc="best")
        plt.grid(True)

        # Save or show the plot
        if save:
            plt.savefig(save)
            logger.debug(f"Calibration plot saved to {save}")
        else:
            plt.show()

    @staticmethod
    def load(filename):
        """
        Loads a saved calibration object from disk using the custom unpickler to handle
        renamed modules.

        Args:
            filename (str): The path to the calibration file.

        Returns:
            Calibration: The loaded calibration object.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            pickle.UnpicklingError: If loading the file fails.
        """
        logging.debug(f"Loading calibration from {filename}")
        try:
            with open(filename, 'rb') as f:
                return CustomUnpickler(f).load()
        except FileNotFoundError as e:
            logger.error(f"File not found: {filename}")
            raise e
        except (pickle.UnpicklingError, EOFError) as e:
            logger.error(f"Error loading file: {filename}. Reason: {e}")
            raise e

    @staticmethod
    def func(netOD, a=20, b=40, c=3):
        """
        The calibration function that maps NetOD to dose (Dw) using the form:
        Dw = a * netOD + b * netOD^c

        Args:
            netOD (float or np.ndarray): Net Optical Density. Can be a scalar or a 1D/2D array.
            a (float, optional): Fitted parameter. Defaults to 20.
            b (float, optional): Fitted parameter. Defaults to 40.
            c (float, optional): Fitted parameter. Defaults to 3.

        Returns:
            float or np.ndarray: Dose in Gy. If `netOD` is an array, a corresponding array of dose values is returned.
        """
        return a * netOD + b * np.power(netOD, c)

    def dose(self, netOD):
        """
        Calculates the dose (in Gy) for a given NetOD value using the fitted calibration curve.

        Args:
            netOD (float or np.ndarray): The NetOD value(s) to convert to dose. Can be a scalar or an array.

        Returns:
            float or np.ndarray: The corresponding dose in Gy.
        """
        netOD = np.asarray(netOD)
        return self.func(netOD, *self.fitparams)


# fix for old pickle files:import pickle

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Remap 'calibration' to 'radscan.calibration'
        if module == 'calibration':
            module = 'radscan.calibration'
        return super().find_class(module, name)


def load(filename):
    """
    Loads a saved calibration object from disk using the custom unpickler to handle
    renamed modules.

    Args:
        filename (str): The path to the calibration file.

    Returns:
        Calibration: The loaded calibration object.
    """
    logging.debug(f"Loading calibration from {filename}")
    with open(filename, 'rb') as f:
        return CustomUnpickler(f).load()
