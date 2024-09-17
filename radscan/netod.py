import numpy as np
import logging

logger = logging.getLogger(__name__)


class NetOD:
    """
    A class to handle the calculation of Net Optical Density (NetOD) from pixel values.

    The class supports two modes:
    1. Converting mean pixel values from ROIs (scalars) to NetOD.
    2. Converting an entire post-irradiation image (2D array) to a NetOD map.

    Attributes:
        pvb (float): Mean pixel value before irradiation (scalar).
        pva (float or np.ndarray): Pixel value after irradiation (scalar or 2D array).
        pvcb (float, optional): Control pixel value before irradiation (scalar).
        pvca (float, optional): Control pixel value after irradiation (scalar).
        pvbk (float, optional): Background pixel value (scalar).
        spvb (float, optional): Standard error of pvb (scalar).
        spva (float or np.ndarray, optional): Standard error of pva (scalar or 2D array).
        spvcb (float, optional): Standard error of pvcb (scalar).
        spvca (float, optional): Standard error of pvca (scalar).
        spvbk (float, optional): Standard error of pvbk (scalar).
        simplified (bool): Whether to use the simplified NetOD calculation.

    Usage:
    - When using a Region of Interest (ROI), both `pvb` and `pva` should be scalars.
    - When converting an entire image, `pva` should be a 2D array while other values
      (e.g., `pvb`, `pvcb`, `pvca`) are scalars.
    """

    def __init__(self, pvb, pva, pvcb=None, pvca=None, pvbk=None,
                 spvb=0, spva=0, spvcb=0, spvca=0, spvbk=0, simplified=True):
        """
        Initializes the NetOD class.

        Args:
            pvb (float): Pixel value before irradiation (scalar).
            pva (float or np.ndarray): Pixel value after irradiation. Can be scalar (ROI) or 2D array (entire image).
            pvcb (float, optional): Control pixel value before irradiation (scalar). This represents a control region
                                    that was not irradiated.
            pvca (float, optional): Control pixel value after irradiation (scalar).
            pvbk (float, optional): Background pixel value (scalar).
            spvb (float, optional): Standard error of pvb (scalar).
            spva (float or np.ndarray, optional): Standard error of pva. Can be scalar (ROI) or 2D array.
            spvcb (float, optional): Standard error of pvcb (scalar).
            spvca (float, optional): Standard error of pvca (scalar).
            spvbk (float, optional): Standard error of pvbk (scalar).
            simplified (bool): Whether to use the simplified NetOD calculation. Defaults to True.
        """
        self.pvb = np.asarray(pvb)
        self.pva = np.asarray(pva)
        self.pvcb = np.asarray(pvcb) if pvcb is not None else None
        self.pvca = np.asarray(pvca) if pvca is not None else None
        self.pvbk = np.asarray(pvbk) if pvbk is not None else None

        self.spvb = spvb
        self.spva = np.asarray(spva)
        self.spvcb = spvcb
        self.spvca = spvca
        self.spvbk = spvbk

        self.simplified = simplified

    @staticmethod
    def simple(pvb, pva, spvb=0, spva=0):
        """
        Simple calculation of Net Optical Density (NetOD), typically used when no background correction is needed.

        Args:
            pvb (float): Pixel value before irradiation (scalar).
            pva (float or np.ndarray): Pixel value after irradiation. Can be scalar (ROI) or 2D array (full image).
            spvb (float, optional): Standard error of pvb (scalar). Defaults to 0.
            spva (float or np.ndarray, optional): Standard error of pva. Can be scalar or 2D array. Defaults to 0.

        Returns:
            tuple: NetOD and its standard error. If `pva` is a 2D array, NetOD is returned as a 2D array.
        """
        dn = np.log10(pvb / pva)
        sn = (1.0 / np.log(10.0)) * \
            np.sqrt((spvb / pvb) ** 2 + (spva / pva) ** 2)
        return dn, sn

    @staticmethod
    def calc(pvb, pva, pvcb, pvca, pvbk, spvb, spva, spvcb, spvca, spvbk):
        """
        Full calculation of NetOD with background and control correction.

        Args:
            pvb (float): Pixel value before irradiation (scalar).
            pva (float or np.ndarray): Pixel value after irradiation. Can be scalar (ROI) or 2D array (full image).
            pvcb (float): Control pixel value before irradiation (scalar). This represents a control region
                          that was not irradiated, used for additional correction.
            pvca (float): Control pixel value after irradiation (scalar).
            pvbk (float): Background pixel value (scalar).
            spvb (float): Standard error of pvb (scalar).
            spva (float or np.ndarray): Standard error of pva. Can be scalar or 2D array.
            spvcb (float): Standard error of pvcb (scalar).
            spvca (float): Standard error of pvca (scalar).
            spvbk (float): Standard error of pvbk (scalar).

        Returns:
            tuple: NetOD and its standard error. If `pva` is a 2D array, NetOD is returned as a 2D array.

        Background-subtracted pixel values are calculated for both the irradiated region (`pvb`, `pva`)
        and the control region (`pvcb`, `pvca`). Error propagation is performed to account for uncertainties
        in all pixel values.
        """
        # Background-subtracted pixel values
        pvb_bk = pvb - pvbk
        pva_bk = pva - pvbk
        pvcb_bk = pvcb - pvbk
        pvca_bk = pvca - pvbk

        # NetOD calculation
        dn = np.log10(pvb_bk / pva_bk) - np.log10(pvcb_bk / pvca_bk)

        # Error propagation
        l1 = (spvb / pvb_bk) ** 2 + (spva / pva_bk) ** 2
        l3 = (spvcb / pvcb_bk) ** 2 + (spvca / pvca_bk) ** 2
        l2 = ((pvb - pva) ** 2) / (pvb_bk * pva_bk) ** 2 * spvbk ** 2
        l4 = ((pvcb - pvca) ** 2) / (pvcb_bk * pvca_bk) ** 2 * spvbk ** 2

        sn = (1.0 / np.log(10.0)) * np.sqrt(l1 + l2 + l3 + l4)
        return dn, sn

    def dnetOD(self):
        """
        Calculate NetOD and its standard error.

        This method decides whether to use the simplified or full NetOD calculation based on the `simplified` attribute.
        If `simplified=True`, the `simple` method is used, otherwise the `calc` method is used.

        Returns:
            tuple: NetOD and its standard error. If `pva` is a 2D array, NetOD is returned as a 2D array.
        """
        if self.simplified:
            logger.debug("Using simplified NetOD calculation.")
            return self.simple(self.pvb, self.pva, self.spvb, self.spva)
        else:
            logger.debug(
                "Using full NetOD calculation with background and control correction.")
            return self.calc(self.pvb, self.pva, self.pvcb, self.pvca, self.pvbk,
                             self.spvb, self.spva, self.spvcb, self.spvca, self.spvbk)
