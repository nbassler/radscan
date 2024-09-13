Calibration Class Documentation
===============================

This class handles the calibration for EBT films, including generating and saving the calibration curve.

.. automodule:: ebt.calibration
    :members:
    :undoc-members:
    :show-inheritance:

Example Usage:
--------------

Here's an example of how to use the Calibration class:

.. code-block:: python

    doses = [0, 1, 2, 3, 4, 5]
    netODs = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    calib = Calibration(doses, netODs, lot="12345678", channel="RED")
    calib.save()
    loaded_calib = Calibration.load("ebt_calibration_lot12345678_RED.pkl")
    dose = loaded_calib.dose(0.5)
    print(f"Dose for NetOD 0.5: {dose} Gy")
