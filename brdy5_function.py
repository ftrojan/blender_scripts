"""
Create a function which maps current elevation in meters to a fictive elevation 
with the following properties:

(1) For x <= 400m the elevation is unchanged. y = x.
(2) For x >= 500m slopes are 5x maginifed. y = 5*x + h
(3) For 400<=x<=500 there is a quadratic function which interpolates smoothly. y = a*x^2 + b*x + c.
"""
import logging
import numpy as np


def brdy_params() -> tuple:
    a = 0.02
    b = -15
    c = 3200
    h = a*np.power(500, 2) + (b - 5)*500 + c
    return a, b, c, h


def brdy5(x: float) -> float:
    """Solved on paper."""
    a, b, c, h = brdy_params()
    x2 = np.power(x, 2)
    if x <= 400:
        y = x
    elif x <= 500:
        y = a*x2 + b*x + c
    else:
        y = 5*x + h
    return y


def brdy5_grad(x: float) -> float:
    a, b, c, h = brdy_params()
    if x <= 400:
        y = 1
    elif x <= 500:
        y = 2*a*x + b
    else:
        y = 5
    return y


def main():
    logging.basicConfig(level="INFO")
    xx = [0, 400, 493, 500, 609, 640, 654, 690, 721, 865]
    for x in xx:
        y = brdy5(x)
        dy = brdy5_grad(x)
        logging.info(f"{x} -> {y:.0f} grad={dy:.3f}")
    logging.info("completed")


if __name__ == "__main__":
    main()