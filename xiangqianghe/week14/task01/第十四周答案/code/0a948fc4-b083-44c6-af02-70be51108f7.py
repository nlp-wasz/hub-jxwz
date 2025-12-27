
import math  # 必须保留
import numpy as np  # 必须保留
from fastmcp import FastMCP  # 必须保留
from typing import Annotated, Union  # 必须保留
mcp = FastMCP(name="aquaculture_do_simulator")  # 必须保留

@mcp.tool
def simulate_dissolved_oxygen(
    a: Annotated[float, "Initial dissolved oxygen release amount, reflecting the initial oxygen content of the system"],
    b: Annotated[float, "Dissolved oxygen decay coefficient, characterizing the rate at which oxygen naturally decreases over time"],
    c: Annotated[float, "Amplitude of environmental disturbance, reflecting the intensity of periodic external factors (e.g., day-night cycles) on DO concentration"],
    d: Annotated[float, "Frequency of environmental disturbance, reflecting how fast the disturbance cycle occurs"],
    t: Annotated[float, "Time variable, representing elapsed time in hours or days"]
) -> Annotated[dict, "Returns the predicted dissolved oxygen concentration at time t"]:
    """
    Simulates the dissolved oxygen (DO) concentration in an aquaculture system over time.
    
    The model combines an exponential decay term and a periodic sinusoidal perturbation term 
    to capture both natural oxygen consumption and cyclic environmental influences such as 
    temperature, light, and wind changes throughout the day. This dynamic is critical for 
    maintaining optimal conditions for aquatic life and supports decision-making in water quality management.

    Parameters:
    - a (float): Initial oxygen level (mg/L or arbitrary unit), representing initial release or saturation.
    - b (float): Decay rate constant (>0), controls how quickly oxygen depletes due to respiration and oxidation.
    - c (float): Perturbation amplitude, scales the effect of periodic environmental forces (e.g., photosynthesis during daylight).
    - d (float): Perturbation frequency (rad/unit time), determines the cycle length (e.g., 2π/24 for daily cycle if t in hours).
    - t (float): Elapsed time (in hours or days).

    Returns:
    - dict: A dictionary with key 'DO(t)' containing the predicted DO concentration at time t.
            Example: {"DO(t)": 6.8}

    Model Equation:
        DO(t) = a * exp(-b * t) + c * sin(d * t)

    This equation models:
      - Exponential decline from initial oxygen levels due to biological and chemical consumption.
      - Superimposed oscillations simulating diurnal effects like algal photosynthesis (increasing DO in daylight) 
        and respiration (decreasing DO at night).
    """
    try:
        # Compute DO concentration using the given formula
        do_t = a * math.exp(-b * t) + c * math.sin(d * t)
        return {"DO(t)": do_t}
    except Exception as e:
        # In case of numerical errors (e.g., overflow, invalid values)
        return {"DO(t)": float("nan")}
