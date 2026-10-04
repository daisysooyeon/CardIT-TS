"""Fixed-loan-cost residual forecasts with Linear Regression.

The model predicts inflow and the non-contractual outflow residual.  The
contractual loan repayment schedule is added back to outflow after prediction.
"""

from residual_forecast_common import main


if __name__ == "__main__":
    main("linear_regression")
