from enum import Enum

from model.costs.mae_cost import MAE
from model.costs.mse_cost import MSE


class Costs(Enum):
    mae = MAE()
    mse = MSE()
