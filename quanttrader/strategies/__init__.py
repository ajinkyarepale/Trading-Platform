"""strategies – built-in example strategies."""
from .zscore_mean_reversion import ZScoreMeanReversion
from .dual_ma_crossover import DualMACrossover
from .rsi_mean_reversion import RSIMeanReversion
from .pairs_trading import PairsTrading
from .breakout import BreakoutStrategy

__all__ = [
    "ZScoreMeanReversion", "DualMACrossover", "RSIMeanReversion",
    "PairsTrading", "BreakoutStrategy",
]
