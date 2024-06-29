import datetime as dt
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict



# Type of an option
OptionType = Literal["call", "put"]  # call or put option

# Action to be taken on an option
Action = Literal["buy", "sell"]  # buy or sell option

# Type of a strategy
StrategyType = Literal["stock"] | OptionType | Literal["closed"]  # stock, option or closed position

# Range of values
Range = tuple[float, float]  # tuple of two floats representing a range

# Type of distribution
Distribution = Literal["black-scholes", "normal", "laplace", "array"]  # distribution type

# Country for holidays
Country = Literal[
    "US",  # United States
    "Canada",  # Canada
    "Mexico",  # Mexico
    "Brazil",  # Brazil
    "China",  # China
    "India",  # India
    "South Korea",  # South Korea
    "Russia",  # Russia
    "Japan",  # Japan
    "UK",  # United Kingdom
    "France",  # France
    "Germany",  # Germany
    "Italy",  # Italy
    "Australia",  # Australia
]  # country for holidays


class BaseStrategy(BaseModel):
    """
    Base class for strategies.

    Attributes:
        action (Action): The action to be taken on the strategy, either 'buy' or 'sell'.
        prev_pos (float | None): The total value of the position to be closed, which can be positive if it made a profit or negative if it is a loss.
            If not defined, the position remains open.
    """

    action: Action  # Action to be taken on the strategy
    prev_pos: float | None = None  # The total value of the position to be closed


class StockStrategy(BaseStrategy):
    """
    "type" : string
        It must be 'stock'. It is mandatory.
    "n" : int
        Number of shares. It is mandatory.
    "action" : string
        Either 'buy' or 'sell'. It is mandatory.
    "prev_pos" : float
        Stock price effectively paid or received in a previously
        opened position. If positive, it means that the position
        remains open and the payoff calculation takes this price
        into account, not the current price of the stock. If
        negative, it means that the position is closed and the
        difference between this price and the current price is
        considered in the payoff calculation.

    """

    type: Literal["stock"] = "stock"
    n: int = Field(gt=0)
    premium: float | None = None


class OptionStrategy(BaseStrategy):
    """
    "type" : string
        Either 'call' or 'put'. It is mandatory.
    "strike" : float
        Option strike price. It is mandatory.
    "premium" : float
        Option premium. It is mandatory.
    "n" : int
        Number of options. It is mandatory
    "action" : string
        Either 'buy' or 'sell'. It is mandatory.
    "prev_pos" : float
        Premium effectively paid or received in a previously opened
        position. If positive, it means that the position remains
        open and the payoff calculation takes this price into
        account, not the current price of the option. If negative,
        it means that the position is closed and the difference
        between this price and the current price is considered in
        the payoff calculation.
    "expiration" : string | int, optional.
        Expiration date or days to maturity. If not defined, will use `target_date` or `days_to_target_date`.
    """

    type: OptionType
    strike: float = Field(gt=0)
    premium: float = Field(gt=0)
    n: int = Field(gt=0)
    expiration: dt.date | int | None = None

    @field_validator("expiration")
    def validate_expiration(cls, v: dt.date | int | None) -> dt.date | int | None:
        if isinstance(v, int) and v <= 0:
            raise ValueError("If expiration is an integer, it must be greater than 0.")
        return v


class ClosedPosition(BaseModel):
    """
    Represents a previously opened position that is being closed.

    Attributes:
        type (Literal["closed"]): It must be 'closed'. It is mandatory.
        prev_pos (float): The total value of the position to be closed,
            which can be positive if it made a profit or negative if it
            is a loss. It is mandatory.
    """

    # It must be 'closed'. It is mandatory.
    type: Literal["closed"] = "closed"

    # The total value of the position to be closed,
    # which can be positive if it made a profit or negative if it
    # is a loss. It is mandatory.
    prev_pos: float


Strategy = StockStrategy | OptionStrategy | ClosedPosition


class ProbabilityOfProfitInputs(BaseModel):
    """
    Represents the inputs for the PoP calculation.

    Attributes:
        source (Literal["black-scholes", "normal", "laplace"]): The statistical distribution used to compute the PoP.
        stock_price (float): Spot price of the underlying.
        volatility (float): Annualized volatility.
        years_to_maturity (float): Time left to maturity in units of year.
        interest_rate (float | None): Annualized risk-free interest rate. Required for 'black-schols' PoP calculation.
        dividend_yield (float): Annualized dividend yield.
    """

    # The statistical distribution used to compute the PoP.
    # Possible values are 'black-schols', 'normal' or 'laplace'.
    source: Literal["black-scholes", "normal", "laplace"]

    # Spot price of the underlying.
    stock_price: float = Field(gt=0)

    # Annualized volatility.
    volatility: float = Field(gt=0, le=1)

    # Time left to maturity in units of year.
    years_to_maturity: float = Field(gt=0)

    # Annualized risk-free interest rate. Required for 'black-schols' PoP calculation.
    interest_rate: float | None = Field(None, gt=0, le=0.2)

    # Annualized dividend yield.
    dividend_yield: float = Field(0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_black_scholes_model(self) -> "ProbabilityOfProfitInputs":
        """
        Validates the PoP inputs for the 'black-schols' model.

        Raises:
            ValueError: If interest rate is not provided for 'black-schols' PoP calculations.

        Returns:
            ProbabilityOfProfitInputs: The validated inputs.
        """
        # If the source is 'black-schols' and the interest rate is not provided, raise an error.
        if self.source == "black-schols" and not self.interest_rate:
            raise ValueError(
                "Risk-free interest rate must be provided for 'black-schols' PoP calculations!"
            )
        return self


class ProbabilityOfProfitArrayInputs(BaseModel):
    """
    array : np.ndarray
        the probability of profit is calculated from a 1D numpy array of stock prices
        typically at maturity generated by a Monte Carlo simulation (or another user-defined
        data generation process)
    """

    source: Literal["array"] = "array"
    array: np.ndarray
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("array", mode="before")
    @classmethod
    def validate_arrays(cls, v: np.ndarray | list[float]) -> np.ndarray:
        arr = np.asarray(v)
        if arr.shape[0] == 0:
            raise ValueError("The array of stock prices is empty!")
        return arr


class Inputs(BaseModel):
    """
    stock_price : float
        Spot price of the underlying.
    volatility : float
        Annualized volatility.
    interest_rate : float
        Annualized risk-free interest rate.
    min_stock : float
        Minimum value of the stock in the stock price domain.
    max_stock : float
        Maximum value of the stock in the stock price domain.
    strategy : list
        A list of `Strategy`
    dividend_yield : float, optional
        Annualized dividend yield. Default is 0.0.
    profit_target : float, optional
        Target profit level. Default is None, which means it is not
        calculated.
    loss_limit : float, optional
        Limit loss level. Default is None, which means it is not calculated.
    opt_commission : float
        Broker commission for options transactions. Default is 0.0.
    stock_commission : float
        Broker commission for stocks transactions. Default is 0.0.
    compute_expectation : logical, optional
        Whether or not the strategy's average profit and loss must be
        computed from a numpy array of random terminal prices generated from
        the chosen distribution. Default is False.
    discard_nonbusinessdays : logical, optional
        Whether to discard Saturdays and Sundays (and maybe holidays) when
        counting the number of days between two dates. Default is True.
    country : string, optional
        Country for which the holidays will be considered if 'discard_nonbusinessdyas'
        is True. Default is 'US'.
    start_date : dt.date, optional
        Start date in the calculations. If not provided, days_to_target_date must be provided.
    target_date : dt.date, optional
        Start date in the calculations. If not provided, days_to_target_date must be provided.
    days_to_target_date : int, optional
        Days to maturity. If not provided, start_date and end_date must be provided.
    distribution : string, optional
        Statistical distribution used to compute probabilities. It can be
        'black-scholes', 'normal', 'laplace' or 'array'. Default is 'black-scholes'.
    mc_prices_number : int, optional
        Number of random terminal prices to be generated when calculationg
        the average profit and loss of a strategy. Default is 100,000.

    """

    stock_price: float = Field(gt=0)
    volatility: float
    interest_rate: float = Field(gt=0, le=0.2)
    min_stock: float
    max_stock: float
    strategy: list[Strategy] = Field(..., min_length=1, discriminator="type")
    dividend_yield: float = 0.0
    profit_target: float | None = None
    loss_limit: float | None = None
    opt_commission: float = 0.0
    stock_commission: float = 0.0
    compute_expectation: bool = False
    discard_nonbusiness_days: bool = True
    country: Country = "US"
    start_date: dt.date | None = None
    target_date: dt.date | None = None
    days_to_target_date: int = Field(0, ge=0)
    distribution: Distribution = "black-scholes"
    mc_prices_number: int = 100_000
    array_prices: list[float] | None = None

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: list[Strategy]) -> list[Strategy]:
        types = [strategy.type for strategy in v]
        if types.count("closed") > 1:
            raise ValueError("Only one position of type 'closed' is allowed!")
        return v

    @model_validator(mode="after")
    def validate_dates(self) -> "Inputs":
        """
        Validates the dates provided in the inputs.

        Raises:
            ValueError: If the start date is after the target date or if the
                expiration dates are after or on the target date.
            ValueError: If neither start_date and target_date nor
                days_to_target_date are provided.
            ValueError: If there is a strategy expiration date and
                days_to_target_date is provided.
        """
        # Extract the expiration dates from the strategy
        expiration_dates = [
            strategy.expiration
            for strategy in self.strategy
            if isinstance(strategy, OptionStrategy) and isinstance(strategy.expiration, dt.date)
        ]

        # Validate the start and target dates
        if self.start_date and self.target_date:
            if any(expiration_date < self.target_date for expiration_date in expiration_dates):
                raise ValueError("Expiration dates must be after or on target date!")
            if self.start_date >= self.target_date:
                raise ValueError("Start date must be before target date!")
            return self

        # Validate the days_to_target_date
        if self.days_to_target_date:
            if len(expiration_dates) > 0:
                raise ValueError(
                    "You can't mix a strategy expiration with a days_to_target_date."
                )
            return self

        # Raise an error if neither start_date and target_date nor
        # days_to_target_date are provided
        raise ValueError(
            "Either start_date and target_date or days_to_maturity must be provided"
        )

    @model_validator(mode="after")
    def validate_compute_expectation(self) -> "Inputs":
        if self.distribution != "array":
            return self
        if not self.array_prices:
            raise ValueError(
                "Array of prices must be provided if distribution is 'array'."
            )
        if len(self.array_prices) == 0:
            raise ValueError(
                "Array of prices must be provided if distribution is 'array'."
            )
        return self


class BlackScholesInfo(BaseModel):
    """
    Model for storing Black Scholes option information.
    """

    # Call option price
    call_price: float
    # Put option price
    put_price: float
    # Call option delta
    call_delta: float
    # Put option delta
    put_delta: float
    # Call option theta
    call_theta: float
    # Put option theta
    put_theta: float
    # Call and put option gamma
    gamma: float
    # Call and put option vega
    vega: float
    # Call option in-the-money probability
    call_itm_prob: float
    # Put option in-the-money probability
    put_itm_prob: float
    """
    Attributes:
        call_price (float): The price of the call option.
        put_price (float): The price of the put option.
        call_delta (float): The call option's delta.
        put_delta (float): The put option's delta.
        call_theta (float): The call option's theta.
        put_theta (float): The put option's theta.
        gamma (float): The call and put option's gamma.
        vega (float): The call and put option's vega.
        call_itm_prob (float): The probability of the call option being in the money.
        put_itm_prob (float): The probability of the put option being in the money.
    """


class OptionInfo(BaseModel):
    price: float
    delta: float
    theta: float


def init_empty_array() -> np.ndarray:
    return np.array([])


class EngineDataResults(BaseModel):
    """
    Data structure for storing results from the strategy engine.

    Attributes:
        stock_price_array (np.ndarray): Array of stock prices.
        terminal_stock_prices (np.ndarray): Array of terminal stock prices.
        profit (np.ndarray): Array of profit values.
        profit_mc (np.ndarray): Array of Monte Carlo profit values.
        strategy_profit (np.ndarray): Array of total strategy profit values.
        strategy_profit_mc (np.ndarray): Array of Monte Carlo total strategy profit values.
        strike (list[float]): List of strike prices.
        premium (list[float]): List of premiums.
        n (list[int]): List of number of contracts.
        action (list[Action | Literal["n/a"]]): List of actions.
        type (list[StrategyType]): List of strategy types.
    """
    stock_price_array: np.ndarray
    terminal_stock_prices: np.ndarray
    profit: np.ndarray = Field(default_factory=init_empty_array)
    profit_mc: np.ndarray = Field(default_factory=init_empty_array)
    strategy_profit: np.ndarray = Field(default_factory=init_empty_array)
    strategy_profit_mc: np.ndarray = Field(default_factory=init_empty_array)
    strike: list[float] = []  # Strike prices
    premium: list[float] = []  # Premiums
    n: list[int] = []  # Number of contracts
    action: list[Action | Literal["n/a"]] = []  # Actions
    type: list[StrategyType] = []  # Strategy types
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Model configurations

class EngineData(EngineDataResults):
    """
    EngineData class inherits from EngineDataResults and adds additional
    attributes for storing intermediate results from the strategy engine.
    """
    inputs: Inputs  # Inputs to the strategy engine

    _previous_position: list[float] = []  # Previous position of each leg
    _use_bs: list[bool] = []  # Flag to indicate if Black-Scholes model is used
    _profit_ranges: list[Range] = []  # Ranges of stock prices yielding profit
    _profit_target_range: list[Range] = []  # Range of stock prices yielding profit target
    _loss_limit_ranges: list[Range] = []  # Range of stock prices yielding loss limit
    _days_to_maturity: list[int] = []  # Days to maturity for each option
    _days_in_year: int = 365  # Number of days in a year

    days_to_target: int = 30  # Days to target date
    """
    Days to target date. If dates are used, this is ignored and days to target
    is calculated.
    """

    implied_volatility: list[float | np.ndarray] = []  # Implied volatility for each option
    itm_probability: list[float] = []  # ITM probability for each option
    delta: list[float] = []  # Delta for each option
    gamma: list[float] = []  # Gamma for each option
    vega: list[float] = []  # Vega for each option
    theta: list[float] = []  # Theta for each option
    cost: list[float] = []  # Cost of each option

    profit_probability: float = 0.0  # Probability of profit
    profit_target_probability: float = 0.0  # Probability of profit target
    loss_limit_probability: float = 0.0  # Probability of loss limit
    """
    Probability of loss limit. This is not used in the current version of the
    code.
    """


class Outputs(BaseModel):
    """
    data: EngineDataResults
        Further results calculated by the engine.
    probability_of_profit : float
        Probability of the strategy yielding at least $0.01.
    profit_ranges : list
        A list of minimum and maximum stock prices defining
        ranges in which the strategy makes at least $0.01.
    strategy_cost : float
        Total strategy cost.
    per_leg_cost : list
        A list of costs, one per strategy leg.
    implied_volatility : list
        A Python list of implied volatilities, one per strategy leg.
    in_the_money_probability : list
        A list of ITM probabilities, one per strategy leg.
    delta : list
        A list of Delta values, one per strategy leg.
    gamma : list
        A list of Gamma values, one per strategy leg.
    theta : list
        A list of Theta values, one per strategy leg.
    vega : list
        A list of Vega values, one per strategy leg.
    minimum_return_in_the_domain : float
        Minimum return of the strategy within the stock price domain.
    maximum_return_in_the_domain : float
        Maximum return of the strategy within the stock price domain.
    probability_of_profit_target : float, optional
        Probability of the strategy yielding at least the profit target.
    profit_target_ranges : list, optional
        A list of minimum and maximum stock prices defining
        ranges in which the strategy makes at least the profit
        target.
    probability_of_loss_limit : float, optional
        Probability of the strategy losing at least the loss limit.
    average_profit_from_mc : float, optional
        Average profit as calculated from Monte Carlo-created terminal
        stock prices for which the strategy is profitable.
    average_loss_from_mc : float, optional
        Average loss as calculated from Monte Carlo-created terminal
        stock prices for which the strategy ends in loss.
    probability_of_profit_from_mc : float, optional
        Probability of the strategy yielding at least $0.01 as calculated
        from Monte Carlo-created terminal stock prices.
    """

    inputs: Inputs
    data: EngineDataResults
    probability_of_profit: float
    profit_ranges: list[Range]
    per_leg_cost: list[float]
    strategy_cost: float
    minimum_return_in_the_domain: float
    maximum_return_in_the_domain: float
    implied_volatility: list[float]
    in_the_money_probability: list[float]
    delta: list[float]
    gamma: list[float]
    theta: list[float]
    vega: list[float]
    probability_of_profit_target: float | None = None
    profit_target_ranges: list[Range] | None = None
    probability_of_loss_limit: float | None = None
    average_profit_from_mc: float | None = None
    average_loss_from_mc: float | None = None
    probability_of_profit_from_mc: float | None = None

    @property
    def return_in_the_domain_ratio(self) -> float:
        return abs(
            self.maximum_return_in_the_domain / self.minimum_return_in_the_domain
        )
