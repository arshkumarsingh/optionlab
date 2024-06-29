from __future__ import division

import datetime as dt
from datetime import timedelta
from functools import lru_cache

import numpy as np
from holidays import country_holidays

from optionlab.models import Country, EngineData


def get_nonbusiness_days(
    start_date: dt.date, end_date: dt.date, country: Country = "US"
) -> int:
    """
    Calculate the number of non-business days between the start and end date.

    Args:
        start_date (datetime.date): Start date.
        end_date (datetime.date): End date.
        country (str, optional): Country for which the holidays will be counted as
            non-business days. Defaults to "US".

    Returns:
        int: Number of non-business days.

    Raises:
        ValueError: If end date is before start date.
    """
    # Check if end date is after start date
    if end_date < start_date:
        raise ValueError("End date must be after start date!")

    # Calculate the number of days between the start and end date
    n_days = (end_date - start_date).days

    # Initialize the counter for non-business days
    nonbusiness_days = 0

    # Get the holidays for the specified country
    holidays = country_holidays(country)

    # Iterate over each day between the start and end date
    for i in range(n_days):
        current_date = start_date + timedelta(days=i)

        # Check if the current day is a weekend (Saturday or Sunday)
        # or if it is a holiday in the specified country
        if current_date.weekday() >= 5 or current_date.strftime("%Y-%m-%d") in holidays:
            nonbusiness_days += 1

    return nonbusiness_days


def get_pl(data: EngineData, leg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the profit/loss profile of either a leg or the whole strategy.

    Parameters
    ----------
    data : EngineData
        Object containing engine data.
    leg : int, optional
        Index of the leg. Default is None (whole strategy).

    Returns
    -------
    stock_prices : numpy.ndarray
        Sequence of stock prices within the bounds of the stock price domain.
    pl_profile : numpy.ndarray
        Profit/loss profile of either a leg or the whole strategy.

    """
    # If a leg is specified and it is within the bounds of profit array,
    # return the profit profile of that leg
    if data.profit.size > 0 and leg and leg < data.profit.shape[0]:
        return data.stock_price_array, data.profit[leg]

    # Otherwise, return the profit profile of the whole strategy
    return data.stock_price_array, data.strategy_profit


def pl_to_csv(
    data: EngineData, filename: str = "pl.csv", leg: int | None = None
) -> None:
    """
    Saves the profit/loss data to a .csv file.

    Parameters
    ----------
    data : EngineData
        Object containing engine data.
    filename : str, optional
        Name of the .csv file. Default is 'pl.csv'.
    leg : int, optional
        Index of the leg. Default is None (whole strategy).

    Returns
    -------
    None
    """
    # Create an array of stock price and profit/loss profile
    # for either a leg or the whole strategy
    if data.profit.size > 0 and leg and leg < data.profit.shape[0]:
        arr = np.stack((data.stock_price_array, data.profit[leg]))
    else:
        arr = np.stack((data.stock_price_array, data.strategy_profit))

    # Save the array to a .csv file with specified filename
    # and header indicating the columns
    np.savetxt(
        filename,
        arr.transpose(),
        delimiter=",",
        header="StockPrice,Profit/Loss"
    )
