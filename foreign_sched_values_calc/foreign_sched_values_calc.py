"""
This program calculates CG, dividend, schedule FA tables, FSI, form 67 for your
foreign assets.

Copyright (C) 2025  Siddh Raman Pant <@siddhpant on GitHub>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


##############################################################################


# Standard library dependencies.
from collections import OrderedDict
from collections.abc import Callable, Iterator, Hashable
from copy import deepcopy
import csv
from dataclasses import dataclass, field as dataclass_field
from datetime import date as Date, timedelta
from fractions import Fraction
from inspect import cleandoc
import json
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any

# External dependencies.
import numpy as np
import yaml
import yahooquery as yq


##############################################################################


file_dir = Path(__file__).parent / "data"
file_dir.mkdir(exist_ok=True)

output_dir = file_dir / "output"
output_dir.mkdir(exist_ok=True)

input_yaml_path = file_dir / "input.yaml"
input_dict = None
metadata_dict = {}

fx_rate_repo_dir = file_dir / "sbi-fx-ratekeeper"
fx_rate_repo_link = "https://github.com/sahilgupta/sbi-fx-ratekeeper.git"
ttbr_dict = {}

# Use Fraction for avoiding precision errors due to float.
yaml.add_constructor(
    "tag:yaml.org,2002:float",
    lambda loader, node: Fraction(loader.construct_scalar(node)),
    Loader=yaml.SafeLoader
)

# Use Fraction for ints too for consistency.
yaml.add_constructor(
    "tag:yaml.org,2002:int",
    lambda loader, node: Fraction(loader.construct_scalar(node)),
    Loader=yaml.SafeLoader
)

ZERO = Fraction(0)

PARSING_OPENING_LEDGER = False


##############################################################################


def fetch_input() -> None:
    global input_dict

    with open(input_yaml_path) as f:
        input_dict = yaml.safe_load(f)

    for key, value in input_dict["metadata"].items():
        metadata_dict[key] = value

    fy_start = metadata_dict["current_financial_year_start"]
    fy_end = metadata_dict["current_financial_year_end"]

    if (
        fy_end.year - 1 != fy_start.year
        or fy_start.day != 1
        or fy_start.month != 4
        or fy_end.day != 31
        or fy_end.month != 3
    ):
        raise ValueError("Improper financial year provided. Must be from "
                         "1 April to 31 March.")

    cy_start = Date(fy_start.year, 1, 1)
    cy_end = Date(cy_start.year, 12, 31)

    metadata_dict["current_calendar_year_start"] = cy_start
    metadata_dict["current_calendar_year_end"] = cy_end


def fetch_fx_rates() -> None:
    global ttbr_dict

    if fx_rate_repo_dir.exists():
        subprocess.check_output(["git", "pull", "--ff-only"],
                                cwd=fx_rate_repo_dir)
    else:
        subprocess.check_output(["git", "clone", fx_rate_repo_link],
                                cwd=fx_rate_repo_dir.parent)

    prefix, suffix = "SBI_REFERENCE_RATES_", ".csv"
    for p in (fx_rate_repo_dir / "csv_files").glob(f"{prefix}*{suffix}"):
        currency = p.name.removeprefix(prefix).removesuffix(suffix)
        if currency in ttbr_dict:
            raise ValueError(f"Found {currency} again.")

        with open(p) as f:
            csv_rows = list(csv.DictReader(f))

        ttbr_dict[currency] = {
            Date.fromisoformat(row["DATE"].split()[0]): Fraction(row["TT BUY"])
            for row in csv_rows
        }


##############################################################################


def cy_start() -> Date:
    return metadata_dict["current_calendar_year_start"]


def cy_end() -> Date:
    return metadata_dict["current_calendar_year_end"]


def prev_cy_end() -> Date:
    return cy_start() - timedelta(days=1)


def next_cy_start() -> Date:
    return cy_end() + timedelta(days=1)


def fy_start() -> Date:
    return metadata_dict["current_financial_year_start"]


def fy_end() -> Date:
    return metadata_dict["current_financial_year_end"]


def next_fy_start() -> Date:
    return fy_end() + timedelta(days=1)


def date_in_cy(date: Date) -> bool:
    return cy_start() <= date <= cy_end()


def date_in_fy(date: Date) -> bool:
    return fy_start() <= date <= fy_end()


def date_in_fy_advance_tax_installment_1(date: Date) -> bool:
    year = fy_start().year
    return fy_start() <= date <= Date(year, 6, 15)


def date_in_fy_advance_tax_installment_2(date: Date) -> bool:
    year = fy_start().year
    return Date(year, 6, 16) <= date <= Date(year, 9, 15)


def date_in_fy_advance_tax_installment_3(date: Date) -> bool:
    year = fy_start().year
    return Date(year, 9, 16) <= date <= Date(year, 12, 15)


def date_in_fy_advance_tax_installment_4(date: Date) -> bool:
    year_1 = fy_start().year
    year_2 = fy_end().year
    return Date(year_1, 12, 16) <= date <= Date(year_2, 3, 15)


def date_in_fy_advance_tax_installment_5(date: Date) -> bool:
    year = fy_end().year
    return Date(year, 3, 16) <= date <= fy_end()


def date_range(
    start: Date,
    end_exclusive: Date,
    reverse: bool = False
) -> Iterator[Date]:
    dates = np.arange(start, end_exclusive, dtype="datetime64[D]").astype(Date)
    dates = dates.tolist()
    if reverse:
        dates.reverse()

    for date in dates:
        yield date


def more_than_two_years(sell_date: Date, buy_date: Date) -> bool:
    """A month/year in income tax is a calendar month/year, not day count."""
    year_plus_two = buy_date.year + 2

    if (buy_date.month == 2) and (buy_date.day == 29):
        # Current date is in a leap year. Two years later won't be a leap year.
        two_years_plus_one = Date(year_plus_two, 3, 1)
    else:
        two_years_plus_one = Date(year_plus_two, buy_date.month, buy_date.day)

    return sell_date >= two_years_plus_one


##############################################################################


def ensure_unique_key(key: str, seen_keys: set[str], exc: Exception) -> None:
    if key in seen_keys:
        raise exc
    else:
        seen_keys.add(key)


def ensure_unique_yaml_key(key: str) -> None:
    try:
        already_used = ensure_unique_yaml_key.yaml_keys_used
    except AttributeError:
        already_used = set()
        ensure_unique_yaml_key.yaml_keys_used = already_used

    ensure_unique_key(key, already_used,
                      ValueError(f"Duplicate YAML key '{key}'."))


##############################################################################


@dataclass(kw_only=True)
class _BasePostInit:
    # Make class inheritance easy by adding a super().__post_init__().
    def __post_init__(self) -> None:
        pass


##############################################################################


# country_id -> Country
countries = {}


@dataclass(kw_only=True)
class Country(_BasePostInit):
    country_id: str
    name: str
    code: str
    currency: str
    tax_withholding_rate_percent_for_dividend: Fraction
    tax_withholding_rate_percent_for_ltcg: Fraction
    tax_withholding_rate_percent_for_stcg: Fraction
    dtaa_article_dividend: str
    dtaa_article_ltcg: str
    dtaa_article_stcg: str
    dtaa_tax_rate_percent_dividend: Fraction
    dtaa_tax_rate_percent_ltcg: Fraction
    dtaa_tax_rate_percent_stcg: Fraction

    # Last day of month -> (TTBR date, TTBR) mapping.
    _prev_month_last_ttbr_cache: dict[Date, tuple[Date, Fraction]] = \
        dataclass_field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        global countries

        ensure_unique_yaml_key(self.country_id)

        if not self.name:
            raise ValueError("Empty name provided for country with ID "
                             f"{self.country_id}.")
        else:
            # Normalise format to store.
            self.name = self.name.title()

        if self.currency not in ttbr_dict:
            raise ValueError(f"Unknown currency {self.currency} mentioned for "
                             f"country {self.country_id}.")

        if not self.code.isdigit():
            raise ValueError(f"Invalid non-number code {self.code} for "
                             f"country {self.country_id}.")

        if not (article := self.dtaa_article_dividend).isdigit():
            raise ValueError(f"Invalid non-number dividend article {article} "
                             f"for country {self.country_id} DTAA.")

        if not (article := self.dtaa_article_ltcg).isdigit():
            raise ValueError(f"Invalid non-number LTCG article {article} for "
                             "country {self.country_id} DTAA.")

        if not (article := self.dtaa_article_stcg).isdigit():
            raise ValueError(f"Invalid non-number STCG article {article} for "
                             "country {self.country_id} DTAA.")

        countries[self.country_id] = self

    def get_ttbr_on(self, date: Date) -> Fraction:
        return ttbr_dict[self.currency][date]

    def _convert_to_inr_same_day(self, date: Date, amt: Fraction) -> Fraction:
        return amt * self.get_ttbr_on(date)

    def convert_to_inr_on(
        self,
        date: Date,
        amount: Fraction,
        *,
        exact_date_match: bool = False,
    ) -> tuple[Date, Fraction]:
        """
        First try to get value at the exact same date. If we face KeyError due
        to date not being in ttbr_dict, try previous days only if explicitly
        specified.
        """
        try:
            return date, self._convert_to_inr_same_day(date, amount)
        except KeyError:
            if exact_date_match:
                raise ValueError(
                    f"Cannot find TTBR for the given date {date}. Specified "
                    "exact match, so not checking previous days."
                ) from None

        md_key = "__convert_to_inr_same_day_search_last_n"
        n = int(metadata_dict.get(md_key, 5))

        loop_end = date - timedelta(days=n)
        loop_start_excl = date  # Range end is exclusive.

        for loop_dt in date_range(loop_end, loop_start_excl, reverse=True):
            try:
                return loop_dt, self._convert_to_inr_same_day(loop_dt, amount)
            except KeyError:
                continue

        raise RuntimeError(
            f"Cannot find TTBR for the given date {date} and also the last "
            f"{n} days from it. Was there a very big holiday season > {n} "
            "days? If yes, increase number of days to search by specifying "
            f"'{md_key}' in the metadata."
        )

    def _get_prev_month_last_day_ttbr(self, date) -> tuple[Date, Fraction]:
        curr_mon_1st_day = Date(date.year, date.month, 1)
        prev_mon_last_day = curr_mon_1st_day - timedelta(days=1)

        if prev_mon_last_day in self._prev_month_last_ttbr_cache:
            return self._prev_month_last_ttbr_cache[prev_mon_last_day]

        md_key = "__ttbr_prev_month_search_last_n"
        n = int(metadata_dict.get(md_key, 10))
        prev_mon_last_nth_day = prev_mon_last_day - timedelta(days=n)

        for prev_month_date in date_range(
            prev_mon_last_nth_day, curr_mon_1st_day, reverse=True
        ):
            try:
                ret_tup = (prev_month_date, self.get_ttbr_on(prev_month_date))
            except KeyError:
                continue  # Can be holiday so we need to go further back.
            else:
                break
        else:
            # No break.
            raise RuntimeError(
                f"Cannot find TTBR for previous month last day for {date}. "
                f"Searched the last {n} days of the previous month. Was "
                f"there a very big holiday season > {n} days? If yes, "
                f"increase number of days to search by specifying '{md_key}' "
                "in the metadata."
            )

        self._prev_month_last_ttbr_cache[prev_mon_last_day] = ret_tup
        return ret_tup

    def _convert_to_inr_prev_mon_last_day(
        self,
        date: Date,
        amount: Fraction
    ) -> Fraction:
        _, prev_mon_last_ttbr = self._get_prev_month_last_day_ttbr(date)
        return amount * prev_mon_last_ttbr

    def convert_to_inr_for_tax(self, date: Date, amount: Fraction) -> Fraction:
        """
        For taxation for our purposes, we have to use TTBR of the last day of
        previous month.

        Source: Rule 115(1): https://incometaxindia.gov.in/_layouts/15/dit/Pages/viewer.aspx?grp=Rule&cname=CMSID&cval=103120000000007546  # noqa: E501

        > 115. (1)
        > The rate of exchange for the calculation of the value in rupees of
        > any income accruing or arising or deemed to accrue or arise to the
        > assessee in foreign currency or received or deemed to be received by
        > him or on his behalf in foreign currency shall be the telegraphic
        > transfer buying rate of such currency as on the specified date.
        >
        > Explanation. — For the purposes of this rule, —
        >
        > (1) "telegraphic transfer buying rate" shall have the same meaning
        >     as in the Explanation to rule 26;
        >
        > (2) "specified date" means—
        >
        >   (a) in respect of income chargeable under the head "Salaries", the
        >       last day of the month immediately preceding the month in which
        >       the salary is due, or is paid in advance or in arrears;
        >
        >   (b) in respect of income by way of "interest on securities", the
        >       last day of the month immediately preceding the month in which
        >       the income is due;
        >
        >   (c) in respect of income chargeable under the heads "Income from
        >       house property", "Profits and gains of business or profession"
        >       [not being income referred to in clause (d)] and "Income from
        >       other sources" (not being income by way of dividends and
        >       "Interest on securities"), the last day of the previous year of
        >       the assessee;
        >
        >   (d) in respect of income chargeable under the head "Profits and
        >       gains of business or profession" in the case of a non-resident
        >       engaged in the business of operation of ships, the last day of
        >       the month immediately preceding the month in which such income
        >       is deemed to accrue or arise in India;
        >
        >   (e) in respect of income by way of dividends, the last day of the
        >       month immediately preceding the month in which the dividend is
        >       declared, distributed or paid by the company;
        >
        >   (f) in respect of income chargeable under the head "Capital gains",
        >       the last day of the month immediately preceding the month in
        >       which the capital asset is transferred :
        >           Provided that the specified date, in respect of income
        >           referred to in sub-clauses (a) to (f) payable in foreign
        >           currency and from which tax has been deducted at source
        >           under rule 26, shall be the date on which the tax was
        >           required to be deducted] under the provisions of the
        >           Chapter XVII-B.
        >
        > (2) Nothing contained in sub-rule (1) shall apply in respect of
        >     income referred to in clause (c) of the Explanation to sub-rule
        >     (1) where such income is received in, or brought into India by
        >     the assessee or on his behalf before the specified date in
        >     accordance with the provisions of the Foreign Exchange Regulation
        >     Act, 1973 (46 of 1973).
        """
        if not date_in_fy(date):
            raise ValueError("Attempted to get INR value for tax for a date "
                             "outside the financial year (Apr - Mar).")

        return self._convert_to_inr_prev_mon_last_day(date, amount)

    def convert_to_inr_for_FA_income(self, amount: Fraction) -> Fraction:
        """
        For reporting income in CY in schedule FA, we must use TTBR of the end
        of the calendar year.

        Source: https://www.incometax.gov.in/iec/foportal/sites/default/files/2024-11/Enhancing%20Tax%20Transparency%20on%20Foreign%20Assets%20and%20Income.pdf  # noqa: E501

        > For the purpose of this Schedule, the rate of exchange for conversion
        > of the peak balance or value of investment or the amount of foreign
        > sourced income in Indian currency shall be the “telegraphic transfer
        > buying rate” of the foreign currency as on the date of peak balance
        > in the account or on the date of investment or the closing date of
        > the calendar year ending as on 31st December.
        """
        # Advance to next year to take advantage of prev-month-last-day logic.
        return self._convert_to_inr_prev_mon_last_day(next_cy_start(), amount)

    def get_peak_value_in_cy(
        self,
        peak_value_native_on_date_callback: Callable[[Date], Fraction],
        *,
        date_start: Date = None,
        date_end_exclusive: Date = None,
        in_inr: bool,
        extra_cb_args: list[Any] = [],
        extra_inr_value_to_consider: Fraction = None,
    ) -> Fraction:
        """
        We can't simply use the native peak value directly for INR peak value,
        since the TTBR fluctuates each day.
        """
        if not in_inr and extra_inr_value_to_consider is not None:
            raise ValueError("extra_inr_value_to_consider specified when "
                             "specified in_inr to be False.")

        if date_start is None:
            date_start = cy_start()

        if date_end_exclusive is None:
            date_end_exclusive = next_cy_start()

        peak_value = ZERO

        for date in date_range(date_start, date_end_exclusive):
            day_peak = peak_value_native_on_date_callback(date, *extra_cb_args)

            if in_inr:
                _, day_peak = self.convert_to_inr_on(date, day_peak)

            if peak_value < day_peak:
                peak_value = day_peak

        if in_inr and extra_inr_value_to_consider is not None:
            if peak_value < extra_inr_value_to_consider:
                peak_value = extra_inr_value_to_consider

        return peak_value


@dataclass(kw_only=True)
class MapToCountry(_BasePostInit):
    country_id: str

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.country_id not in countries:
            raise ValueError(f"Invalid country ID '{self.country_id}'.")

    @property
    def country(self) -> Country:
        return countries[self.country_id]


##############################################################################


ticker_history_type = dict[Date, dict[str, Fraction]]


def fetch_ticker_prices(ticker: str) -> ticker_history_type:
    price_dir = file_dir / "saved_share_prices"
    price_dir.mkdir(exist_ok=True)

    # We add a buffer for start since 1st Jan is a holiday.
    start = cy_start() - timedelta(days=10)
    end_excl = next_fy_start()  # Exclusive end for range.

    saved_history_path = price_dir / f"{ticker}_{start}_{end_excl}.pickle"

    if saved_history_path.exists():
        with open(saved_history_path, "rb") as f:
            return pickle.load(f)

    # We are here only if the file doesn't exist.
    history = {}

    err_dict = {ticker: f"Quote not found for symbol: {ticker}"}

    # Causes network calls.
    ticker_obj = yq.Ticker(ticker)
    if ticker_obj.quote_type == err_dict:
        raise ValueError(f"Invalid ticker '{ticker}'.")

    try:
        historical_df = ticker_obj.history(start=start, end=end_excl,
                                           adj_ohlc=False)
    except Exception as e:
        raise ValueError("Failed to fetch historical data for ticker "
                         f"{ticker}: {e}")

    for date in date_range(start, end_excl):
        key = (ticker, date)

        try:
            open_val = Fraction(historical_df.loc[key, "open"])
            high_val = Fraction(historical_df.loc[key, "high"])
            low_val = Fraction(historical_df.loc[key, "low"])
            close_val = Fraction(historical_df.loc[key, "close"])
        except KeyError:
            history[date] = None
        else:
            history[date] = {
                "open": open_val,
                "high": high_val,
                "low": low_val,
                "close": close_val
            }

    with open(saved_history_path, "wb") as f:
        pickle.dump(history, f)

    return history


###############################################################################


# entity_id -> Entity
entities = {}
seen_tickers: set[str] = set()


@dataclass(kw_only=True)
class Entity(MapToCountry):
    entity_id: str
    entity_type: str
    ticker: str
    name: str
    company_name: str
    company_address_without_zipcode: str
    company_address_zipcode: str

    ticker_obj: yq.Ticker = dataclass_field(init=False)
    _history: ticker_history_type = dataclass_field(init=False)

    @property
    def cash_type(self) -> bool:
        return self.entity_type in ("cash_parking_mmf", "cash_wallet")

    @property
    def nature(self) -> str:
        match self.entity_type:
            case "stock":
                return "Listed company"
            case "etf":
                return "Exchange Traded Fund"
            case "mf":
                return "Mutual Fund"
            case "cash_parking_mmf":
                return "Money Market Fund used by broker to park cash"
            case "cash_wallet":
                return "Cash wallet"
            case _:
                raise ValueError(f"Invalid entity_type '{self.entity_type}' "
                                 f"for entity '{self.entity_id}').")

    def __post_init__(self) -> None:
        global entities, seen_tickers

        super().__post_init__()
        ensure_unique_yaml_key(self.entity_id)

        # nature() raises error when unknown entity_type is encountered.
        _ = self.nature

        if self.entity_type == "cash_wallet":
            err_common = (f"'{self.entity_type}' is of type cash_wallet but "
                          "has non-null ")
            if self.ticker is not None:
                raise ValueError(err_common + "'ticker'.")
            if self.name is not None:
                raise ValueError(err_common + "'name'.")

        # Let's verify ticker and save historical prices.
        if not self.cash_type:
            ensure_unique_key(
                self.ticker,
                seen_tickers,
                ValueError(
                    f"Duplicate ticker {self.ticker} for entity "
                    f"{self.entity_id}."
                )
            )

            self._history = fetch_ticker_prices(self.ticker)

        entities[self.entity_id] = self

    @property
    def country(self) -> Country:
        return countries[self.country_id]

    def get_share_price(
        self,
        date: Date,
        ohlc_column: str,  # open, high, low, close.
        *,
        for_peak_value_reporting: bool = False,
    ) -> Fraction:
        if self.cash_type:
            return 1

        share_prices = self._history[date]
        if share_prices is None:
            if not for_peak_value_reporting:
                raise ValueError(
                    f"No share price data available on {date} for ticker "
                    f"{self.ticker} of entity {self.entity_id}. Is the date a "
                    "holiday? How were you able to do the transaction on a "
                    "holiday?"
                )

            # For peak value reporting in schedule FA, we should use the last
            # closing price.

            md_key = "__share_price_search_last_n"
            n = int(metadata_dict.get(md_key, 4))
            date_end = date - timedelta(days=n)

            for date in date_range(date_end, date, reverse=True):
                share_prices = self._history[date]
                if share_prices is not None:
                    break
            else:
                # No break.
                raise ValueError(
                    f"No share price data available on {date} for ticker "
                    f"{self.ticker} of entity {self.entity_id}. Tried to "
                    f"search previous {n} days but no data for all of them "
                    f"too. Was there a very big holiday season? If yes, "
                    "increase number of days to search by specifying "
                    f"'{md_key}' in the metadata."
                )

        return share_prices[ohlc_column]


@dataclass(kw_only=True)
class MapToEntity(_BasePostInit):
    entity_id: str

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.entity_id not in entities:
            raise ValueError(f"Invalid entity '{self.entity_id}'.")

    @property
    def entity(self) -> Entity:
        return entities[self.entity_id]

    @property
    def country(self) -> Country:
        return self.entity.country


###############################################################################


@dataclass(kw_only=True)
class _TotalValueMixin(_BasePostInit):
    date: Date

    def __post_init__(self) -> None:
        super().__post_init__()

    @property
    def gross_total_value_native(self) -> Fraction:
        raise NotImplementedError

    @property
    def net_total_value_native(self) -> Fraction:
        raise NotImplementedError

    def _total_value_ttbr_and_inr_same_day(
        self,
        attr: str
    ) -> tuple[Date, Fraction]:
        return self.country.convert_to_inr_on(self.date, getattr(self, attr))

    def _total_value_inr_for_FA_income(self, attr: str) -> Fraction:
        return self.country.convert_to_inr_for_FA_income(getattr(self, attr))

    @property
    def applicable_ttbr_date(self) -> Date:
        attr = "gross_total_value_native"
        date, _ = self._total_value_ttbr_and_inr_same_day(attr)
        return date

    @property
    def gross_total_value_inr_same_day(self) -> Fraction:
        attr = "gross_total_value_native"
        _, value = self._total_value_ttbr_and_inr_same_day(attr)
        return value

    @property
    def gross_total_value_inr_for_FA_income(self) -> Fraction:
        return self._total_value_inr_for_FA_income("gross_total_value_native")

    @property
    def net_total_value_inr_same_day(self) -> Fraction:
        attr = "net_total_value_native"
        _, value = self._total_value_ttbr_and_inr_same_day(attr)
        return value

    @property
    def net_total_value_inr_for_FA_income(self) -> Fraction:
        return self._total_value_inr_for_FA_income("net_total_value_native")


@dataclass(kw_only=True)
class _TaxWithholdingAndFees(_BasePostInit):
    misc_fees: Fraction
    tax_withholding_rate: Fraction
    tax_withholding_amount_native: Fraction
    calculate_tax_withholding_amount_from_attribute: str = None

    @property
    def tax_withholding_rate_in_country_for_this_type(self) -> Fraction:
        raise NotImplementedError

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.misc_fees < 0:
            raise ValueError("misc_fees < 0.")

        if not (0 <= self.tax_withholding_rate < 1):
            raise ValueError("tax_withholding_rate not in range [0, 1).")

        if self.tax_withholding_amount_native < 0:
            raise ValueError("tax_withholding_amount < 0.")

        if self.gross_total_value_native < self.tax_withholding_amount_native:
            raise ValueError("total_amount < tax_withholding_amount.")

        if (
            self.tax_withholding_rate == 0
            and self.tax_withholding_amount_native != 0
        ):
            raise ValueError("tax_withholding_rate == 0 but "
                             "tax_withholding_amount != 0")

        country_rate = self.tax_withholding_rate_in_country_for_this_type

        if (
            self.tax_withholding_rate != 0
            and self.tax_withholding_rate != country_rate
        ):
            given_pc = self.tax_withholding_rate * 100
            country_pc = country_rate * 100
            raise ValueError(f"tax_withholding_rate {given_pc}% is not equal "
                             f"to the country's rate {country_pc}%.")

        if (
            self.tax_withholding_amount_native == 0
            and self.tax_withholding_rate != 0
        ):
            if self.calculate_tax_withholding_amount_from_attribute is None:
                raise ValueError(
                    "tax_withholding_amount == 0 but tax_withholding_rate != 0"
                    " and calculate_tax_withholding_amount_from_attribute not"
                    " specified."
                )

            attr_name = self.calculate_tax_withholding_amount_from_attribute
            amount = getattr(self, attr_name)

            if amount > 0:
                self.tax_withholding_amount_native = \
                    amount * self.tax_withholding_rate

    @property
    def total_deduction_native(self) -> Fraction:
        return self.misc_fees + self.tax_withholding_amount_native

    @property
    def tax_withheld_inr(self) -> Fraction:
        # This is for objects which have the date and country attributes.
        return self.country.convert_to_inr_for_tax(
            self.date, self.tax_withholding_amount_native
        )


###############################################################################


# txn_id -> _Transaction
all_transactions = OrderedDict()


@dataclass(kw_only=True)
class _Transaction(MapToEntity, _TotalValueMixin):
    txn_id: str

    def __post_init__(self) -> None:
        global all_transactions

        super().__post_init__()

        if self.txn_id.startswith("__"):
            raise ValueError("Transaction IDs cannot start with '__'.")

        ensure_unique_yaml_key(self.txn_id)
        all_transactions[self.txn_id] = self


###############################################################################


# txn_id -> _ShareTransaction
all_share_transactions = OrderedDict()


@dataclass(kw_only=True)
class _ShareTransaction(_Transaction):
    units: Fraction
    cost_per_unit: Fraction

    def __post_init__(self) -> None:
        global all_share_transactions

        super().__post_init__()

        if self.units <= 0:
            raise ValueError("units <= 0")

        if self.cost_per_unit <= 0:
            raise ValueError("cost_per_unit <= 0 for transaction "
                             f"'{self.txn_id}'.")

        all_share_transactions[self.txn_id] = self

    @property
    def buy(self) -> bool:
        raise NotImplementedError

    @property
    def gross_total_value_native(self) -> Fraction:
        return self.units * self.cost_per_unit


@dataclass(kw_only=True)
class _BuyTransaction(_ShareTransaction):
    @property
    def buy(self) -> bool:
        return True

    @property
    def total_buy_value_inr_for_initial_acquire(self) -> Fraction:
        raise NotImplementedError

    @property
    def total_buy_value_inr_for_tax(self) -> Fraction:
        raise AttributeError("Buy value is related to sell value for the "
                             "purposes of capital gain taxation. Use details "
                             "from the sell object.")


@dataclass(kw_only=True)
class ManualBuyTransaction(_BuyTransaction):
    @property
    def net_total_value_native(self) -> Fraction:
        return self.gross_total_value_native

    @property
    def total_buy_value_inr_for_initial_acquire(self) -> Fraction:
        return self.gross_total_value_inr_same_day


@dataclass(kw_only=True)
class VestingTransaction(_BuyTransaction):
    """
    The merchant banker can use a different TTBR value due to section 26, which
    states that (source https://indiankanoon.org/doc/17586817/):

    > 26. Rate of exchange for the purpose of deduction of tax at source on
    > income payable in foreign currency.
    >
    > For the purpose of deduction of tax at source on any income payable in
    > foreign currency, the rate of exchange for the calculation of the value
    > in rupees of such income payable to an assessee outside India shall be
    > the telegraphic transfer buying rate of such currency as on the date on
    > which the tax is required to be deducted at source under the provisions
    > of Chapter XVIIB by the person responsible for paying such income.

    IOW, TTBR to use FOR TAXATION is of the day when the TDS is deducted (I
    suppose when it is officially registered by the employer as tax gets
    deducted on vest), which is the salary day occurring on the last day of the
    month (RSU is in the perquisite part of salary).

    Note: We do NOT override and disable the FA income methods which use the
    TTBR of the day of acquisition. This is because the guidelines say the
    initial value of acquisition has to use TTBR of the same date - see the
    docstring for Country.convert_to_inr_for_FA_income(). So WHEN REPORTING IN
    SCHEDULE FA, we DO NOT use the merchant banker's TTBR.

    Remember: Taxation is disjoint from FA reporting. FA reporting probably
    needs to match the values reported to India via FATCA/CRS.
    """

    merchant_ttbr: Fraction

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.merchant_ttbr <= 0:
            raise ValueError("merchant_ttbr <= 0 for transaction "
                             f"'{self.txn_id}'.")

    @property
    def gross_total_value_native(self) -> Fraction:
        return self.units * self.cost_per_unit

    @property
    def net_total_value_native(self) -> Fraction:
        return self.gross_total_value_native

    @property
    def total_buy_value_inr_for_initial_acquire(self) -> Fraction:
        """Yes, we don't use merchant's TTBR. Read the above paragraph."""
        return self.gross_total_value_inr_same_day

    @property
    def total_value_inr_merchant_banker(self) -> Fraction:
        return self.gross_total_value_native * self.merchant_ttbr


@dataclass(kw_only=True)
class SellTransaction(_ShareTransaction, _TaxWithholdingAndFees):
    buy_txn_id: str

    def __post_init__(self) -> None:
        super().__post_init__()

        try:
            buy_txn = all_share_transactions[self.buy_txn_id]
        except KeyError:
            raise ValueError(f"Invalid buy transaction '{self.buy_txn_id}'.")

        if not isinstance(buy_txn, _BuyTransaction):
            raise ValueError(f"Specified buy txn ID '{self.buy_txn_id}' is "
                             f"not a buy txn (sell txn ID {self.txn_id}).")

        if buy_txn.date > self.date:
            raise ValueError("buy_date > sell_date for transaction "
                             f"'{self.txn_id}'.")

    @property
    def buy(self) -> bool:
        return False

    @property
    def buy_txn_obj(self) -> _BuyTransaction:
        return all_share_transactions[self.buy_txn_id]

    @property
    def net_total_value_native(self) -> Fraction:
        return self.gross_total_value_native - self.total_deduction_native

    @property
    def total_sell_value_inr_for_tax(self) -> Fraction:
        return self.country.convert_to_inr_for_tax(
            self.date, self.gross_total_value_native
        )

    @property
    def gain_amount_native(self) -> Fraction:
        gain_per_unit = self.cost_per_unit - self.buy_txn_obj.cost_per_unit
        return self.units * gain_per_unit

    @property
    def gain_amount_inr_for_tax(self) -> Fraction:
        """
        We convert the gain value from native to INR instead of separately
        converting buy and sell price from their respective dates. This is
        because you can have 0 actual gain, while TTBR fluctation / difference
        when separately converting results in a calculated gain or loss when
        that is not the case.

        This is supported by a ruling of the Income Tax Appellate Tribunal in
        2020 (links also saved in the Internet Archive):
        - https://indiankanoon.org/doc/91892030/ (see paragraph 14)
        - https://bcajonline.org/journal/section-45-rule-115-foreign-exchange-gain-realised-on-remittance-of-amount-received-on-redemption-of-shares-at-par-in-foreign-subsidiary-is-a-capital-receipt-not-liable-to-tax/  # noqa: E501
        """
        return self.country.convert_to_inr_for_tax(self.date,
                                                   self.gain_amount_native)

    @property
    def scaled_buying_price_inr_for_gain_tax(self) -> Fraction:
        """
        For tax filing in schedule CG, we need to show the buying and selling
        price. As mentioned above, gain in INR is to be calculated by using
        the gain in native currency and then converting it to INR. By using the
        divisibility principle of mathematics, this implies the TTBR used for
        gain value conversion is the one to be used for buy value conversion.
        """
        cost_on_buy_for_units = self.units * self.buy_txn_obj.cost_per_unit
        return self.country.convert_to_inr_for_tax(self.date,
                                                   cost_on_buy_for_units)

    @property
    def is_long_term(self) -> bool:
        """
        Budget 2024 introduced a split in taxation. We only support the new
        rates (the old style will result in unnecessary code debt, just for
        it to become obsolete in a few days). See:
            - https://pib.gov.in/PressReleaseIframePage.aspx?PRID=2036604
            - https://zerodha.com/varsity/chapter/foreign-stocks-and-taxation/
        """
        return more_than_two_years(self.date, self.buy_txn_obj.date)

    @property
    def is_short_term(self) -> bool:
        return not self.is_long_term

    @property
    def is_gain(self) -> bool:
        return self.gain_amount_native > 0

    @property
    def is_zero_gain(self) -> bool:
        return self.gain_amount_native == 0

    @property
    def is_loss(self) -> bool:
        return self.gain_amount_native < 0

    @property
    def capital_gain_tax_inr(self) -> Fraction:
        """Return the CG tax, 0 if not a gain."""
        if not self.is_gain:
            return ZERO

        if self.is_long_term:
            tax = self.gain_amount_inr_for_tax * 0.125
        else:
            slab_rate = metadata_dict["slab_rate_percent_for_stcg"] / 100
            tax = self.gain_amount_inr_for_tax * slab_rate

        return tax

    @property
    def tax_withholding_rate_in_country_for_this_type(self) -> Fraction:
        if self.is_long_term:
            return self.country.tax_withholding_rate_percent_for_ltcg / 100
        else:
            return self.country.tax_withholding_rate_percent_for_stcg / 100


###############################################################################

# txn_id -> _CashTransaction
all_cash_transactions = OrderedDict()


@dataclass(kw_only=True)
class _CashTransaction(_Transaction, _TotalValueMixin):
    amount: Fraction

    def __post_init__(self) -> None:
        global all_cash_transactions

        super().__post_init__()

        if self.amount <= 0:
            raise ValueError("amount <= 0 for cash transaction "
                             f"'{self.txn_id}'.")

        if not self.entity.cash_type:
            raise ValueError(
                f"Attempted cash transaction with non-cash entity "
                f"'{self.entity_id}'."
            )

        all_cash_transactions[self.txn_id] = self

    @property
    def credit(self) -> bool:
        raise NotImplementedError

    @property
    def gross_total_value_native(self) -> Fraction:
        return self.amount


@dataclass(kw_only=True)
class CashCreditTransaction(_CashTransaction, _TaxWithholdingAndFees):
    def __post_init__(self) -> None:
        super().__post_init__()

        if self.total_deduction_native > self.amount:
            raise ValueError("total_deduction_native > gross amount for cash "
                             f"transaction '{self.txn_id}'.")

    @property
    def credit(self) -> bool:
        return True

    @property
    def tax_withholding_rate_in_country_for_this_type(self) -> Fraction:
        return self.country.tax_withholding_rate_percent_for_dividend / 100

    @property
    def net_total_value_native(self) -> Fraction:
        return self.gross_total_value_native - self.total_deduction_native

    @property
    def net_total_value_inr_for_tax(self) -> Fraction:
        return self.country.convert_to_inr_for_tax(self.date,
                                                   self.net_total_value_native)


@dataclass(kw_only=True)
class CashDebitTransaction(_CashTransaction):
    ghar_vaapsi: bool

    @property
    def credit(self) -> bool:
        return False

    @property
    def net_total_value_native(self) -> Fraction:
        return self.gross_total_value_native


###############################################################################


@dataclass(kw_only=True)
class _TransactionTracker(_BasePostInit):
    _history: dict[Date, list[str]] = dataclass_field(init=False,
                                                      default_factory=dict)
    _total_cache: dict[str, Fraction] = dataclass_field(init=False,
                                                        default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._total_cache["gross_total_value_native"] = ZERO
        self._total_cache["net_total_value_native"] = ZERO

    def _get_txn_from_id(self, txn_id: str) -> _Transaction:
        """
        Fetch transaction using the txn_id appropriately, doing a check for its
        type too.
        """
        raise NotImplementedError

    def _disallow_other_txn_on_its_date(self, txn: _Transaction) -> None:
        """
        If you want to ensure a single txn on a date, raise exception when a
        txn already exists. Otherwise just return normally.
        """
        raise NotImplementedError

    def add_txn(self, txn_id: str) -> None:
        txn = self._get_txn_from_id(txn_id)
        self._disallow_other_txn_on_its_date(txn)

        if txn.date not in self._history:
            self._history[txn.date] = []

        # When making a transaction we ensure a unique ID, and thus the date
        # will always be the same for a given txn_id.
        if txn_id in self._history[txn.date]:
            raise ValueError(f"Attempted to add {txn_id} again in tracker.")

        self._history[txn.date].append(txn_id)

        # Optimisation for properties so that we don't have to loop everytime.
        for key in self._total_cache.keys():
            self._total_cache[key] += getattr(txn, key)

    def _sum_txn_attr_between_dates(
        self,
        attr: str,
        from_date: Date | None,  # Specify None to match all before to_date.
        to_date: Date,
        filter_cb: Callable[[_Transaction], bool],
    ) -> Any:
        if from_date is None:
            # Nothing can be before previous CY end.
            from_date = prev_cy_end()

        return sum(
            getattr(txn, attr)
            for txn_id_list in self._history.values()
            for txn_id in txn_id_list
            if (
                # We will never have None, this is just a hack to set the var.
                (txn := self._get_txn_from_id(txn_id)) is not None
                and from_date <= txn.date <= to_date
                and filter_cb(txn)
            )
        )

    def gross_total_value_native_between_dates(
        self,
        from_date: Date,
        to_date: Date,
        filter_cb: Callable[[_Transaction], bool] = lambda _: True,
    ) -> Fraction:
        return self._sum_txn_attr_between_dates(
            "gross_total_value_native", from_date, to_date, filter_cb
        )

    def net_total_value_native_between_dates(
        self,
        from_date: Date,
        to_date: Date,
        filter_cb: Callable[[_Transaction], bool] = lambda _: True,
    ) -> Fraction:
        return self._sum_txn_attr_between_dates(
            "net_total_value_native", from_date, to_date, filter_cb
        )

    @property
    def gross_total_value_native(self) -> Fraction:
        """Optimised so that we don't have to loop everytime."""
        return self._total_cache["gross_total_value_native"]

    @property
    def net_total_value_native(self) -> Fraction:
        """Optimised so that we don't have to loop everytime."""
        return self._total_cache["net_total_value_native"]


@dataclass(kw_only=True)
class DividendTracker(_TransactionTracker):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _get_txn_from_id(self, txn_id: str) -> CashCreditTransaction:
        try:
            dividend_txn = all_cash_transactions[txn_id]
        except KeyError:
            raise ValueError(f"Dividend txn_id {txn_id} not in cash txn dict.")

        if not isinstance(dividend_txn, CashCreditTransaction):
            raise ValueError(f"Dividend txn_id {txn_id} is not a "
                             "CashCreditTransaction.")

        return dividend_txn

    def _disallow_other_txn_on_its_date(
        self,
        txn: CashCreditTransaction
    ) -> None:
        if txn.date in self._history:
            raise ValueError(
                f"Dividend txn_id {txn.txn_id} is of date {txn.date}, but "
                "there already exists a dividend on that date."
            )

    def net_total_value_inr_for_tax_between_dates(
        self,
        from_date: Date,
        to_date: Date,
        filter_cb: Callable[[_Transaction], bool] = lambda _: True,
    ) -> Fraction:
        return self._sum_txn_attr_between_dates(
            "net_total_value_inr_for_tax", from_date, to_date, filter_cb
        )

    def total_tax_withheld_inr_between_dates(
        self,
        from_date: Date,
        to_date: Date,
        filter_cb: Callable[[_Transaction], bool] = lambda _: True,
    ) -> Fraction:
        return self._sum_txn_attr_between_dates(
            "tax_withheld_inr", from_date, to_date, filter_cb
        )

    def net_total_amount_inr_for_which_tax_was_withheld_between_dates(
        self,
        from_date: Date,
        to_date: Date,
        filter_cb: Callable[[_Transaction], bool] = lambda _: True,
    ) -> Fraction:
        return self._sum_txn_attr_between_dates(
            "net_total_value_inr_for_tax",
            from_date,
            to_date,
            lambda txn: (txn.tax_withholding_amount_native != 0
                         and filter_cb(txn)),
        )


@dataclass(kw_only=True)
class CashCreditTracker(_TransactionTracker):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _get_txn_from_id(self, txn_id: str) -> CashCreditTransaction:
        try:
            credit_txn = all_cash_transactions[txn_id]
        except KeyError:
            raise ValueError(f"Cash credit txn_id {txn_id} not in cash txn "
                             "dict.")

        if not isinstance(credit_txn, CashCreditTransaction):
            raise ValueError(f"Cash credit txn_id {txn_id} is not a "
                             "CashCreditTransaction.")

        return credit_txn

    def _disallow_other_txn_on_its_date(self, _) -> None:
        return


@dataclass(kw_only=True)
class CashDebitTracker(_TransactionTracker):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _get_txn_from_id(self, txn_id: str) -> CashDebitTransaction:
        try:
            debit_txn = all_cash_transactions[txn_id]
        except KeyError:
            raise ValueError(f"Cash debit txn_id {txn_id} not in cash txn "
                             "dict.")

        if not isinstance(debit_txn, CashDebitTransaction):
            raise ValueError(f"Cash debit txn_id {txn_id} is not a "
                             "CashDebitTransaction.")

        return debit_txn

    def _disallow_other_txn_on_its_date(self, _) -> None:
        return

    def total_ghar_vaapsi_native_between_dates(
        self,
        from_date: Date,
        to_date: Date,
        cb: Callable[[_Transaction], bool] = lambda _: True,
    ) -> Fraction:
        return self._sum_txn_attr_between_dates(
            "net_total_value_native",
            from_date,
            to_date,
            lambda txn: cb(txn) and txn.ghar_vaapsi,
        )


@dataclass(kw_only=True)
class SellingTracker(_TransactionTracker):
    def __post_init__(self) -> None:
        super().__post_init__()
        self._total_cache["units"] = ZERO

    def _get_txn_from_id(self, txn_id: str) -> SellTransaction:
        try:
            sell_txn = all_share_transactions[txn_id]
        except KeyError:
            raise ValueError(f"Sell txn_id {txn_id} not in share txn dict.")

        if not isinstance(sell_txn, SellTransaction):
            raise ValueError(f"Sell txn_id {txn_id} is not a SellTransaction.")

        return sell_txn

    def _disallow_other_txn_on_its_date(self, _) -> None:
        return

    def total_units_between_dates(
        self,
        from_date: Date,
        to_date: Date,
        filter_cb: Callable[[_Transaction], bool] = lambda _: True,
    ) -> Fraction:
        return self._sum_txn_attr_between_dates(
            "units", from_date, to_date, filter_cb
        )

    @property
    def total_units(self) -> Fraction:
        return self._total_cache["units"]

    def gain_amount_inr_for_tax_between_dates(
        self,
        from_date: Date,
        to_date: Date,
        filter_cb: Callable[[_Transaction], bool] = lambda _: True,
    ) -> Fraction:
        return self._sum_txn_attr_between_dates(
            "gain_amount_inr_for_tax", from_date, to_date, filter_cb
        )

    def capital_gain_tax_inr_between_dates(
        self,
        from_date: Date,
        to_date: Date,
        filter_cb: Callable[[_Transaction], bool] = lambda _: True,
    ) -> Fraction:
        return self._sum_txn_attr_between_dates(
            "capital_gain_tax_inr", from_date, to_date, filter_cb
        )

    def total_tax_withheld_inr_between_dates(
        self,
        from_date: Date,
        to_date: Date,
        filter_cb: Callable[[_Transaction], bool] = lambda _: True,
    ) -> Fraction:
        return self._sum_txn_attr_between_dates(
            "tax_withheld_inr", from_date, to_date, filter_cb
        )

    def total_ltcg_amount_inr_for_tax_between_dates(
        self,
        from_date: Date,
        to_date: Date,
        filter_cb: Callable[[_Transaction], bool] = lambda _: True,
    ) -> Fraction:
        return self._sum_txn_attr_between_dates(
            "gain_amount_inr_for_tax",
            from_date,
            to_date,
            lambda txn: txn.is_long_term and filter_cb(txn),
        )

    def total_stcg_amount_inr_for_tax_between_dates(
        self,
        from_date: Date,
        to_date: Date,
        filter_cb: Callable[[_Transaction], bool] = lambda _: True,
    ) -> Fraction:
        return self._sum_txn_attr_between_dates(
            "gain_amount_inr_for_tax",
            from_date,
            to_date,
            lambda txn: txn.is_short_term and filter_cb(txn),
        )

    def total_ltcg_tax_inr_between_dates(
        self,
        from_date: Date,
        to_date: Date,
        filter_cb: Callable[[_Transaction], bool] = lambda _: True,
    ) -> Fraction:
        return self._sum_txn_attr_between_dates(
            "capital_gain_tax_inr",
            from_date,
            to_date,
            lambda txn: txn.is_long_term and filter_cb(txn),
        )

    def total_stcg_tax_inr_between_dates(
        self,
        from_date: Date,
        to_date: Date,
        filter_cb: Callable[[_Transaction], bool] = lambda _: True,
    ) -> Fraction:
        return self._sum_txn_attr_between_dates(
            "capital_gain_tax_inr",
            from_date,
            to_date,
            lambda txn: txn.is_short_term and filter_cb(txn),
        )

    def total_ltcg_tax_withheld_inr_between_dates(
        self,
        from_date: Date,
        to_date: Date,
        filter_cb: Callable[[_Transaction], bool] = lambda _: True,
    ) -> Fraction:
        return self._sum_txn_attr_between_dates(
            "tax_withheld_inr",
            from_date,
            to_date,
            lambda txn: txn.is_long_term and filter_cb(txn),
        )

    def total_stcg_tax_withheld_inr_between_dates(
        self,
        from_date: Date,
        to_date: Date,
        filter_cb: Callable[[_Transaction], bool] = lambda _: True,
    ) -> Fraction:
        return self._sum_txn_attr_between_dates(
            "tax_withheld_inr",
            from_date,
            to_date,
            lambda txn: txn.is_short_term and filter_cb(txn),
        )

    def ltcg_amount_inr_for_which_tax_was_withheld_between_dates(
        self,
        from_date: Date,
        to_date: Date,
        filter_cb: Callable[[_Transaction], bool] = lambda _: True,
    ) -> Fraction:
        return self._sum_txn_attr_between_dates(
            "gain_amount_inr_for_tax",
            from_date,
            to_date,
            lambda txn: (txn.is_long_term
                         and txn.tax_withholding_amount_native != 0
                         and filter_cb(txn)),
        )

    def stcg_amount_inr_for_which_tax_was_withheld_between_dates(
        self,
        from_date: Date,
        to_date: Date,
        filter_cb: Callable[[_Transaction], bool] = lambda _: True,
    ) -> Fraction:
        return self._sum_txn_attr_between_dates(
            "gain_amount_inr_for_tax",
            from_date,
            to_date,
            lambda txn: (txn.is_short_term
                         and txn.tax_withholding_amount_native != 0
                         and filter_cb(txn)),
        )

    def _total_buy_sell_gain_tuple_inr_for_tax_between_dates(
        self,
        from_date: Date,
        to_date: Date,
        filter_cb: Callable[[_Transaction], bool],
    ) -> tuple[Fraction, Fraction, Fraction]:
        sum_func = self._sum_txn_attr_between_dates
        common_args = (from_date, to_date, filter_cb)

        buy = sum_func("scaled_buying_price_inr_for_gain_tax", *common_args)
        sell = sum_func("total_sell_value_inr_for_tax", *common_args)
        gain = sum_func("gain_amount_inr_for_tax", *common_args)

        return (buy, sell, gain)

    def total_buy_sell_ltcg_tuple_inr_for_tax_between_dates(
        self,
        from_date: Date,
        to_date: Date,
        filter_cb: Callable[[_Transaction], bool] = lambda _: True,
    ) -> tuple[Fraction, Fraction, Fraction]:
        return self._total_buy_sell_gain_tuple_inr_for_tax_between_dates(
            from_date,
            to_date,
            lambda txn: txn.is_long_term and filter_cb(txn)
        )

    def total_buy_sell_stcg_tuple_inr_for_tax_between_dates(
        self,
        from_date: Date,
        to_date: Date,
        filter_cb: Callable[[_Transaction], bool] = lambda _: True,
    ) -> tuple[Fraction, Fraction, Fraction]:
        return self._total_buy_sell_gain_tuple_inr_for_tax_between_dates(
            from_date,
            to_date,
            lambda txn: txn.is_short_term and filter_cb(txn)
        )


###############################################################################


@dataclass(kw_only=True)
class DatewiseLog(_BasePostInit):
    # Mapping: date -> txn_id -> txn. Dict is ordered.
    _txn_log: dict[Date, OrderedDict[str, _Transaction]] = dataclass_field(
        init=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        """Initialise for all dates."""
        super().__post_init__()

        # If we are in FY2024-2025, we need to have 1st Jan 2024 (cy_start) to
        # 31st Mar 2025 (fy_end). Since we need an opening balance, we will
        # just use 31st Dec 2023 to enforce an opening balance for 1st Jan.
        for date in date_range(prev_cy_end(), next_fy_start()):
            self._txn_log[date] = OrderedDict()

    def add_txn_to_log(self, txn: _Transaction) -> None:
        if PARSING_OPENING_LEDGER and txn.date not in self._txn_log:
            # In case of old acquisitons.
            self._txn_log[txn.date] = OrderedDict()

        # When making a transaction we ensure a unique ID, and thus the date
        # will always be the same for a given txn_id.
        if txn.txn_id in self._txn_log[txn.date]:
            raise ValueError(f"Attempt to add txn_id {txn.txn_id} again in "
                             "a datewise log")

        self._txn_log[txn.date][txn.txn_id] = txn

    def get_all_txns_on(self, date: Date) -> OrderedDict[str, _Transaction]:
        return self._txn_log[date]

    def get_all_txns(self) -> OrderedDict[str, _Transaction]:
        final_dict = {}

        for txn_dict in self._txn_log.values():
            final_dict |= txn_dict

        return final_dict


###############################################################################


@dataclass(kw_only=True)
class ShareLot(MapToEntity, DatewiseLog):
    buy_txn_id: str

    sellings: SellingTracker = dataclass_field(
        init=False, default_factory=SellingTracker
    )

    dividends: DividendTracker = dataclass_field(
        init=False, default_factory=DividendTracker
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        try:
            buy_txn = all_share_transactions[self.buy_txn_id]
        except KeyError:
            raise ValueError(f"Invalid buy transaction '{self.buy_txn_id}'.")

        if not isinstance(buy_txn, _BuyTransaction):
            raise ValueError(f"Specified lot buy txn ID '{self.buy_txn_id}' "
                             f"is not a buy txn.")

        self.add_txn_to_log(buy_txn)

    @property
    def buy_txn_obj(self) -> _BuyTransaction:
        return all_share_transactions[self.buy_txn_id]

    @property
    def remaining_units(self) -> Fraction:
        return self.buy_txn_obj.units - self.sellings.total_units

    @property
    def empty(self) -> bool:
        return self.remaining_units == 0

    def sell_units(self, sell_txn: SellTransaction) -> None:
        if sell_txn.units <= 0:
            raise ValueError("Malformed txn: Invalid number of units to sell "
                             f"({sell_txn.units}). Must be > 0.")

        if self.remaining_units < sell_txn.units:
            raise ValueError(f"{sell_txn.txn_id} sells {sell_txn.units} units "
                             f"but lot has only {self.remaining_units} units "
                             "remaining.")

        self.sellings.add_txn(sell_txn.txn_id)
        self.add_txn_to_log(sell_txn)

    def receive_dividend(self, dividend_txn: CashCreditTransaction) -> None:
        self.dividends.add_txn(dividend_txn.txn_id)
        self.add_txn_to_log(dividend_txn)

    def opening_units_on(self, date: Date) -> Fraction:
        """
        Units held at 00:00 of `date` (i.e. before that day’s trades).
        """
        if date <= self.buy_txn_obj.date:
            raise ValueError("The lot was not bought on any date before the "
                             f"given date {date}.")

        prev_date = date - timedelta(days=1)
        sold_till_prev = self.sellings.total_units_between_dates(None,
                                                                 prev_date)

        return self.buy_txn_obj.units - sold_till_prev

    def get_holding_values_on(
        self,
        date: Date,
    ) -> OrderedDict[str, tuple[Fraction, Fraction]]:
        """
        Get all possible values on a date, as transactions can cause a split.

        Returns dict: {
            "__opening":       TupType,  # Opening value or None. See below.
            "txn_id_1_before": TupType,  # Value before txn_id_1 executed.
            "txn_id_1_after":  TupType,  # Value after txn_id_1 executed.
            [...]
            "__closing":       TupType   # Closing value.
        }

        where TupType = tuple[Fraction, Fraction] is the tuple containing the
        remaining units and the value of those units. That is, it is the tuple
        (units_in_lot, value_of_units).

        We unfortunately can't have intraday, forget the other implications, we
        don't even have the requisite granular data. We have the high price,
        but we don't know when that high happened and how many units we have at
        that time unless we didn't sell anything.

        But when we sell, we know the price of the stock at that point in time,
        and hence we add the _before and _after keys to calculate the values.
        If the sell price was higher than both opening and peak, that becomes
        the peak value.

        If there was no transaction in the day and we had opening units, then
        only in this case we know the high price, which will be added in the
        "__highest_intraday" key.

        Since dict is ordered, the last value is the closing value.

        On the date the lot was bought with "buy_txn", the "__opening" and
        "buy_txn_before" keys will be zero, and they will be the first two
        keys. The third element with "buy_txn_after" will have the value at
        buy.

        Make sure to not use value at buy in the vesting case for calculation
        in INR for taxation purposes, since the merchant likely would have used
        a different TTBR for taxation (see docstring of VestingTransaction).
        """
        holding_values = OrderedDict()

        if date < self.buy_txn_obj.date:
            holding_values["__opening"] = (ZERO, ZERO)
            holding_values["__closing"] = (ZERO, ZERO)
            return holding_values

        if date == self.buy_txn_obj.date:
            units = ZERO
            holding_values["__opening"] = (units, ZERO)
        else:
            units = self.opening_units_on(date)
            opening_price = self.entity.get_share_price(
                date, "open", for_peak_value_reporting=True
            )
            holding_values["__opening"] = (units, units * opening_price)

        all_txns = self.get_all_txns_on(date)

        if not all_txns and units != 0:
            # No transaction happened. We can use the high value.
            high_price = self.entity.get_share_price(
                date, "high", for_peak_value_reporting=True
            )
            holding_values["__highest_intraday"] = (units, units * high_price)
        else:
            # Walk through this lot’s transactions for the given date.
            for idx, txn in enumerate(all_txns.values()):
                before_key = txn.txn_id + "_before"
                after_key = txn.txn_id + "_after"

                # Lot buy.
                if isinstance(txn, _BuyTransaction):
                    if idx != 0:
                        raise RuntimeError(
                            f"Encountered buy txn {txn.txn_id} on {date} at a "
                            f"non-zero index {idx} for lot having buy txn ID "
                            f"{self.buy_txn_id}."
                        )

                    if txn != self.buy_txn_obj:
                        raise RuntimeError(
                            f"Encountered buy txn {txn.txn_id} on a date "
                            f"{date} different from the lot's buy txn "
                            f"{self.buy_txn_obj.date}."
                        )

                    if (
                        holding_values["__opening"] != (ZERO, ZERO)
                        or units != ZERO
                    ):
                        raise RuntimeError(
                            f"Encountered buy txn {txn.txn_id} for non-zero "
                            f"lot opening value / units on date {date}."
                        )

                    holding_values[before_key] = (ZERO, ZERO)

                    units = txn.units
                    price_at_buy = txn.gross_total_value_native
                    holding_values[after_key] = (units, price_at_buy)

                # Only SellTransaction affects units in this lot.
                elif isinstance(txn, SellTransaction):
                    # When we sell something, we know it's at that price.
                    price_at_sell = txn.cost_per_unit
                    holding_values[before_key] = (units, units * price_at_sell)

                    units -= txn.units
                    holding_values[after_key] = (units, units * price_at_sell)
            # End of for loop.

        closing_price = self.entity.get_share_price(
            date, "close", for_peak_value_reporting=True
        )
        holding_values["__closing"] = (units, units * closing_price)

        return holding_values

    def get_peak_holding_value_on(self, date: Date) -> Fraction:
        holding_values = self.get_holding_values_on(date)
        return max(t[1] for t in holding_values.values())

    @property
    def peak_holding_value_native_in_calendar_year(self) -> Fraction:
        """
        We include the vesting txn for native value, so do NOT use this for the
        INR value for the purposes of taxation.
        """
        return self.country.get_peak_value_in_cy(
            self.get_peak_holding_value_on,
            in_inr=False,
        )

    @property
    def peak_holding_value_inr_in_calendar_year(self) -> Fraction:
        """Do not use this for taxation purposes. Only use for FA reporting."""
        return self.country.get_peak_value_in_cy(
            self.get_peak_holding_value_on,
            in_inr=True,
        )

    @property
    def closing_value_native_in_calendar_year(self) -> Fraction:
        holding_values = self.get_holding_values_on(cy_end())
        _, closing_value = holding_values["__closing"]
        return closing_value

    @property
    def closing_value_inr_in_calendar_year(self) -> Fraction:
        _, value = self.country.convert_to_inr_on(
            cy_end(),
            self.closing_value_native_in_calendar_year,
        )
        return value

    @property
    def total_dividends_native_in_calendar_year(self) -> Fraction:
        return self.dividends.gross_total_value_native_between_dates(
            cy_start(), cy_end()
        )

    @property
    def total_dividends_inr_in_calendar_year(self) -> Fraction:
        return self.country.convert_to_inr_for_FA_income(
            self.total_dividends_native_in_calendar_year
        )

    @property
    def total_sale_proceeds_native_in_calendar_year(self) -> Fraction:
        return self.sellings.gross_total_value_native_between_dates(
            cy_start(), cy_end()
        )

    @property
    def total_sale_proceeds_inr_in_calendar_year(self) -> Fraction:
        return self.country.convert_to_inr_for_FA_income(
            self.total_sale_proceeds_native_in_calendar_year
        )

    @property
    def total_dividends_inr_in_financial_year_for_tax(self) -> Fraction:
        return self.dividends.net_total_value_inr_for_tax_between_dates(
            fy_start(), fy_end()
        )

    @property
    def total_gain_inr_in_financial_year_for_tax(self) -> Fraction:
        return self.sellings.capital_gain_tax_inr_between_dates(
            fy_start(), fy_end()
        )

    @property
    def total_tax_withheld_on_dividends_inr_in_financial_year(
        self,
    ) -> Fraction:
        return self.dividends.total_tax_withheld_inr_between_dates(
            fy_start(), fy_end()
        )

    @property
    def total_tax_withheld_on_sell_inr_in_financial_year(self) -> Fraction:
        return self.sellings.total_tax_withheld_inr_between_dates(
            fy_start(), fy_end()
        )

    @property
    def total_ltcg_amount_inr_in_financial_year(self) -> Fraction:
        return self.sellings.total_ltcg_amount_inr_for_tax_between_dates(
            fy_start(), fy_end()
        )

    @property
    def total_stcg_amount_inr_in_financial_year(self) -> Fraction:
        return self.sellings.total_stcg_amount_inr_for_tax_between_dates(
            fy_start(), fy_end()
        )

    @property
    def total_ltcg_tax_inr_in_financial_year(self) -> Fraction:
        return self.sellings.total_ltcg_tax_inr_between_dates(
            fy_start(), fy_end()
        )

    @property
    def total_stcg_tax_inr_in_financial_year(self) -> Fraction:
        return self.sellings.total_stcg_tax_inr_between_dates(
            fy_start(), fy_end()
        )

    @property
    def total_ltcg_tax_withheld_inr_in_financial_year(self) -> Fraction:
        return self.sellings.total_ltcg_tax_withheld_inr_between_dates(
            fy_start(), fy_end()
        )

    @property
    def total_stcg_tax_withheld_inr_in_financial_year(self) -> Fraction:
        return self.sellings.total_stcg_tax_withheld_inr_between_dates(
            fy_start(), fy_end()
        )

    @property
    def ltcg_amount_inr_for_which_tax_was_withheld_in_financial_year(
        self
    ) -> Fraction:
        return self.sellings.ltcg_amount_inr_for_which_tax_was_withheld_between_dates(  # noqa: E501
            fy_start(), fy_end()
        )

    @property
    def stcg_amount_inr_for_which_tax_was_withheld_in_financial_year(
        self
    ) -> Fraction:
        return self.sellings.stcg_amount_inr_for_which_tax_was_withheld_between_dates(  # noqa: E501
            fy_start(), fy_end()
        )

    @property
    def dividend_amount_inr_for_which_tax_was_withheld_in_financial_year(
        self
    ) -> Fraction:
        return self.dividends.net_total_amount_inr_for_which_tax_was_withheld_between_dates(  # noqa: E501
            fy_start(), fy_end()
        )

    @property
    def total_buy_sell_ltcg_tuple_inr_in_financial_year(
        self,
    ) -> tuple[Fraction, Fraction, Fraction]:
        return self.sellings.total_buy_sell_ltcg_tuple_inr_for_tax_between_dates(  # noqa: E501
            fy_start(), fy_end()
        )

    @property
    def total_buy_sell_stcg_tuple_inr_in_financial_year(
        self,
    ) -> tuple[Fraction, Fraction, Fraction]:
        return self.sellings.total_buy_sell_stcg_tuple_inr_for_tax_between_dates(  # noqa: E501
            fy_start(), fy_end()
        )

    def total_ltcg_amount_inr_for_advance_tax_installment(
        self,
        installment: int,
    ) -> Fraction:
        func_name = f"date_in_fy_advance_tax_installment_{installment}"
        date_in_installment = globals()[func_name]

        return self.sellings.total_ltcg_amount_inr_for_tax_between_dates(
            fy_start(),
            fy_end(),
            lambda txn: txn.is_gain and date_in_installment(txn.date)
        )

    def total_stcg_amount_inr_for_advance_tax_installment(
        self,
        installment: int,
    ) -> Fraction:
        func_name = f"date_in_fy_advance_tax_installment_{installment}"
        date_in_installment = globals()[func_name]

        return self.sellings.total_stcg_amount_inr_for_tax_between_dates(
            fy_start(),
            fy_end(),
            lambda txn: txn.is_gain and date_in_installment(txn.date)
        )

    def total_dividend_amount_inr_for_advance_tax_installment(
        self,
        installment: int,
    ) -> Fraction:
        func_name = f"date_in_fy_advance_tax_installment_{installment}"
        date_in_installment = globals()[func_name]

        return self.dividends.net_total_value_inr_for_tax_between_dates(
            fy_start(), fy_end(), lambda txn: date_in_installment(txn.date)
        )


###############################################################################


@dataclass(kw_only=True)
class CashWallet(MapToEntity, DatewiseLog):
    open_date: Date
    close_date: Date = dataclass_field(init=False, default=None)

    credits: CashCreditTracker = dataclass_field(
        init=False, default_factory=CashCreditTracker
    )

    debits: CashDebitTracker = dataclass_field(
        init=False, default_factory=CashDebitTracker
    )

    # We credit when receiving dividend, so this will have transactions already
    # in credits.
    dividends: DividendTracker = dataclass_field(
        init=False, default_factory=DividendTracker
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        if not self.entity.cash_type:
            raise ValueError(
                f"Attempted init of cash _wallet with non-cash entity "
                f"'{self.entity_id}'."
            )

    @property
    def closed(self) -> bool:
        return self.close_date is not None

    def _ensure_not_closed(self) -> None:
        if self.closed:
            raise ValueError("Attempted to use a closed cash wallet (entity "
                             f"{self.entity_id}). Closed on {self.close_date}")

    @property
    def balance(self) -> Fraction:
        # We must use net, as the tax withheld never landed in the wallet.
        net_additions = self.credits.net_total_value_native

        # Withdrawals are gross. They don't have any withholding, and even if
        # they did, the tax withheld would be cut after selling.
        withdrawals = self.debits.gross_total_value_native

        return net_additions - withdrawals

    def credit(
        self,
        *,
        txn_id: str,
        date: Date,
        amount: Fraction,
        misc_fees: Fraction,
        tax_withholding_dict: dict[str, Fraction],
    ) -> CashCreditTransaction:
        self._ensure_not_closed()

        txn = CashCreditTransaction(
            txn_id=txn_id,
            date=date,
            entity_id=self.entity_id,
            amount=amount,
            misc_fees=misc_fees,
            tax_withholding_rate=(tax_withholding_dict["rate_percent"] / 100),
            tax_withholding_amount_native=tax_withholding_dict["amount"],
        )

        self.credits.add_txn(txn.txn_id)
        self.add_txn_to_log(txn)

        return txn

    def debit(
        self,
        *,
        txn_id: str,
        date: Date,
        amount: Fraction,
        ghar_vaapsi: bool = False,
    ) -> CashDebitTransaction:
        self._ensure_not_closed()

        if amount > self.balance:
            raise ValueError("Debiting amount > balance in the cash wallet "
                             f"(entity {self.entity_id}).")

        txn = CashDebitTransaction(
            txn_id=txn_id,
            date=date,
            entity_id=self.entity_id,
            amount=amount,
            ghar_vaapsi=ghar_vaapsi,
        )

        self.debits.add_txn(txn.txn_id)
        self.add_txn_to_log(txn)

        return txn

    def receive_dividend_from_self_into_self(
        self,
        **kwargs
    ) -> CashCreditTransaction:
        self._ensure_not_closed()

        dividend_txn = self.credit(**kwargs)
        self.dividends.add_txn(dividend_txn.txn_id)

        return dividend_txn

    def close(self, date: Date) -> None:
        self._ensure_not_closed()

        if self.balance != 0:
            raise ValueError("Attempted to close a cash wallet (entity "
                             f"{self.entity_id}) with remaining balance.")

        self.close_date = date

    def opening_balance_on(self, date: Date) -> Fraction:
        """
        Balance held at 00:00 of `date` (i.e. before that day’s trades).
        """
        prev_date = date - timedelta(days=1)

        # See the balance() function.
        net_additions = self.credits.net_total_value_native_between_dates(
            None, prev_date
        )
        withdrawals = self.debits.gross_total_value_native_between_dates(
            None, prev_date
        )

        return net_additions - withdrawals

    def get_balances_on(self, date: Date) -> OrderedDict[str, Fraction]:
        """
        Get all possible balances on a date, as transactions can cause a split.

        Returns dict: {
            "__opening": Fraction,  # Opening balance.
            "txn_id_1":  Fraction,  # Balance after txn_id_1 executed.
            [...]
            "__closing": Fraction,  # Closing balance.
        }

        Since dict is ordered, the last value is the closing value.
        """
        balances_in_day = OrderedDict()
        balance = self.opening_balance_on(date)

        balances_in_day["__opening"] = balance

        for txn in self.get_all_txns_on(date).values():
            if isinstance(txn, CashCreditTransaction):
                balance += txn.net_total_value_native

            elif isinstance(txn, CashDebitTransaction):
                balance -= txn.gross_total_value_native

            else:
                raise ValueError(f"Unknown txn type {type(txn)} in cash "
                                 f"wallet (entity {self.entity_id})")

            balances_in_day[txn.txn_id] = balance

        balances_in_day["__closing"] = balance

        return balances_in_day

    def _get_wallet_date_range(
        self,
        min_start: Date,
        max_end_exclusive: Date
    ) -> tuple[Date, Date]:
        """Returns (start, end_exclusive) to pass to date_range()."""
        start = max(min_start, self.open_date)

        if not self.close_date:
            end_exclusive = max_end_exclusive
        else:
            date_after_close_date = self.close_date + timedelta(days=1)
            end_exclusive = min(max_end_exclusive, date_after_close_date)

        return start, end_exclusive

    def get_peak_balance_on(self, date: Date) -> Fraction:
        balances = self.get_balances_on(date)
        return max(balances.values())

    @property
    def peak_balance_native_in_calendar_year(self) -> Fraction:
        start, end = self._get_wallet_date_range(cy_start(), next_cy_start())
        return self.country.get_peak_value_in_cy(
            self.get_peak_balance_on,
            date_start=start,
            date_end_exclusive=end,
            in_inr=False,
        )

    @property
    def peak_balance_inr_in_calendar_year(self) -> Fraction:
        start, end = self._get_wallet_date_range(cy_start(), next_cy_start())
        return self.country.get_peak_value_in_cy(
            self.get_peak_balance_on,
            date_start=start,
            date_end_exclusive=end,
            in_inr=True,
        )

    @property
    def total_dividends_native_in_calendar_year(self) -> Fraction:
        return self.dividends.gross_total_value_native_between_dates(
            cy_start(), cy_end()
        )

    @property
    def total_dividends_inr_in_calendar_year(self) -> Fraction:
        return self.country.convert_to_inr_for_FA_income(
            self.total_dividends_native_in_calendar_year
        )

    @property
    def total_ghar_vaapsi_native_in_calendar_year(self) -> Fraction:
        return self.debits.total_ghar_vaapsi_native_between_dates(
            cy_start(), cy_end()
        )

    @property
    def total_ghar_vaapsi_inr_in_calendar_year(self) -> Fraction:
        return self.country.convert_to_inr_for_FA_income(
            self.total_ghar_vaapsi_native_in_calendar_year
        )

    @property
    def total_dividends_inr_in_financial_year_for_tax(self) -> Fraction:
        return self.dividends.net_total_value_inr_for_tax_between_dates(
            fy_start(), fy_end()
        )

    @property
    def total_tax_withheld_on_dividends_inr_in_financial_year(
        self,
    ) -> Fraction:
        return self.dividends.total_tax_withheld_inr_between_dates(
            fy_start(), fy_end()
        )

    @property
    def dividend_amount_inr_for_which_tax_was_withheld_in_financial_year(
        self
    ) -> Fraction:
        return self.dividends.net_total_amount_inr_for_which_tax_was_withheld_between_dates(  # noqa: E501
            fy_start(), fy_end()
        )

    def total_dividend_amount_inr_for_advance_tax_installment(
        self,
        installment: int,
    ) -> Fraction:
        func_name = f"date_in_fy_advance_tax_installment_{installment}"
        date_in_installment = globals()[func_name]

        return self.dividends.net_total_value_inr_for_tax_between_dates(
            fy_start(), fy_end(), lambda txn: date_in_installment(txn.date)
        )


###############################################################################


# broker_id -> Broker
brokers = {}


@dataclass(kw_only=True)
class Broker(MapToCountry, DatewiseLog):
    broker_id: str
    name: str
    address_without_zipcode: str
    address_zipcode: str
    account_number: str
    account_opening_date: Date
    i_am_legal_owner: bool = False
    i_am_beneficial_owner: bool = False
    i_am_beneficiary: bool = False

    _wallet: CashWallet = dataclass_field(init=False, default=None)
    _prev_wallets: list[CashWallet] = dataclass_field(init=False,
                                                      default_factory=list)
    # entity_id -> lot buy txn id -> Lot. Dict is ordered, so just use values()
    # on the inner dict to get the list of Lots for FIFO.
    _lots: dict[str, dict[str, ShareLot]] = dataclass_field(
        init=False, default_factory=dict
    )

    @property
    def a2_status(self) -> str:
        status_bools = (self.i_am_legal_owner, self.i_am_beneficial_owner,
                        self.i_am_beneficiary)

        if len(set(status_bools)) == 1:
            raise ValueError("All i_am_* status bools have the same value for "
                             f"broker {self.broker_id}.")

        if status_bools.count(True) != 1:
            raise ValueError("More than one i_am_* status bool set to True "
                             f"for broker {self.broker_id}.")

        if self.i_am_legal_owner:
            return "OWNER"

        if self.i_am_beneficial_owner:
            return "BENEFICIAL_OWNER"

        if self.i_am_beneficiary:
            return "BENIFICIARY"

        raise RuntimeError("Code error - we should not be here!")

    def __post_init__(self) -> None:
        global brokers

        super().__post_init__()
        ensure_unique_yaml_key(self.broker_id)

        # We check in the function so this serves as verification.
        _ = self.a2_status

        brokers[self.broker_id] = self

    def _ensure_wallet_init(self) -> None:
        if self._wallet is None:
            raise ValueError("Attempted transaction / use of cash wallet with "
                             f"an uninitialized wallet for {self.broker_id}.")

    @property
    def _all_wallets(self) -> list[CashWallet]:
        self._ensure_wallet_init()
        return self._prev_wallets + [self._wallet]

    def add_cash(self, **kwargs) -> None:
        self._ensure_wallet_init()

        if "tax_withholding_dict" in kwargs:
            raise ValueError("Tax withholding dict given when manually adding "
                             "cash.")
        kwargs["tax_withholding_dict"] = dict(rate_percent=ZERO, amount=ZERO)

        txn = self._wallet.credit(**kwargs)
        self.add_txn_to_log(txn)

    def withdraw_cash(self, **kwargs) -> None:
        self._ensure_wallet_init()
        txn = self._wallet.debit(**kwargs)
        self.add_txn_to_log(txn)

    def cash_init(
        self,
        *,
        txn_id: str,
        entity_id: str,
        amount: Fraction,
    ) -> None:
        if self._wallet is not None:
            raise ValueError(f"Attempted reinit of cash fund with {entity_id} "
                             f"in {txn_id}. In use: {self._wallet.entity_id}.")

        # We use 31st December to have opening balance.
        self._wallet = CashWallet(entity_id=entity_id, open_date=prev_cy_end())

        if amount > 0:
            self.add_cash(date=prev_cy_end(), txn_id=txn_id, amount=amount,
                          misc_fees=0)

    def cash_fund_switch(
        self,
        *,
        txn_id: str,
        date: Date,
        new_entity_id: str,
        misc_fees: Fraction,
    ) -> None:
        self._ensure_wallet_init()
        balance = self._wallet.balance

        self.withdraw_cash(txn_id=f"{txn_id}//WITHDRAW_FOR_SWITCH",
                           date=date, amount=balance)
        self._wallet.close(date)
        self._prev_wallets.append(self._wallet)

        self._wallet = CashWallet(entity_id=new_entity_id, open_date=date)
        self.add_cash(txn_id=f"{txn_id}//ADD_FOR_SWITCH", date=date,
                      amount=balance, misc_fees=misc_fees)

    def add_lot_buy(
        self, *,
        txn_id: str,
        date: Date,
        entity_id: str,
        units: Fraction,
        cost_per_unit: Fraction,
        merchant_ttbr_if_vest: Fraction = None,
        for_opening_lot: bool = False,
    ) -> None:
        self._ensure_wallet_init()

        # The class ctors does all checks so we don't need to do here.
        buy_txn_args = dict(
            date=date, entity_id=entity_id, txn_id=txn_id, units=units,
            cost_per_unit=cost_per_unit
        )

        if merchant_ttbr_if_vest is None:
            if not for_opening_lot:
                cash_txn_id = txn_id + "//USE_CASH_FOR_BUY"
                self.withdraw_cash(txn_id=cash_txn_id, date=date,
                                   amount=(units * cost_per_unit))
            buy_txn = ManualBuyTransaction(**buy_txn_args)
        else:
            buy_txn = VestingTransaction(**buy_txn_args,
                                         merchant_ttbr=merchant_ttbr_if_vest)

        self.add_txn_to_log(buy_txn)

        lot = ShareLot(entity_id=entity_id, buy_txn_id=txn_id)

        if entity_id not in self._lots:
            self._lots[entity_id] = {}
        elif txn_id in self._lots[entity_id]:
            raise ValueError(f"Lot with {txn_id} already exists for broker "
                             f"{self.broker_id}.")

        self._lots[entity_id][txn_id] = lot

    def _total_units_in_all_lots(self, entity_id: str) -> Fraction:
        total_units = ZERO

        for lot in self._lots[entity_id].values():
            if lot.remaining_units < 0:
                raise RuntimeError("Encountered lot with negative remaining "
                                   f"units for broker {self.broker_id}, "
                                   f"entity {lot.entity_id}.")

            total_units += lot.remaining_units

        return total_units

    def sell_shares_fifo(
        self,
        *,
        txn_id: str,
        date: Date,
        entity_id: str,
        units: Fraction,
        cost_per_unit: Fraction,
        misc_fees: Fraction,
        tax_withholding_dict: dict[str, Fraction]
    ) -> None:
        self._ensure_wallet_init()

        if units <= 0:
            raise ValueError(f"{txn_id} attempts to sell units <= 0 for "
                             f"entity {entity_id}.")

        existing_units = self._total_units_in_all_lots(entity_id)
        if existing_units < units:
            raise ValueError(f"{txn_id} attempts to sell units more than the "
                             f"existing units for entity {entity_id} across "
                             "all lots.")

        overall_total_amount = units * cost_per_unit
        overall_net_amount = \
            overall_total_amount - tax_withholding_dict["amount"] - misc_fees

        tax_withheld_rate = tax_withholding_dict["rate_percent"] / 100

        sell_txn_common_args = dict(
            date=date,
            entity_id=entity_id,
            cost_per_unit=cost_per_unit,
            tax_withholding_rate=tax_withheld_rate,
        )

        all_lots = [lot for lot in self._lots[entity_id].values()
                    if lot.remaining_units != 0]

        # Divide the fees paid among each sell transaction.
        misc_fees_per_lot = misc_fees / len(all_lots)

        # Tax withholding cannot be calculated earlier, so we will calculate
        # later and sum it up in this variable.
        tax_withheld_in_loop = ZERO

        # The sell txn created in the loop. After the loop ends, this will have
        # the last sell txn done.
        sell_txn = None

        for lot in all_lots:
            units_to_sell = min(units, lot.remaining_units)

            lot_sell_txn_id = txn_id + f"//FIFO_SELL_LOT_{lot.buy_txn_id}"

            # We cannot calculate tax withholding without knowing the gain,
            # since the tax is on the gain and not on the total amount. So we
            # will pass the gain attribute name so tax withholding could be
            # calculated automatically.
            gain_attr = "gain_amount_native"

            sell_txn = SellTransaction(
                **sell_txn_common_args,
                txn_id=lot_sell_txn_id,
                units=units_to_sell,
                buy_txn_id=lot.buy_txn_id,
                misc_fees=misc_fees_per_lot,
                tax_withholding_amount_native=0,
                calculate_tax_withholding_amount_from_attribute=gain_attr,
            )

            lot.sell_units(sell_txn)
            self.add_txn_to_log(sell_txn)

            tax_withheld_in_loop += sell_txn.tax_withholding_amount_native
            units -= units_to_sell

            if units == 0:
                break

        if units != 0:
            raise RuntimeError("Code didn't sell all units, even though it "
                               "checked for sufficiency earlier.")

        # Due to rounding by broker, we may have a difference between the given
        # and calculated tax withholding. Let's ensure it's not more than the
        # typical lowest precision.
        calc_difference = tax_withholding_dict["amount"] - tax_withheld_in_loop
        if calc_difference >= Fraction("0.01"):
            raise ValueError("Calculated tax withheld when doing FIFO selling "
                             f"for {txn_id} doesn't match given tax withheld "
                             "(margin >= 0.01).")

        # We have the last loop variable. Let's add the difference amount in
        # the last sell so that any broker rounding is accounted for.
        if sell_txn is not None:
            sell_txn.tax_withholding_amount_native += calc_difference

        cash_txn_id = txn_id + "//GET_CASH_AFTER_SELL"
        self.add_cash(txn_id=cash_txn_id, date=date, amount=overall_net_amount,
                      misc_fees=0)

    def sell_shares_specific(
        self,
        *,
        txn_id: str,
        buy_txn_id: str,
        date: Date,
        entity_id: str,
        units: Fraction,
        cost_per_unit: Fraction,
        misc_fees: Fraction,
        tax_withholding_dict: dict[str, Fraction],
        amount_deducted_on_sell_to_cover: Fraction,
    ) -> None:
        self._ensure_wallet_init()

        if units <= 0:
            raise ValueError(f"{txn_id} attempts to sell units <= 0 for "
                             f"entity {entity_id}.")

        if buy_txn_id not in self._lots[entity_id]:
            raise ValueError(f"{txn_id} attempts to sell units for an unknown "
                             f"lot with buy ID {buy_txn_id} and entity ID "
                             f"{entity_id} for broker {self.broker_id}.")

        lot = self._lots[entity_id][buy_txn_id]

        if lot.buy_txn_obj.entity_id != entity_id:
            raise ValueError(f"{txn_id} attempts to sell units of entity "
                             f"{entity_id} from a specific lot with buy ID "
                             f"{buy_txn_id}, but the latter is of a different "
                             f"entity {lot.buy_txn_obj.entity_id}.")

        if lot.remaining_units < units:
            raise ValueError(f"{txn_id} attempts to sell units more than the "
                             f"existing units for entity {entity_id} and the "
                             f"specified lot with buy ID {lot.buy_txn_id}.")

        sell_txn = SellTransaction(
            date=date,
            txn_id=txn_id,
            entity_id=entity_id,
            units=units,
            cost_per_unit=cost_per_unit,
            buy_txn_id=lot.buy_txn_id,
            misc_fees=misc_fees,
            tax_withholding_rate=(tax_withholding_dict["rate_percent"] / 100),
            tax_withholding_amount_native=tax_withholding_dict["amount"],
        )

        lot.sell_units(sell_txn)
        self.add_txn_to_log(sell_txn)

        net_amount = sell_txn.net_total_value_native
        cash_txn_id = txn_id + "//GET_CASH_AFTER_SELL"

        if amount_deducted_on_sell_to_cover != ZERO:
            net_amount -= amount_deducted_on_sell_to_cover

        self.add_cash(txn_id=cash_txn_id, date=date, amount=net_amount,
                      misc_fees=0)

    def receive_stock_dividend(
        self,
        *,
        txn_id: str,
        date: Date,
        entity_id: str,
        amount: Fraction,
        misc_fees: Fraction,
        tax_withholding_dict: dict[str, Fraction],
    ) -> None:
        self._ensure_wallet_init()
        total_units = self._total_units_in_all_lots(entity_id)

        if total_units == 0:
            raise ValueError(f"Cannot distribute dividend {amount} for entity "
                             f"{entity_id} with zero total units in broker "
                             f"{self.broker_id}.")

        dividend_per_unit = amount / total_units

        total_tax_withheld = tax_withholding_dict["amount"]
        tax_withheld_per_unit = total_tax_withheld / total_units

        all_lots = [lot for lot in self._lots[entity_id].values()
                    if lot.remaining_units > 0]

        last_lot_index = len(all_lots) - 1

        dividend_disbursed_in_lots = ZERO
        tax_withholding_disbursed_in_lots = ZERO
        misc_fees_disbursed_in_lots = ZERO

        # Divide the fees paid among each lot.
        misc_fees_per_each_lot = misc_fees / len(all_lots)

        for index, lot in enumerate(all_lots):
            dividend_for_lot = dividend_per_unit * lot.remaining_units
            tax_withheld_for_lot = tax_withheld_per_unit * lot.remaining_units
            misc_fees_for_lot = misc_fees_per_each_lot

            dividend_disbursed_in_lots += dividend_for_lot
            tax_withholding_disbursed_in_lots += tax_withheld_for_lot
            misc_fees_disbursed_in_lots += misc_fees_for_lot

            # We can be dividing into all lots, but may have some very small
            # difference between disbursed total and broker's value due to the
            # rounding drift. So we add the remaining amount in the last lot to
            # tally the calculation.
            if index == last_lot_index:
                remaining_dividend = amount - dividend_disbursed_in_lots
                remaining_tax_withheld = \
                    total_tax_withheld - tax_withholding_disbursed_in_lots
                remaining_fees = misc_fees - misc_fees_disbursed_in_lots

                if remaining_dividend < 0:
                    raise ValueError(
                        "Disbursed more dividend than actual "
                        f"({dividend_disbursed_in_lots} > {amount}) "
                        f"for entity {entity_id} in broker {self.broker_id}."
                    )

                if remaining_dividend >= Fraction("0.01"):
                    raise ValueError(
                        "Remaining dividend is significant (>= 0.01) "
                        f"({amount} - {dividend_disbursed_in_lots}) "
                        f"for entity {entity_id} in broker {self.broker_id}."
                    )

                if remaining_tax_withheld < 0:
                    raise ValueError(
                        "Withheld more tax on dividend than actual "
                        f"({tax_withholding_disbursed_in_lots} > "
                        f"{total_tax_withheld}) for entity {entity_id} in "
                        "broker {self.broker_id}."
                    )

                if remaining_tax_withheld >= Fraction("0.01"):
                    raise ValueError(
                        "Remaining tax to withhold on dividend is significant "
                        f"(>= 0.01) ({total_tax_withheld} - "
                        f"{tax_withholding_disbursed_in_lots}) for entity "
                        f"{entity_id} in broker {self.broker_id}."
                    )

                if remaining_fees < 0:
                    raise ValueError(
                        "Disbursed more misc fees than actual "
                        f"({misc_fees_disbursed_in_lots} > {misc_fees}) "
                        f"for entity {entity_id} in broker {self.broker_id}."
                    )

                if remaining_fees >= Fraction("0.01"):
                    raise ValueError(
                        "Remaining misc fees is significant (>= 0.01) "
                        f"({misc_fees} - {misc_fees_disbursed_in_lots}) "
                        f"for entity {entity_id} in broker {self.broker_id}."
                    )

                dividend_for_lot += remaining_dividend
                tax_withheld_for_lot += remaining_tax_withheld
                misc_fees_for_lot += remaining_fees

                dividend_disbursed_in_lots += remaining_dividend
                tax_withholding_disbursed_in_lots += remaining_tax_withheld
                misc_fees_disbursed_in_lots += remaining_fees

            tax_withholding_dict_for_lot = deepcopy(tax_withholding_dict)
            tax_withholding_dict_for_lot["amount"] = tax_withheld_for_lot

            lot_div_txn_id = txn_id + f"//DIVIDEND_DIVIDE_LOT_{lot.buy_txn_id}"

            dividend_txn = self._wallet.credit(
                txn_id=lot_div_txn_id,
                date=date,
                amount=dividend_for_lot,
                misc_fees=misc_fees_for_lot,
                tax_withholding_dict=tax_withholding_dict_for_lot,
            )

            lot.receive_dividend(dividend_txn)
            self.add_txn_to_log(dividend_txn)
        # End of for loop.

        if dividend_disbursed_in_lots != amount:
            raise RuntimeError(
                f"Dividend amount {dividend_disbursed_in_lots} not equal to "
                f"the given amount {amount} after distribution among lots "
                f"for entity {entity_id} in broker {self.broker_id}."
            )

        if tax_withholding_disbursed_in_lots != total_tax_withheld:
            calc = tax_withholding_disbursed_in_lots
            raise RuntimeError(
                f"Tax withheld on dividend amount {calc} not equal to the "
                f"given amount {total_tax_withheld} after distribution among "
                f"lots for entity {entity_id} in broker {self.broker_id}."
            )

        if misc_fees_disbursed_in_lots != misc_fees:
            raise RuntimeError(
                f"Misc fees deducted {misc_fees_disbursed_in_lots} not equal "
                f"to the given amount {misc_fees} after distribution among "
                f"lots for entity {entity_id} in broker {self.broker_id}."
            )

    def receive_cash_dividend(self, **kwargs) -> None:
        """Cash has no lots so we can just add blindly."""
        self._ensure_wallet_init()
        dividend_txn = self._wallet.receive_dividend_from_self_into_self(
            **kwargs
        )
        self.add_txn_to_log(dividend_txn)

    def get_cash_balances_on(self, date: Date) -> Fraction:
        """
        Get all possible cash balances in account as on a date, as transactions
        can cause a split.

        See CashWallet.get_balances_on() for return value.
        """
        self._ensure_wallet_init()
        applicable_wallet = None

        for wallet in self._all_wallets:
            if wallet.open_date > date:
                continue

            if wallet.closed and date > wallet.close_date:
                continue

            if applicable_wallet is not None:
                raise RuntimeError(
                    f"Found two applicable wallets for same date {date}: "
                    f"{applicable_wallet.entity_id}, {wallet.entity_id}"
                )

            applicable_wallet = wallet

        if applicable_wallet is None:
            raise ValueError(f"Cannot find applicable wallet for date {date}.")

        return applicable_wallet.get_balances_on(date)

    def get_specific_entity_total_holding_values_on(
        self,
        date: Date,
        entity_id: str,
    ) -> OrderedDict[str, Fraction]:
        """
        Same as get_holding_values_on(), except only the value is returned and
        not the number of units.

        You MUST read docstring for get_holding_values_on(). Same warnings and
        considerations apply to the use of this function.
        """
        all_lot_holding_values = []

        for loop_entity_id, lot_map in self._lots.items():
            if loop_entity_id == entity_id:
                for lot in lot_map.values():
                    all_lot_holding_values.append(
                        lot.get_holding_values_on(date)
                    )

        if not all_lot_holding_values:
            raise ValueError(f"No lots found for entity {entity_id}")

        units = ZERO
        closing_units = ZERO

        entity_holding_values = OrderedDict()
        entity_holding_values["__opening"] = ZERO
        entity_holding_values["__closing"] = ZERO

        if not all_lot_holding_values:  # Nothing held.
            return entity_holding_values

        for hv in all_lot_holding_values:
            hv_opening_units, hv_opening_value = hv["__opening"]
            hv_closing_units, hv_closing_value = hv["__closing"]

            units += hv_opening_units
            closing_units += hv_closing_units

            entity_holding_values["__opening"] += hv_opening_value
            entity_holding_values["__closing"] += hv_closing_value

        # If no transaction happened for the equity on this date, then we can
        # calculate the "__highest_intraday" value by simply summing it up.
        if all(
            list(hv.keys()) == ["__opening", "__highest_intraday", "__closing"]
            for hv in all_lot_holding_values
        ):
            entity_holding_values["__highest_intraday"] = sum(
                hv["__highest_intraday"][1] for hv in all_lot_holding_values
            )
            entity_holding_values.move_to_end("__closing")
            return entity_holding_values

        # Since we are here, that means we have at least one share transaction
        # happened. Thus, the highest value is of no use as we don't know when
        # it happened.
        #
        # But since have a transaction happening, we can calculate the value at
        # that moment from the share price at transaction. It might be high or
        # low, but it is something, and if high it can be the peak value we can
        # know of.

        for txn_id, txn in self.get_all_txns_on(date).items():
            if txn.entity_id != entity_id:
                continue

            if not isinstance(txn, _ShareTransaction):
                raise RuntimeError("Encountered a non-share txn for entity "
                                   f"{entity_id} which has a lot in broker "
                                   f"{self.broker_id}.")

            new_share_price = txn.cost_per_unit
            before_key = txn.txn_id + "_before"
            after_key = txn.txn_id + "_after"

            entity_holding_values[before_key] = units * new_share_price

            if txn.buy:
                units += txn.units
            else:
                units -= txn.units

            entity_holding_values[after_key] = units * new_share_price
        # End of for loop.

        if units != closing_units:  # Sanity check.
            raise RuntimeError("After iteration, units != closing_units "
                               f"({units} != {closing_units}) for entity "
                               f"{entity_id}, broker {self.broker_id}.")

        entity_holding_values.move_to_end("__closing")
        return entity_holding_values

    def get_total_stock_holding_values_on(
        self,
        date: Date,
    ) -> OrderedDict[str, Fraction]:
        """
        Same as get_specific_entity_total_holding_values_on(), but sum total of
        all the shares / stocks we have in our portfolio.

        You MUST read get_specific_entity_total_holding_values_on() docstring.
        Same warnings and considerations apply to the use of this function.
        """
        all_entity_holding_values = {}

        for entity_id in self._lots.keys():
            all_entity_holding_values[entity_id] = (
                self.get_specific_entity_total_holding_values_on(
                    date, entity_id
                )
            )

        combined_holding_values = OrderedDict()
        combined_holding_values["__opening"] = ZERO
        combined_holding_values["__closing"] = ZERO

        if not all_entity_holding_values:  # Nothing held.
            return combined_holding_values

        for hv in all_entity_holding_values.values():
            combined_holding_values["__opening"] += hv["__opening"]
            combined_holding_values["__closing"] += hv["__closing"]

        # If no transaction happened for ANY equity on this date, then we can
        # calculate the "__highest_intraday" value by simply summing it up.
        if all(
            list(hv.keys()) == ["__opening", "__highest_intraday", "__closing"]
            for hv in all_entity_holding_values.values()
        ):
            combined_holding_values["__highest_intraday"] = sum(
                hv["__highest_intraday"]
                for hv in all_entity_holding_values.values()
            )
            combined_holding_values.move_to_end("__closing")
            return combined_holding_values

        # Since we are here, that means we have at least one share transaction
        # happened. Thus, the highest value is of no use as we don't know when
        # it happened.
        #
        # But since have a transaction happening, we can calculate the value at
        # before and after that transaction. The share price for a transaction
        # might be high or low for that specific entity, but it is something,
        # and if high adding it can result in the peak value we can know of.

        entity_val = {
            entity_id: hv["__opening"]
            for entity_id, hv in all_entity_holding_values.items()
        }

        for txn_id, txn in self.get_all_txns_on(date).items():
            entity_id = txn.entity_id
            before_key = txn.txn_id + "_before"
            after_key = txn.txn_id + "_after"

            for hv_eid, hv in all_entity_holding_values.items():
                if hv_eid != entity_id:
                    continue

                if not (before_key in hv or after_key in hv):
                    continue

                if before_key not in hv and after_key in hv:
                    raise RuntimeError(
                        f"{before_key} not present but {after_key} present "
                        f"for entity {entity_id}, broker {self.broker_id}."
                    )

                if before_key in hv and after_key not in hv:
                    raise RuntimeError(
                        f"{before_key} present but {after_key} not present "
                        f"for entity {entity_id}, broker {self.broker_id}."
                    )

                entity_val[entity_id] = hv[before_key]
                combined_holding_values[before_key] = sum(entity_val.values())

                entity_val[entity_id] = hv[after_key]
                combined_holding_values[after_key] = sum(entity_val.values())
            # End of inner for-loop.
        # End of outer for-loop.

        combined_holding_values.move_to_end("__closing")
        return combined_holding_values

    def get_portfolio_values_on(
        self,
        date: Date,
    ) -> OrderedDict[str, Fraction]:
        """
        Combined portfolio value -> cash + stock values.

        Format same as the dict format -> a combination of what the functions
        get_cash_balances_on() and get_total_stock_holding_values_on() return.

        You MUST read docstring for get_total_stock_holding_values_on(). Same
        warnings and considerations apply to the use of this function.
        """
        cash_balances = self.get_cash_balances_on(date)
        stock_values = self.get_total_stock_holding_values_on(date)

        total_portfolio_values = OrderedDict()
        total_portfolio_values["__opening"] = (
            cash_balances["__opening"] + stock_values["__opening"]
        )
        total_portfolio_values["__closing"] = (
            cash_balances["__closing"] + stock_values["__closing"]
        )

        # If no transaction happened for ANY cash or stock on this date, then
        # we can calculate the "__highest_intraday" value by simply summing it
        # up.
        if (
            list(cash_balances.keys()) == ["__opening", "__closing"]
            and list(stock_values.keys()) == [
                "__opening", "__highest_intraday", "__closing"
            ]
        ):
            total_portfolio_values["__highest_intraday"] = (
                cash_balances["__opening"] + stock_values["__highest_intraday"]
            )
            total_portfolio_values.move_to_end("__closing")
            return total_portfolio_values

        # Since we are here, that means least one cash or share transaction
        # happened. Thus, the highest value for equity is of no use as we don't
        # know when it happened.
        #
        # But since have a transaction happening, we can calculate the value at
        # before and after that transaction. The share price for a transaction
        # might be high or low for that specific entity, but it is something,
        # and if high adding it can result in the peak value we can know of.

        cash_val = cash_balances["__opening"]
        stock_val = stock_values["__opening"]

        for txn_id, txn in self.get_all_txns_on(date).items():
            if isinstance(txn, _CashTransaction):
                cash_val = cash_balances[txn_id]
                total_portfolio_values[txn_id] = cash_val + stock_val
                continue

            # Else we have stock val, which must have before and after values.
            before_key = txn.txn_id + "_before"
            after_key = txn.txn_id + "_after"

            if not (before_key in stock_values and after_key in stock_values):
                raise RuntimeError(
                    f"Either {before_key} or {after_key} not present in the "
                    f"combined stock_values dict for broker {self.broker_id}."
                )

            stock_val = stock_values[before_key]
            total_portfolio_values[before_key] = cash_val + stock_val

            stock_val = stock_values[after_key]
            total_portfolio_values[after_key] = cash_val + stock_val
        # End of for loop.

        total_portfolio_values.move_to_end("__closing")
        return total_portfolio_values

    def get_peak_portfolio_value_on(self, date: Date) -> Fraction:
        """Don't use for taxation."""
        portfolio_values = self.get_portfolio_values_on(date)
        return max(portfolio_values.values())

    @property
    def peak_portfolio_value_native_in_calendar_year(self) -> Fraction:
        """Don't use for taxation."""
        return self.country.get_peak_value_in_cy(
            self.get_peak_portfolio_value_on,
            in_inr=False,
        )

    @property
    def peak_portfolio_value_inr_in_calendar_year(self) -> Fraction:
        """Don't use for taxation."""
        return self.country.get_peak_value_in_cy(
            self.get_peak_portfolio_value_on,
            in_inr=True,
        )

    @property
    def closing_portfolio_value_native_in_calendar_year(self) -> Fraction:
        """Don't use for taxation."""
        portfolio_values = self.get_portfolio_values_on(cy_end())
        return portfolio_values["__closing"]

    @property
    def closing_portfolio_value_inr_in_calendar_year(self) -> Fraction:
        """Don't use for taxation."""
        return self.country.convert_to_inr_for_FA_income(
            self.closing_portfolio_value_native_in_calendar_year
        )

    def _sum_all_cash_and_stocks(
        self,
        attr: str,
        attr_args: list[Any] = None,
    ) -> Fraction:
        self._ensure_wallet_init()

        def get_attr_val(obj: object) -> Fraction:
            attr_value = getattr(obj, attr)
            return attr_value(*attr_args) if attr_args else attr_value

        total_cash_attr_value = sum(
            get_attr_val(wallet)
            for wallet in self._all_wallets
        )

        total_stock_attr_value = sum(
            get_attr_val(lot)
            for lot_map in self._lots.values()
            for lot in lot_map.values()
        )

        return total_cash_attr_value + total_stock_attr_value

    @property
    def total_dividends_native_in_calendar_year(self) -> Fraction:
        attr = "total_dividends_native_in_calendar_year"
        return self._sum_all_cash_and_stocks(attr)

    @property
    def total_dividends_inr_in_calendar_year(self) -> Fraction:
        return self.country.convert_to_inr_for_FA_income(
            self.total_dividends_native_in_calendar_year
        )

    @property
    def total_sale_proceeds_native_in_calendar_year(self) -> Fraction:
        return sum(
            lot.total_sale_proceeds_native_in_calendar_year
            for lot_map in self._lots.values()
            for lot in lot_map.values()
        )

    @property
    def total_sale_proceeds_inr_in_calendar_year(self) -> Fraction:
        return self.country.convert_to_inr_for_FA_income(
            self.total_sale_proceeds_native_in_calendar_year
        )

    @property
    def total_dividends_inr_in_financial_year_for_tax(self) -> Fraction:
        attr = "total_dividends_inr_in_financial_year_for_tax"
        return self._sum_all_cash_and_stocks(attr)

    @property
    def total_tax_withheld_on_dividends_inr_in_financial_year(
        self,
    ) -> Fraction:
        attr = "total_tax_withheld_on_dividends_inr_in_financial_year"
        return self._sum_all_cash_and_stocks(attr)

    @property
    def dividend_amount_inr_for_which_tax_was_withheld_in_financial_year(
        self
    ) -> Fraction:
        attr = "dividend_amount_inr_for_which_tax_was_withheld_in_financial_year"  # noqa: E501
        return self._sum_all_cash_and_stocks(attr)

    @property
    def total_captial_gain_inr_in_financial_year_for_tax(self) -> Fraction:
        return sum(
            lot.total_gain_inr_in_financial_year_for_tax
            for lot_map in self._lots.values()
            for lot in lot_map.values()
        )

    @property
    def total_tax_withheld_on_sell_inr_in_financial_year(self) -> Fraction:
        return sum(
            lot.total_tax_withheld_on_sell_inr_in_financial_year
            for lot_map in self._lots.values()
            for lot in lot_map.values()
        )

    @property
    def total_ltcg_amount_inr_in_financial_year(self) -> Fraction:
        return sum(
            lot.total_ltcg_amount_inr_in_financial_year
            for lot_map in self._lots.values()
            for lot in lot_map.values()
        )

    @property
    def total_stcg_amount_inr_in_financial_year(self) -> Fraction:
        return sum(
            lot.total_stcg_amount_inr_in_financial_year
            for lot_map in self._lots.values()
            for lot in lot_map.values()
        )

    @property
    def total_ltcg_tax_inr_in_financial_year(self) -> Fraction:
        return sum(
            lot.total_ltcg_tax_inr_in_financial_year
            for lot_map in self._lots.values()
            for lot in lot_map.values()
        )

    @property
    def total_stcg_tax_inr_in_financial_year(self) -> Fraction:
        return sum(
            lot.total_stcg_tax_inr_in_financial_year
            for lot_map in self._lots.values()
            for lot in lot_map.values()
        )

    @property
    def total_ltcg_tax_withheld_inr_in_financial_year(self) -> Fraction:
        return sum(
            lot.total_ltcg_tax_withheld_inr_in_financial_year
            for lot_map in self._lots.values()
            for lot in lot_map.values()
        )

    @property
    def total_stcg_tax_withheld_inr_in_financial_year(self) -> Fraction:
        return sum(
            lot.total_stcg_tax_withheld_inr_in_financial_year
            for lot_map in self._lots.values()
            for lot in lot_map.values()
        )

    @property
    def ltcg_amount_inr_for_which_tax_was_withheld_in_financial_year(
        self
    ) -> Fraction:
        return sum(
            lot.ltcg_amount_inr_for_which_tax_was_withheld_in_financial_year
            for lot_map in self._lots.values()
            for lot in lot_map.values()
        )

    @property
    def stcg_amount_inr_for_which_tax_was_withheld_in_financial_year(
        self
    ) -> Fraction:
        return sum(
            lot.stcg_amount_inr_for_which_tax_was_withheld_in_financial_year
            for lot_map in self._lots.values()
            for lot in lot_map.values()
        )

    @property
    def total_buy_sell_ltcg_tuple_inr_in_financial_year(
        self,
    ) -> tuple[Fraction, Fraction, Fraction]:
        all_tuples = (
            lot.total_buy_sell_ltcg_tuple_inr_in_financial_year
            for lot_map in self._lots.values()
            for lot in lot_map.values()
        )
        total_tuple = tuple(sum(x) for x in zip(*all_tuples))
        return total_tuple

    @property
    def total_buy_sell_stcg_tuple_inr_in_financial_year(
        self,
    ) -> tuple[Fraction, Fraction, Fraction]:
        all_tuples = (
            lot.total_buy_sell_stcg_tuple_inr_in_financial_year
            for lot_map in self._lots.values()
            for lot in lot_map.values()
        )
        total_tuple = tuple(sum(x) for x in zip(*all_tuples))
        return total_tuple

    def total_ltcg_amount_inr_for_advance_tax_installment(
        self,
        installment: int,
    ) -> Fraction:
        return sum(
            lot.total_ltcg_amount_inr_for_advance_tax_installment(installment)
            for lot_map in self._lots.values()
            for lot in lot_map.values()
        )

    def total_stcg_amount_inr_for_advance_tax_installment(
        self,
        installment: int,
    ) -> Fraction:
        return sum(
            lot.total_stcg_amount_inr_for_advance_tax_installment(installment)
            for lot_map in self._lots.values()
            for lot in lot_map.values()
        )

    def total_dividend_amount_inr_for_advance_tax_installment(
        self,
        installment: int,
    ) -> Fraction:
        attr = "total_dividend_amount_inr_for_advance_tax_installment"
        return self._sum_all_cash_and_stocks(attr, [installment])


###############################################################################


def get_input_items(section_key: str) -> Iterator[tuple[Hashable, Any]]:
    for key, value in input_dict[section_key].items():
        yield deepcopy(key), deepcopy(value)


def get_activity_values(
    activity: dict[str, Any],
) -> tuple[str, str, dict[str, Any], Entity | None, Broker]:
    if not isinstance(activity, dict):
        raise ValueError("Activity list item is not a dict.")

    if len(activity) != 1:
        raise ValueError("A list item dict has multiple top-level keys: "
                         + ", ".join(str(i) for i in activity.keys()))

    activity_id, activity_dict = list(activity.items())[0]

    activity_type = activity_dict["activity_type"]

    broker_id = activity_dict["broker"]
    broker = brokers[broker_id]

    entity_id = activity_dict.get("entity")
    entity = entities[entity_id] if entity_id else None

    if entity and entity.country_id != broker.country_id:
        raise ValueError(
            f"Specified entity {entity.entity_id} of country "
            f"{entity.country_id}, and broker {broker.broker_id} of "
            f"country {broker.country_id}."
        )

    return activity_id, activity_type, activity_dict, entity, broker


###############################################################################


def parse_countries() -> None:
    for country_id, country_dict in get_input_items("countries"):
        Country(country_id=country_id, **country_dict)


def parse_entities() -> None:
    for entity_id, entity_dict in get_input_items("entities"):
        id_kwargs = dict(entity_id=entity_id,
                         country_id=entity_dict.pop("country"))

        Entity(**id_kwargs, **entity_dict)


def parse_brokers() -> None:
    for broker_id, broker_dict in get_input_items("brokers"):
        id_kwargs = dict(broker_id=broker_id,
                         country_id=broker_dict.pop("country"))

        Broker(**id_kwargs, **broker_dict)


###############################################################################


"""
valid_activity_types:
    - vest              # It's free, no cash is involved, you just get stocks.
    - buy               # Cash is used up for buying.
    - stock_dividend    # When we get dividends from stocks. Increases cash.
    - cash_dividend     # When we get dividends from cash. Increases cash.
    - sell_fifo         # FIFO selling. 99% of time this is what you want.
    - sell_specific     # Specific unit/lot selling. Avoid if you don't know.
    - cash_opening      # Sets opening balance and the cash fund to use.
    - cash_fund_switch  # To change cash fund used and transfer existing money.
    - bank_to_cash      # Add funds from bank account to broker account.
    - cash_to_bank      # Withdraw cash from broker account to bank account.


valid_activity_types_in_opening_ledger:
    - cash_opening
    - vest
    - buy


invalid_activity_types_in_normal_ledger:
    - cash_opening


activity_types_which_have_tax_withholding:
    - sell_fifo
    - sell_specific
    - stock_dividend
    - cash_dividend
"""


def parse_opening_ledger() -> None:
    global PARSING_OPENING_LEDGER

    PARSING_OPENING_LEDGER = True
    opening_key = "opening_ledger_on_12_AM_jan_1_prev_fy"

    for activity in input_dict[opening_key]:
        activity_id, activity_type, activity_dict, entity, broker = \
            get_activity_values(activity)

        entity_id = entity.entity_id

        match activity_type:
            case "cash_opening":
                broker.cash_init(txn_id=activity_id, entity_id=entity_id,
                                 amount=activity_dict["amount"])

            case "vest":
                acq_dict = activity_dict["initial_acquisition"]
                broker.add_lot_buy(
                    txn_id=activity_id,
                    date=acq_dict["date"],
                    entity_id=entity_id,
                    units=activity_dict["remaining_units"],
                    cost_per_unit=acq_dict["stock_price_merchant_fmv"],
                    merchant_ttbr_if_vest=acq_dict["merchant_ttbr"],
                    for_opening_lot=True,
                )

            case "buy":
                acq_dict = activity_dict["initial_acquisition"]
                broker.add_lot_buy(
                    txn_id=activity_id,
                    date=acq_dict["date"],
                    entity_id=entity_id,
                    units=activity_dict["remaining_units"],
                    cost_per_unit=acq_dict["stock_price_in_broker_doc"],
                    for_opening_lot=True,
                )

            case _:
                raise ValueError(
                    f"Unsupported activity_type {activity_type} in "
                    f"{opening_key}."
                )

    PARSING_OPENING_LEDGER = False


def parse_main_activities() -> None:
    main_activity_key = "activity_from_jan_1_prev_fy_to_31_mar_current_fy"

    for activity in input_dict[main_activity_key]:
        activity_id, activity_type, activity_dict, entity, broker = \
            get_activity_values(activity)

        match activity_type:
            case "vest":
                broker.add_lot_buy(
                    txn_id=activity_id,
                    date=activity_dict["date"],
                    entity_id=entity.entity_id,
                    units=activity_dict["units"],
                    cost_per_unit=activity_dict["stock_price_merchant_fmv"],
                    merchant_ttbr_if_vest=activity_dict["merchant_ttbr"],
                )

            case "buy":
                broker.add_lot_buy(
                    txn_id=activity_id,
                    date=activity_dict["date"],
                    entity_id=entity.entity_id,
                    units=activity_dict["units"],
                    cost_per_unit=activity_dict["stock_price_in_broker_doc"],
                )

            case "stock_dividend":
                broker.receive_stock_dividend(
                    txn_id=activity_id,
                    date=activity_dict["date"],
                    entity_id=entity.entity_id,
                    amount=activity_dict["amount"],
                    misc_fees=activity_dict["misc_fees"],
                    tax_withholding_dict=activity_dict["tax_withholding"],
                )

            case "cash_dividend":
                broker.receive_cash_dividend(
                    txn_id=activity_id,
                    date=activity_dict["date"],
                    amount=activity_dict["amount"],
                    misc_fees=activity_dict["misc_fees"],
                    tax_withholding_dict=activity_dict["tax_withholding"],
                )

            case "sell_fifo":
                broker.sell_shares_fifo(
                    txn_id=activity_id,
                    date=activity_dict["date"],
                    entity_id=entity.entity_id,
                    units=activity_dict["units"],
                    cost_per_unit=activity_dict["stock_price_in_broker_doc"],
                    misc_fees=activity_dict["misc_fees"],
                    tax_withholding_dict=activity_dict["tax_withholding"],
                )

            case "sell_specific":
                stc_deduct = activity_dict.get(
                    "amount_deducted_on_sell_to_cover", ZERO
                )

                broker.sell_shares_specific(
                    txn_id=activity_id,
                    buy_txn_id=activity_dict["unit_lot_key"],
                    date=activity_dict["date"],
                    entity_id=entity.entity_id,
                    units=activity_dict["units"],
                    cost_per_unit=activity_dict["stock_price_in_broker_doc"],
                    misc_fees=activity_dict["misc_fees"],
                    tax_withholding_dict=activity_dict["tax_withholding"],
                    amount_deducted_on_sell_to_cover=stc_deduct,
                )

            case "cash_fund_switch":
                broker.cash_fund_switch(
                    txn_id=activity_id,
                    date=activity_dict["date"],
                    new_entity_id=activity_dict["new_entity"],
                    misc_fees=activity_dict["misc_fees"],
                )

            case "bank_to_cash":
                broker.add_cash(
                    txn_id=activity_id,
                    date=activity_dict["date"],
                    amount=activity_dict["amount"],
                    misc_fees=activity_dict["misc_fees"],
                )

            case "cash_to_bank":
                broker.withdraw_cash(
                    txn_id=activity_id,
                    date=activity_dict["date"],
                    amount=activity_dict["amount"],
                    ghar_vaapsi=True,
                )

            case _:
                raise ValueError(
                    f"Unsupported activity_type {activity_type} in "
                    f"{main_activity_key}."
                )


###############################################################################


def format_address_for_csv(address: str) -> str:
    """
    First we replace newlines with commas, to make address collapse into one
    line.

    Then we replace all commas with semicolon, because Infosys engineers are
    pathetically incompetent who rolled out their own CSV "parser" which does
    nothing but str.split(","), i.e. it is broken and not compliant with the
    standard.
    """
    return address.replace("\n", ", ").replace(", ", "; ")


def create_schedule_fa_table_a2() -> None:
    """
    This section is basically due to babudom's lazy engineering, and made worse
    by the sub-par engineering of Infosys.

    Essential to understanding this is the reverse order in FATCA & CRS - what
    gets reported to India from outside.

    The guideline PDF for FATCA & CRS by the ITD is as follows (also archived):
    https://incometaxindia.gov.in/Documents/exchange-of-information/Guidance-note-on-fatca-crs.pdf  # noqa: E501

    Basically, what the stupid babu wants is that:
        value_dict_by_institution == value_dict_by_individual

    The institution reports the entire portfolio value. The PDF says:
        For Debt or equity accounts, the account balance is the value of the
        debt or equity interest that the account holder has in the financial
        institution.

    Now when making entry / adding a row, there is a "Nature of Amount" field
    which is a dropdown (allows only one value). But the instituition reports
    all those values. The PDF says:
        In case of custodial account, apart from general reporting
        requirements, the following information is to be reported for each
        reporting period is:
        - The total gross amount of interest paid or credited to the account
        - The total amount of dividends paid or credited to the account,
        - The total gross amount of other income generated with respect to the
          assets held in the account paid or credited to the account,
        - The total gross proceeds from the sale or redemption of Financial
          Assets paid or credited to the account.

    So thanks to the incompetent engineer who made the form, we need to add
    multiple rows duplicating the data except the "Amount" field due to the
    differing "Nature" of the amounts.

    Certainly, the entire schedule FA can be improved massively with a small
    amount of engineering a college guy can do, but it seems ITD babus want a
    minefield for rent-seeking.
    """
    csv_rows = []

    for broker in brokers.values():
        country = broker.country

        broker_data_common = {
            "Country/Region name": country.name,
            "Country Name and Code": country.code,

            "Name of financial institution": broker.name,
            "Address of financial institution": format_address_for_csv(
                broker.address_without_zipcode
            ),
            "ZIP Code": broker.address_zipcode,

            "Account Number": broker.account_number,
            "Status": broker.a2_status,
            "Account opening date": broker.account_opening_date.isoformat(),

            "Peak Balance During the Period":
                int(broker.peak_portfolio_value_inr_in_calendar_year),
            "Closing balance":
                int(broker.closing_portfolio_value_inr_in_calendar_year),
        }

        # From the ITR2 JSON schema, the nature field is specified by:
        #   I - Interest
        #   D - Dividend
        #   S - Proceeds from sale or redemption of financial assets
        #   O - Other income
        #   N - No Amount paid/credited
        #
        # In this program, only dividend and sale proceeds is possible.

        broker_data_dividend = {
            "Nature of Amount": "D",
            "Amount": int(broker.total_dividends_inr_in_calendar_year),
        }

        broker_data_proceeds = {
            "Nature of Amount": "S",
            "Amount": int(broker.total_sale_proceeds_inr_in_calendar_year),
        }

        broker_rows = []

        for amount_data in (broker_data_dividend, broker_data_proceeds):
            if amount_data["Amount"] != 0:
                broker_rows.append(broker_data_common | amount_data)

        if not broker_rows:
            broker_rows.append(
                broker_data_common | {"Nature of Amount": "N", "Amount": 0}
            )

        csv_rows += broker_rows

    csv_path = output_dir / "schedule_fa_table_a2.csv"

    with open(csv_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    rel_path = csv_path.relative_to(Path(__file__).parent)
    print(f"Schedule FA Table A2 stored in {rel_path}")


def create_schedule_fa_table_a3() -> None:
    csv_rows = []

    for broker in brokers.values():
        for entity_lot_map in broker._lots.values():
            for lot in entity_lot_map.values():
                country = lot.country
                entity = lot.entity
                buy_txn = lot.buy_txn_obj

                csv_rows.append({
                    "Country/Region name": country.name,
                    "Country Name and Code": country.code,

                    "Name of entity": entity.name,
                    "Address of entity": format_address_for_csv(
                        entity.company_address_without_zipcode
                    ),
                    "ZIP Code": entity.company_address_zipcode,
                    "Nature of entity": entity.nature,

                    "Date of acquiring the interest": buy_txn.date.isoformat(),
                    "Initial value of the investment":
                        int(buy_txn.total_buy_value_inr_for_initial_acquire),

                    "Peak value of investment during the Period":
                        int(lot.peak_holding_value_inr_in_calendar_year),
                    "Closing balance":
                        int(lot.closing_value_inr_in_calendar_year),

                    "Total gross amount paid/credited with respect to the holding during the period":  # noqa: E501
                        int(lot.total_dividends_inr_in_calendar_year),
                    "Total gross proceeds from sale or redemption of investment during the period":  # noqa: E501
                        int(lot.total_sale_proceeds_inr_in_calendar_year),
                })

    csv_path = output_dir / "schedule_fa_table_a3.csv"

    with open(csv_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    rel_path = csv_path.relative_to(Path(__file__).parent)
    print(f"Schedule FA Table A3 stored in {rel_path}")


def create_capital_gain_and_dividends_and_get_average_tax_rate() -> Fraction:
    cg_dict = {
        "total_cg_or_cl": ZERO,

        "ltcg_or_ltcl": ZERO,
        "long_term_buy": ZERO,
        "long_term_sell": ZERO,

        "stcg_or_stcl": ZERO,
        "short_term_buy": ZERO,
        "short_term_sell": ZERO,

        "dividends": ZERO,

        "ltcg_accruals_advance_tax_installments": [],
        "stcg_accruals_advance_tax_installments": [],
        "dividend_accruals_advance_tax_installments": [],
    }

    for broker in brokers.values():
        cg_dict["total_cg_or_cl"] += \
            broker.total_captial_gain_inr_in_financial_year_for_tax

        ltb, lts, ltcg = broker.total_buy_sell_ltcg_tuple_inr_in_financial_year
        cg_dict["ltcg_or_ltcl"] += ltcg
        cg_dict["long_term_buy"] += ltb
        cg_dict["long_term_sell"] += lts

        stb, sts, stcg = broker.total_buy_sell_stcg_tuple_inr_in_financial_year
        cg_dict["stcg_or_stcl"] += stcg
        cg_dict["short_term_buy"] += stb
        cg_dict["short_term_sell"] += sts

        cg_dict["dividends"] += \
            broker.total_dividends_inr_in_financial_year_for_tax

        ltcg_accruals = []
        stcg_accruals = []
        dividend_accruals = []

        for i in range(1, 6):
            ltcg_accruals.append(
                broker.total_ltcg_amount_inr_for_advance_tax_installment(i)
            )
            stcg_accruals.append(
                broker.total_stcg_amount_inr_for_advance_tax_installment(i)
            )
            dividend_accruals.append(
                broker.total_dividend_amount_inr_for_advance_tax_installment(i)
            )

        cg_dict["ltcg_accruals_advance_tax_installments"].append(ltcg_accruals)
        cg_dict["stcg_accruals_advance_tax_installments"].append(stcg_accruals)
        cg_dict["dividend_accruals_advance_tax_installments"].append(
            dividend_accruals
        )
    # End of for loop.

    # Sum / Flatten all the installments into one.
    for key in ("ltcg", "stcg", "dividend"):
        full_key = key + "_accruals_advance_tax_installments"
        total_list = [sum(x) for x in zip(*cg_dict[full_key])]
        cg_dict[full_key] = total_list

    cg_dict_for_export = {}
    for key, value in cg_dict.items():
        if isinstance(value, Fraction):
            cg_dict_for_export[key] = format(value, ".2f")
        else:
            cg_dict_for_export[key] = [format(i, ".2f") for i in value]

    txt_path = output_dir / "capital_gain_and_dividend.txt"
    with open(txt_path, "w") as f:
        json.dump(cg_dict_for_export, f, indent="\t")

    rel_path = txt_path.relative_to(Path(__file__).parent)

    print("\n\n")
    print(cleandoc(f"""
        Capital gain and dividend info for reference stored in {rel_path}

        Capital gain reporting goes in "Schedule Capital Gain".
        Dividend reporting goes in "Schedule Other Sources".

        In the schedules, the total value has to be reported, so add the values
        we calculated to the existing values if any, making sure the existing
        doesn't contain values for assets which we calculated.

        Fill up all the info and then get back here. We need to do this because
        we want to calculate the average tax rate (total tax / total income)
        for schedule FSI and form 67.

        Press enter to continue...
    """))

    input()

    total_income = Fraction(
        input("Enter aggregate income (Part B - TI): ").replace(",", "")
    )
    total_tax = Fraction(
        input("Enter tax payable after rebate (Part B - TTI): ")
        .replace(",", "")
    )
    print("\n\n")

    avg_tax_rate = total_tax / total_income
    return avg_tax_rate


def get_earnings_per_country() -> dict[str, dict[str, Fraction]]:
    country_id_to_data = {}

    for broker in brokers.values():
        country = broker.country
        country_id = country.country_id

        if country_id not in country_id_to_data:
            country_id_to_data[country_id] = {
                "ltcg_without_tax_withheld": ZERO,
                "ltcg_with_tax_withheld": ZERO,
                "tax_withheld_on_ltcg": ZERO,

                "stcg_without_tax_withheld": ZERO,
                "stcg_with_tax_withheld": ZERO,
                "tax_withheld_on_stcg": ZERO,

                "dividend_without_tax_withheld": ZERO,
                "dividend_with_tax_withheld": ZERO,
                "tax_withheld_on_dividend": ZERO,
            }

        # LTCG:

        ltcg = broker.total_ltcg_amount_inr_in_financial_year
        ltcg_with_witholding = \
            broker.ltcg_amount_inr_for_which_tax_was_withheld_in_financial_year

        country_id_to_data[country_id]["ltcg_without_tax_withheld"] += \
            ltcg - ltcg_with_witholding

        country_id_to_data[country_id]["ltcg_with_tax_withheld"] += \
            ltcg_with_witholding

        country_id_to_data[country_id]["tax_withheld_on_ltcg"] += \
            broker.total_ltcg_tax_withheld_inr_in_financial_year

        # STCG:

        stcg = broker.total_stcg_amount_inr_in_financial_year
        stcg_with_witholding = \
            broker.stcg_amount_inr_for_which_tax_was_withheld_in_financial_year

        country_id_to_data[country_id]["stcg_without_tax_withheld"] += \
            stcg - stcg_with_witholding

        country_id_to_data[country_id]["stcg_with_tax_withheld"] += \
            stcg_with_witholding

        country_id_to_data[country_id]["tax_withheld_on_stcg"] += \
            broker.total_stcg_tax_withheld_inr_in_financial_year

        # Dividend:

        dividend = broker.total_dividends_inr_in_financial_year_for_tax
        dividend_with_witholding = \
            broker.dividend_amount_inr_for_which_tax_was_withheld_in_financial_year  # noqa: E501

        country_id_to_data[country_id]["dividend_without_tax_withheld"] += \
            dividend - dividend_with_witholding

        country_id_to_data[country_id]["dividend_with_tax_withheld"] += \
            dividend_with_witholding

        country_id_to_data[country_id]["tax_withheld_on_dividend"] += \
            broker.total_tax_withheld_on_dividends_inr_in_financial_year
    # End of for loop.

    return country_id_to_data


def create_schedule_fsi_and_form_67(avg_tax_rate: Fraction) -> None:
    avg_tax_rate_percent_str = format(avg_tax_rate * 100, ".4f")
    fsi_csv_rows = []
    form67_csv_rows = []

    def add_to_fsi(
        country: Country,
        income_type_suffix_for_attr: str,
        income_value: Fraction,
        tax_withheld: Fraction,
    ) -> None:
        nonlocal fsi_csv_rows

        income_type_mapping = {"ltcg": "LTCG", "stcg": "STCG",
                               "dividend": "Dividend"}

        income_type = income_type_mapping[income_type_suffix_for_attr]
        income_type += " without" if tax_withheld == 0 else " with"
        income_type += " tax withholding"

        def country_attr(attr: str) -> Any:
            return getattr(country, f"{attr}_{income_type_suffix_for_attr}")

        fsi_csv_rows.append({
            "Country code": country.code,
            "Country name": country.name,

            "Income type": income_type,
            "Income value": int(income_value),

            "Applicable tax rate in India (%)": avg_tax_rate_percent_str,
            "Tax payable in India": max(0, int(income_value * avg_tax_rate)),

            "Tax withheld outside India": int(tax_withheld),
            "Tax withholding rate (%)":
                int(country_attr("tax_withholding_rate_percent_for")),

            "Income DTAA tax rate (%)":
                int(country_attr("dtaa_tax_rate_percent")),
            "Income DTAA article": country_attr("dtaa_article"),
        })
    # End of add_to_fsi().

    def add_to_form67(
        country: Country,
        income_type_key: str,
        income_value: Fraction,
        tax_withheld: Fraction,
    ) -> None:
        int_tax_in_india = int(income_value * avg_tax_rate)
        int_tax_withheld = int(tax_withheld)
        int_credit_claimed = min(int_tax_in_india, int_tax_withheld)

        income_type_mapping = {
            "ltcg": "Long term capital gain",
            "stcg": "Short term capital gain",
            "dividend": "Dividend",
        }

        def country_attr(attr: str) -> Any:
            return getattr(country, f"{attr}_{income_type_key}")

        # We do not support tax rate different from DTAA rate as of now.
        tax_rate_pc_outside_india = format(
            country_attr("dtaa_tax_rate_percent"), ".2f"
        )

        # Since column names are same, the stupid engineer chose a trailing
        # space as the charcter to uniqify the column name.
        space = " "

        form67_csv_rows.append({
            "Sl. No.": len(form67_csv_rows) + 1,

            # Docs say code can be used, but the docs and website is broken,
            # no matter you use name or code. Poor engineering by the Infosys
            # contractor.
            "Name of the country/specified territory": country.name,
            "Please specify": None,

            "Source of income": income_type_mapping[income_type_key],
            f"Please specify{space}": None,

            "Income from outside India": int(income_value),
            "Amount": int_tax_withheld,
            "Rate(%)": tax_rate_pc_outside_india,

            "Tax payable on such income under normal provisions in India":
                int_tax_in_india,
            "Tax payable on such income under Section 115JB/JC": 0,

            "Article No. of Double Taxation Avoidance Agreements":
                country_attr("dtaa_article"),
            "Rate of tax as per Double Taxation Avoidance Agreements(%)":
                tax_rate_pc_outside_india,

            f"Amount{space}": int_credit_claimed,
            "Credit claimed under section 91": 0,
            "Total foreign tax credit claimed": int_credit_claimed,
        })

    for country_id, data in get_earnings_per_country().items():
        country = countries[country_id]

        # Make entry for LTCG without withholding.
        if data["ltcg_without_tax_withheld"] != 0:
            add_to_fsi(country, "ltcg", data["ltcg_without_tax_withheld"],
                       ZERO)

        # Make entry for LTCG with withholding.
        if data["ltcg_with_tax_withheld"] != 0:
            args = (country, "ltcg", data["ltcg_with_tax_withheld"],
                    data["tax_withheld_on_ltcg"])
            add_to_fsi(*args)
            add_to_form67(*args)

        # Make entry for STCG without withholding.
        if data["stcg_without_tax_withheld"] != 0:
            add_to_fsi(country, "stcg", data["stcg_without_tax_withheld"],
                       ZERO)

        # Make entry for STCG with withholding.
        if data["stcg_with_tax_withheld"] != 0:
            args = (country, "stcg", data["stcg_with_tax_withheld"],
                    data["tax_withheld_on_stcg"])
            add_to_fsi(*args)
            add_to_form67(*args)

        # Make entry for dividend without withholding.
        if data["dividend_without_tax_withheld"] != 0:
            add_to_fsi(country, "dividend",
                       data["dividend_without_tax_withheld"], ZERO)

        # Make entry for dividend with withholding.
        if data["dividend_with_tax_withheld"] != 0:
            args = (
                country,
                "dividend",
                data["dividend_with_tax_withheld"],
                data["tax_withheld_on_dividend"],
            )
            add_to_fsi(*args)
            add_to_form67(*args)

    fsi_csv_path = output_dir / "schedule_fsi.csv"

    with open(fsi_csv_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(fsi_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fsi_csv_rows)

    fsi_rel_path = fsi_csv_path.relative_to(Path(__file__).parent)
    print("Schedule FSI partially filled for reference stored in "
          f"{fsi_rel_path}")

    form67_csv_path = output_dir / "form_67.csv"

    with open(form67_csv_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(form67_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(form67_csv_rows)

    form67_rel_path = form67_csv_path.relative_to(Path(__file__).parent)
    print()
    print(cleandoc(f"""
        Form 67 stored in {form67_rel_path}

        Note: After uploading form 67 CSV, you'll see error - open the CSV and
        manually fill the values. It's an issue with the website - if you
        export after filling on the website and upload the same file directly,
        you'll see the same error. The stupid Infosys guy who made the broken
        CSV parser on ITD website didn't even bother to QA it.
    """))


###############################################################################


def parse_input() -> None:
    parse_countries()
    parse_entities()
    parse_brokers()
    parse_opening_ledger()
    parse_main_activities()


def main() -> int:
    # Modifications to this is also an agreement to the license, which applies
    # to the source code.
    print(cleandoc("""
        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU Affero General Public License for more details.

        This program is licensed under the AGPL-3.0-or-later license. Do you
        understand the implications and agree to it?
    """))
    yn = input("(yes/no): ")
    if yn.lower() != "yes":
        # Modifications to this is also an agreement to the license, which
        # applies to the source code.
        print("Okay, exiting. Agreement to the license is a must to proceed.")
        return 1

    print()

    # Modifications to this implies acceptance.
    print(cleandoc("""
        Since you understand the implications, is it clear to you that NOBODY
        else but you, and ONLY you, are responsible for your ITR filing?
    """))
    yn = input("(yes/no): ")
    if yn.lower() != "yes":
        # Modifications to this implies acceptance.
        print("Well then, go and try to understand that simple fact!")
        return 1

    print("\n" + "-" * 79 + "\n")

    print(cleandoc("""
        Have you filled (and not submitted) *EVERYTHING* in your ITR EXCEPT the
        foreign asset stuff (reporting/gain/dividend) for which we are going to
        calculate?
    """))
    yn = input("(yes/no): ")

    if yn.lower() != "yes":
        print("Then go and fill up everything before doing this!")
        return 1

    print("\n")
    fetch_input()
    fetch_fx_rates()
    parse_input()

    print("\n" + "-" * 79 + "\n")

    create_schedule_fa_table_a2()
    create_schedule_fa_table_a3()

    avg_tax_rate = create_capital_gain_and_dividends_and_get_average_tax_rate()
    create_schedule_fsi_and_form_67(avg_tax_rate)

    return 0


if __name__ == "__main__":
    sys.exit(main())
