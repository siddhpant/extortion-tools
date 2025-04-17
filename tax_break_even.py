# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025 Siddh Raman Pant <@siddhpant on GitHub>


from inspect import cleandoc
from textwrap import wrap


# Standard deduction.
old_std_deduction = 50_000
new_std_deduction = 75_000

# NPS rate.
old_nps_rate = 0.10
new_nps_rate = 0.14

# Rebate under section 87A.
old_rebate_max_income = 5_00_000
new_rebate_max_income = 12_00_000

old_slabs = {
    # income > key => tax rate = value; dict is ordered.
    10_00_000: 0.30,
    5_00_000:  0.20,
    2_50_000:  0.05,
}

new_slabs = {
    # income > key => tax rate = value; dict is ordered.
    24_00_000: 0.30,
    20_00_000: 0.25,
    16_00_000: 0.20,
    12_00_000: 0.15,
    8_00_000:  0.10,
    4_00_000:  0.05,
}

surcharge_min = 50_00_000

old_surcharges = {
    # income > key => surcharge = value; dict is ordered.
    5_00_00_000: 0.37,
    2_00_00_000: 0.25,
    1_00_00_000: 0.15,
    50_00_000:   0.10,
}

new_surcharges = {
    # income > key => surcharge = value; dict is ordered.
    2_00_00_000: 0.25,
    1_00_00_000: 0.15,
    50_00_000:   0.10,
}


def calculate_tax_from_rate_dict(
    total: float,
    rate_dict: dict[int, float]
) -> float:
    """Calculate tax from the format of ordered rate dicts as define above."""
    tax = 0.0

    for prev_limit, rate in rate_dict.items():
        if total > prev_limit:
            bracket_amount = total - prev_limit
            tax += bracket_amount * rate
            total = prev_limit

    return tax


def calculate_marginal_relief(
    high_tax: float,
    low_tax: float,
    high_income: float,
    low_income: float,
) -> float:
    """Calcultes the marginal relief."""
    tax_delta = high_tax - low_tax
    income_delta = high_income - low_income
    return (tax_delta - income_delta) if tax_delta > income_delta else 0


def calculate_tax(
    deducted_total: float,
    rebate_max_income: float,
    slabs: dict[int, float],
    surcharges: dict[int, float],
) -> float:
    """Main tax calculator for the given gross taxable income."""
    tax = 0.0
    total = deducted_total

    # Rebate under section 87A.
    if total <= rebate_max_income:
        return tax

    # Calculate slab based tax.
    tax += calculate_tax_from_rate_dict(total, slabs)

    if total < surcharge_min:
        # Subtract marginal relief from rebated income, if any.
        tax -= calculate_marginal_relief(tax, 0, total, rebate_max_income)
    else:
        # Calculate surcharge.
        tax += calculate_tax_from_rate_dict(total, surcharges)

        # Subtract marginal relief income on surcharge thresholds, if any.
        for threshold in surcharges.keys():
            if total > threshold:
                # Calculate tax on threshold.
                lower = calculate_tax_from_rate_dict(threshold, slabs)
                lower += calculate_tax_from_rate_dict(threshold, surcharges)

                # Subtract marginal relief, if any.
                tax -= calculate_marginal_relief(tax, lower, total, threshold)

    # Add 4% health & education cess.
    tax += tax * 0.04

    return tax


def old_regime_tax(total: float) -> float:
    return calculate_tax(total, old_rebate_max_income, old_slabs,
                         old_surcharges)


def new_regime_tax(total: float) -> float:
    return calculate_tax(total, new_rebate_max_income, new_slabs,
                         new_surcharges)


def format_num_indian(num: float) -> str:
    """
    Format number in Indian way (1,23,45,678).
    Can't use locale since user may not have the en_IN locale, or may use an
    inferior OS like Windows.
    """
    whole, fraction = str(round(num, 2)).split(".")
    num_str = ""

    if len(whole) <= 3:
        num_str = str(num)
    else:
        whole_start, whole_end = whole[:-3], whole[-3:]
        whole_fmt = ""

        if len(whole_start) % 2 != 0:
            whole_fmt += whole_start[0] + ","
            whole_start = whole_start[1:]

        whole_fmt += ",".join(wrap(whole_start, 2))

        num_str = f"{whole_fmt},{whole_end}.{fraction}"

    return num_str


def find_break_even_point(given_total: float, target_tax: float) -> float:
    """
    A quickly-whipped binary-search for finding break-even point, which is just
    the reduced total needed to have the target tax.
    """
    reduced_total = None
    high_total = given_total
    low_total = 0

    # If 0 tax, we can just return the max in old regime to get the 0 tax.
    if target_tax == 0:
        return float(old_rebate_max_income)

    while high_total >= low_total:
        mid_total = (high_total + low_total) / 2
        mid_tax = old_regime_tax(mid_total)

        # Need to round otherwise we will have infinite loop due to IEEE repr.
        if round(mid_tax, 2) == round(target_tax, 2):
            reduced_total = mid_total
            break

        elif mid_tax > target_tax:  # Go to left partition.
            high_total = mid_total

        else:  # Go to right partition.
            low_total = mid_total

    if reduced_total is None:
        raise RuntimeError("Binary search for reduced_total failed!")

    return reduced_total


def print_break_even_point(total: float, basic: float) -> float:
    """
    Calculate the minimum amount of deductions needed in old regime to make the
    tax equal to new regime. More deductions mean less tax in old regime.
    """
    # Standard deduction
    old_total = total - old_std_deduction
    new_total = total - new_std_deduction

    old_tax = old_regime_tax(old_total)
    new_tax = new_regime_tax(new_total)

    if new_tax > old_tax:
        raise RuntimeError("Old tax less than new tax without deductions.")

    # Deduct max NPS and calculate tax.
    old_total_nps = old_total - (basic * old_nps_rate)
    new_total_nps = new_total - (basic * new_nps_rate)

    old_nps_tax = old_regime_tax(old_total_nps)
    new_nps_tax = new_regime_tax(new_total_nps)

    # Print all info.
    print("\n")
    print(cleandoc(f"""
        Tax liabilities for given salaried income:
        ------------------------------------------
                Old regime: ₹{format_num_indian(old_tax)}
                New regime: ₹{format_num_indian(new_tax)}

                Old regime with max NPS: ₹{format_num_indian(old_nps_tax)}
                New regime with max NPS: ₹{format_num_indian(new_nps_tax)}
    """))

    reduced_old_normal = find_break_even_point(old_total, new_tax)
    reduced_old_both_nps = find_break_even_point(old_total_nps, new_nps_tax)
    reduced_old_only_nps = find_break_even_point(old_total_nps, new_tax)

    gross_str = format_num_indian(reduced_old_normal)
    deduct_str = format_num_indian(old_total - reduced_old_normal)

    gross_both_nps_str = format_num_indian(reduced_old_both_nps)
    deduct_both_nps_str = format_num_indian(old_total_nps
                                            - reduced_old_both_nps)

    gross_old_only_nps_str = format_num_indian(reduced_old_only_nps)
    deduct_old_only_nps_str = format_num_indian(old_total_nps
                                                - reduced_old_only_nps)

    print("\n")
    print(cleandoc(f"""
        Break-even point for old regime:
        --------------------------------
                Normal:
                        Deductions: ₹{deduct_str}
                        Gross taxable income: ₹{gross_str}

                Both NPS:
                        Deductions: ₹{deduct_both_nps_str}
                        Gross taxable income: ₹{gross_both_nps_str}

                Old max NPS, new no NPS:
                        Deductions: ₹{deduct_old_only_nps_str}
                        Gross taxable income: ₹{gross_old_only_nps_str}
    """))


if __name__ == "__main__":
    total = float(input("Enter total salary (with RSU): ₹").replace(",", ""))
    basic = float(input("Enter the annual basic salary: ₹").replace(",", ""))

    print("\n")
    print(f"Given total salary: ₹{format_num_indian(total)}")
    print(f"Given basic salary: ₹{format_num_indian(basic)}")

    print_break_even_point(total, basic)

    print("\n")
    print("PS: All floats are rounded to 2 decimal places for display.")
