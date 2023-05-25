import datetime
from dateutil.relativedelta import relativedelta


def get_yyyymm(months_delay: int = 0) -> str:
    """
    Computes year-month in the YYYYMM format based on current run date and given delay in months.

    Args:
        months_delay (int): Number of months to adjust. `0` for current month, `1` for previous month, etc.
    
    Returns:
        str: Year-month in the YYYYMM format (e.g. '202204')
    """
    # Run date
    run_date = datetime.date.today().replace(day=1)  # Set to first day

    # Adjust the date as per months_delay
    reference_date = run_date - relativedelta(months=months_delay)
    print(run_date, reference_date)

    year = reference_date.year
    month = reference_date.month
    yyyyymm = year * 100 + month
    return str(yyyyymm)
