from datetime import datetime
from date_utils import get_next_tday
from vol import main as vol_main
import pandas as pd
from quote import QuoteHandler  # Importing the class
import pickle
import os
import asyncio
import time
from prettytable import PrettyTable
from datetime import datetime

pd.set_option("display.max_columns", None)

# Create instance of QuoteHandler
quote_handler = QuoteHandler()


def print_server_stats(stats, prefix=""):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    table = PrettyTable()
    table.field_names = ["Server", "Status"]
    for server, status in stats.items():
        table.add_row([server, status])

    print(f"{prefix}Time: {current_time}")
    print(table)


async def print_server_stats_periodically():
    while True:
        stats = quote_handler.quote_generator.load_balancing_client.get_server_stats()
        print_server_stats(stats)
        await asyncio.sleep(1)  # sleep for one second


async def main_program():
    # today = datetime.today()
    today = datetime(2023, 7, 21)
    today_str_ymd = today.strftime("%Y%m%d")
    next_tday = get_next_tday(today)
    next_tday_str_md = next_tday.strftime("%m%d")
    next_tday_str_ymd = next_tday.strftime("%Y%m%d")
    next_tday_str_ymd_dash = next_tday.strftime("%Y-%m-%d")
    DR_PATH = f"最新报价{next_tday_str_md}.xlsm"
    ZQ_PATH = f"Quote Vol-{today_str_ymd}.xlsx"

    if not os.path.isfile("vanilla_vol_df.pkl") or not os.path.isfile(
        "snowball_vol_df.pkl"
    ):
        vanilla_vol_df, snowball_vol_df = vol_main(
            DR_PATH, ZQ_PATH, next_tday_str_ymd_dash
        )
        with open("vanilla_vol_df.pkl", "wb") as f:
            pickle.dump(vanilla_vol_df, f)
        with open("snowball_vol_df.pkl", "wb") as f:
            pickle.dump(snowball_vol_df, f)
    else:
        with open("vanilla_vol_df.pkl", "rb") as f:
            vanilla_vol_df = pickle.load(f)
        with open("snowball_vol_df.pkl", "rb") as f:
            snowball_vol_df = pickle.load(f)

        # quote_main(vanilla_vol_df, "vanilla")

    await quote_handler.initialize()

    print_stats_task = asyncio.create_task(print_server_stats_periodically())

    start_time = time.time()
    df = await quote_handler.main(snowball_vol_df[:2], "snowball")  # Use instance here
    end_time = time.time()

    print_stats_task.cancel()

    print(f"quote_main function took {end_time - start_time} seconds to run")

    final_server_stats = (
        quote_handler.quote_generator.load_balancing_client.get_server_stats()
    )
    print_server_stats(final_server_stats, prefix="Final server ")

    await quote_handler.close()


if __name__ == "__main__":
    asyncio.run(main_program())
