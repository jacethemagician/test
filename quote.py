import sys
import os
import pandas as pd
import asyncio

path_to_otc_gleba_service_client = os.path.abspath(
    "C:\\Users\\yangzequn\\GalaxyDerivativesProjects\\otc_quotation_generation_new\\otc_gleba_service_client"
)
sys.path.insert(0, path_to_otc_gleba_service_client)

from quote_generator import QuoteGenerator  # updated
from constants import KI_BARRIERS, KO_STARTS, VANILLA_STRIKE2S


class QuoteHandler:
    def __init__(self):
        self.quote_generator = QuoteGenerator()

    async def initialize(self):
        await self.quote_generator.initialize()

    def handle_vanilla_quote(self, row, column):  # updated with self
        structure, tenor = column.split("_")
        start_date = row["报价日期"]
        underlyings = row["股票代码"]
        vol = row[column]

        if structure.endswith("-buy"):
            sell_structure = structure.replace("buy", "sell")
            sell_vol = row[f"{sell_structure}_{tenor}"]
            sell_result = self.quote_generator.get_vanilla_quote(  # updated
                start_date, underlyings, sell_structure, tenor, sell_vol
            )
            buy_result = self.quote_generator.get_vanilla_quote(  # updated
                start_date, underlyings, structure, tenor, vol
            )
            buy_result["报价"] -= (
                1 - float(structure.split("-")[0][2:]) / 100
            ) * sell_result["报价"]
            structure = structure.replace("-buy", "")
            return buy_result
        elif structure.endswith("-sell"):
            return None  # skip the sell structures
        else:
            return self.quote_generator.get_vanilla_quote(  # updated
                start_date, underlyings, structure, tenor, vol
            )

    async def handle_snowball_quote(self, row, column):  # updated with self
        structure, tenor = column.split("_")
        start_date = row["报价日期"]
        underlyings = row["股票代码"]
        vol = row[column]
        return await self.quote_generator.get_snowball_quotes(  # updated
            start_date, underlyings, structure, tenor, vol
        )

    def count_tasks(self, df, quote_type):  # updated with self
        if quote_type == "vanilla":
            total_vanilla_tasks = sum(
                1
                for _, row in df.iterrows()
                for column in df.columns[2:]
                if not column.endswith("-sell") and "_source" not in column
            )
            print(f"Total vanilla tasks to complete: {total_vanilla_tasks}")
        elif quote_type == "snowball":
            total_snowball_tasks = 0
            for _, row in df.iterrows():
                for column in df.columns[2:]:
                    if "_source" not in column:
                        total_snowball_tasks += (
                            len(KI_BARRIERS)
                            * len(KO_STARTS)
                            * (len(VANILLA_STRIKE2S) + 1)
                        )
            print(f"Total snowball tasks to complete: {total_snowball_tasks}")
        else:
            raise ValueError(f"Unknown quote type: {quote_type}")

    async def main(self, df, quote_type):  # updated with self
        self.count_tasks(df, quote_type)

        results = []  # list to store results

        for _, row in df.iterrows():
            tasks = []  # tasks for this row
            for column in df.columns[2:]:
                if "_source" in column:
                    continue

                if quote_type == "vanilla":
                    result = self.handle_vanilla_quote(row, column)  # updated
                    if result is not None:
                        results.append(result)
                elif quote_type == "snowball":
                    tasks.append(self.handle_snowball_quote(row, column))  # updated
                else:
                    raise ValueError(f"Unknown quote type: {quote_type}")

            if tasks:  # if there are any tasks for this row
                results_from_tasks = await asyncio.gather(
                    *tasks
                )  # get results from tasks
                results.extend(
                    result for sublist in results_from_tasks for result in sublist
                )  # flatten and add to results

        return pd.DataFrame(results)

    async def close(self):
        await self.quote_generator.close()
