import itertools
from datetime import datetime
from date_utils import get_expiry_date
from constants import (
    VANILLA_R,
    VANILLA_Q,
    SNOWBALL_RF,
    SNOWBALL_CREDIT_SPREAD,
    SNOWBALL_REPO_SPREAD,
    STRIKE_MAP,
    KI_BARRIERS,
    KO_STARTS,
    VANILLA_STRIKE2S,
    GRPC_SERVER_CONFIGS,
)
from load_balancing_client import DynamicLoadBalancingClient
import asyncio


class QuoteGenerator:
    def __init__(self):
        self.load_balancing_client = DynamicLoadBalancingClient(GRPC_SERVER_CONFIGS)

    async def initialize(self):
        await self.load_balancing_client.initialize(1)

    async def get_vanilla_quote(self, start_date, underlyings, structure, tenor, vol):
        if structure not in STRIKE_MAP:
            raise ValueError("vanilla strike is invalid")

        strike = STRIKE_MAP[structure]
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        expiry_date = get_expiry_date(start_date, tenor)

        vanilla_pricer = self.load_balancing_client.get_vanilla_pricer()

        res = vanilla_pricer.evaluate(
            underlyings=underlyings,
            start_date=start_date,
            expiry_date=expiry_date,
            call_put="CALL",
            strike=strike,
            spot=1,
            spot_date=start_date,
            vol_surface_value=vol,
            rate_curve_value=VANILLA_R,
            borrow_rate=VANILLA_Q,
            fixings_date=start_date.strftime("%Y-%m-%d"),
            fixings_date_value="1",
            model_date=start_date,
            result_params="npv",
        )

        result_dict = {
            "报价日期": start_date.strftime("%Y-%m-%d"),
            "股票代码": underlyings,
            "报价类型": "vanilla",
            "结构": structure,
            "期限": tenor,
            "波动率": vol,
            "报价": res[0],
        }
        return result_dict

    async def get_snowball_quotes(self, start_date, underlyings, structure, tenor, vol):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        periods = int(tenor[:-1])
        if tenor.endswith("M"):
            freq = "monthly"
        else:
            raise ValueError("snowball tenor is invalid")

        quote_params = []
        tasks = []

        for ki_barrier, ko_start in itertools.product(KI_BARRIERS, KO_STARTS):
            for vanilla_strike2 in VANILLA_STRIKE2S + [ki_barrier]:
                param = {
                    "报价日期": start_date.strftime("%Y-%m-%d"),
                    "股票代码": underlyings,
                    "报价类型": "snowball",
                    "结构": structure,
                    "期限": tenor,
                    "敲入": ki_barrier,
                    "敲出": ko_start,
                    "追保": vanilla_strike2,
                    "波动率": vol,
                }
                quote_params.append(param)

                tasks.append(
                    self.get_snowball_quote(
                        structure,
                        underlyings,
                        start_date,
                        freq,
                        periods,
                        ki_barrier,
                        ko_start,
                        vanilla_strike2,
                        vol,
                    )
                )

        quote_results = await asyncio.gather(*tasks)

        quotes = [
            dict(param, 报价=result) for param, result in zip(quote_params, quote_results)
        ]

        return quotes

    async def get_snowball_quote(
        self,
        structure,
        underlyings,
        start_date,
        freq,
        periods,
        ki_barrier,
        ko_start,
        vanilla_strike2,
        vol,
    ):
        server = await self.load_balancing_client.get_least_busy_client()
        try:
            snowball_pricer = await self.load_balancing_client.get_snowball_pricer(
                server
            )
            result = await asyncio.to_thread(
                self.snowball_task_func,
                snowball_pricer,
                structure,
                underlyings,
                start_date,
                freq,
                periods,
                ki_barrier,
                ko_start,
                vanilla_strike2,
                vol,
            )
            return result
        except Exception as e:
            # handle exception here
            pass
        finally:
            await self.load_balancing_client.mark_server_as_idle(server)

    @staticmethod
    def snowball_task_func(
        snowball_pricer,
        structure,
        underlyings,
        start_date,
        freq,
        periods,
        ki_barrier,
        ko_start,
        vanilla_strike2,
        vol,
    ):
        return snowball_pricer.solve(
            contract_type=structure,
            underlyings=underlyings,
            start=start_date,
            freq=freq,
            periods=periods,
            offset=0,
            barrier_direction="up_out",
            abs_rebate=0,
            rebate=0,
            ki_barrier=ki_barrier,
            ko_start=ko_start,
            ko_barrier_step_change=0,
            ko_barrier=None,
            vanilla_strike1=1,
            vanilla_strike2=vanilla_strike2,
            coupon=-999,  # doesn't matter
            # additional
            dividend=None,
            fixed_coupon=None,
            post_periods=None,
            coupon_incrementation=None,
            m_factor=None,
            p_factor=None,
            ki_barrier2=None,
            after_ko_participation=None,
            after_ko_strike=None,
            premium=None,
            low_coupon=None,
            # market_data
            vol=vol,
            rf=SNOWBALL_RF,
            credit_spread=SNOWBALL_CREDIT_SPREAD,
            repo_spread=SNOWBALL_REPO_SPREAD,
            spot=1,
            spot_date=None,
            fixings_date=None,
            fixings_date_value=None,
            model_date=None,
            model_date_fraction=0,
            result_params="npv",
        )

    async def close(self):
        await self.load_balancing_client.close()
