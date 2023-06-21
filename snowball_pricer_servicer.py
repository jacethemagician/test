from client_server import pricer_pb2
from client_server import pricer_pb2_grpc
from server.base_pricer_servicer import BasePricerServicer
import json
import pandas as pd
from gleba.base.date_util import *
import numpy as np


class SnowballPricerServicer(
    pricer_pb2_grpc.SnowballPricerServicer, BasePricerServicer
):
    def __init__(self, logger):
        self._contract_type_func_map = {
            "general": self._create_general_snowball_json,
            "dividend": self._create_dividend_snowball_json,
            "phased": self._create_phased_snowball_json,
            "modified": self._create_modified_snowball_json,
            "trigger": self._create_trigger_snowball_json,
            "forward": self._create_forward_snowball_json,
            "phoenix": self._create_phoenix_snowball_json,
            "otm": self._create_otm_snowball_json,
            "kiko": self._create_kiko_snowball_json,
            "doubleki": self._create_doubleki_snowball_json,
            "lineargain": self._create_lineargain_snowball_json,
            "trinary": self._create_trinary_snowball_json,
            "binary": self._create_binary_snowball_json,
        }
        self._solver_func_map = {
            "general": self._create_regular_coupon_snowball_solver_json,
            "dividend": self._create_regular_coupon_snowball_solver_json,
            "phased": self._create_regular_coupon_snowball_solver_json,
            "modified": self._create_irregular_coupon_snowball_solver_json,
            "trigger": self._create_irregular_coupon_snowball_solver_json,
            "forward": self._create_irregular_coupon_snowball_solver_json,
            "phoenix": self._create_irregular_coupon_snowball_solver_json,
            "otm": self._create_irregular_coupon_snowball_solver_json,
            "kiko": self._create_irregular_coupon_snowball_solver_json,
            "doubleki": self._create_irregular_coupon_snowball_solver_json,
            "lineargain": self._create_irregular_coupon_snowball_solver_json,
            "trinary": self._create_irregular_coupon_snowball_solver_json,
            "binary": self._create_irregular_coupon_snowball_solver_json,
        }

        super().__init__(logger)

    def EvaluateSnowball(self, request, context):
        self.logger.info(f"Snowball Contract Type is: {request.contract_type}")
        contract_type_function = self._contract_type_func_map.get(request.contract_type)

        if contract_type_function is not None:
            return self._evaluate(
                request, contract_type_function, pricer_pb2.EvaluationOutput
            )
        else:
            raise ValueError(f"Unsupported contract type: {request.contract_type}")

    def SolveSnowball(self, request, context):
        self.logger.info(f"Snowball Contract Type is: {request.contract_type}")
        solver_function = self._solver_func_map.get(request.contract_type)

        if solver_function is not None:
            return self._solve(request, solver_function, pricer_pb2.SolverOutput)
        else:
            raise ValueError(f"Unsupported contract type: {request.contract_type}")

    def get_expiry_date(self, start, freq, periods):
        start = DateUtil.to_datetime(start)
        end_date = None
        if freq == DateFrequency.WEEKLY.value:
            end_date = (start + pd.DateOffset(weeks=periods)).date()
        elif freq == DateFrequency.BI_WEEKLY.value:
            end_date = (start + pd.DateOffset(weeks=2 * periods)).date()
        elif freq == DateFrequency.MONTHLY.value:
            end_date = (start + pd.DateOffset(months=periods)).date()
        elif freq == DateFrequency.BI_MONTHLY.value:
            end_date = (start + pd.DateOffset(months=2 * periods)).date()
        elif freq == DateFrequency.QUARTERLY.value:
            end_date = (start + pd.DateOffset(months=3 * periods)).date()
        elif freq == DateFrequency.SEMI_ANNUALLY.value:
            end_date = (start + pd.DateOffset(months=6 * periods)).date()
        elif freq == DateFrequency.ANNUALLY.value:
            end_date = (start + pd.DateOffset(years=periods)).date()

        if not DateUtil.is_business_day(end_date, Calendar.CN):
            end_date = DateUtil.next_business_day(end_date, Calendar.CN)

        return end_date

    def calculate_schedule(self, input_data):
        start = input_data.start
        freq = input_data.freq
        periods = input_data.periods
        end = self.get_expiry_date(start, freq, periods)
        return DateUtil.generate_schedule(
            start,
            end,
            freq,
            calendars=Calendar.CN,
            bdc=BDC.FOLLOWING,
            include_start=False,
            return_dates=True,
        )

    def calculate_N(self, schedule, offset):
        return len(schedule) - offset + 1

    def calculate_ko_barrier(self, input_data, N):
        if not input_data.ko_barrier:
            return list(
                np.ones(N) * input_data.ko_start
                + np.linspace(0, N - 1, N) * input_data.ko_barrier_step_change
            )
        else:
            return input_data.ko_barrier

    def calculate_r(self, rf, credit_spread):
        return rf + credit_spread

    def calculate_q(self, credit_spread, repo_spread):
        return credit_spread - repo_spread

    def calculate_multiplier(self, N):
        return list(np.ones(N + 1))

    def calculate_additor(self, N):
        return list(np.zeros(N + 1))

    def calculate_cp_nominal(self, coupon, multiplier, additor):
        return list(coupon * np.array(multiplier) + additor)

    def calculate_coupons(self, coupon, multiplier, additor, schedule, offset):
        cp_nominal = self.calculate_cp_nominal(coupon, multiplier, additor)
        ko_coupon = dict(
            zip(DateUtil.to_date_str(schedule[offset - 1 :]), cp_nominal[0:-1])
        )
        d_coupon = cp_nominal[-1]

        return ko_coupon, d_coupon

    def calculate_and_update_coupons(
        self, input_data, general_snowball_json, multiplier, additor, schedule
    ):
        ko_coupon, d_coupon = self.calculate_coupons(
            input_data.coupon, multiplier, additor, schedule, input_data.offset
        )

        general_snowball_json["contract"]["ko_coupon"] = ko_coupon
        general_snowball_json["contract"]["d_coupon"] = d_coupon

        return general_snowball_json

    def _create_general_snowball_json(self, input_data):
        schedule = self.calculate_schedule(input_data)
        N = self.calculate_N(schedule, input_data.offset)
        ko_barrier = self.calculate_ko_barrier(input_data, N)
        r = self.calculate_r(input_data.rf, input_data.credit_spread)
        q = self.calculate_q(input_data.credit_spread, input_data.repo_spread)
        multiplier = self.calculate_multiplier(N)
        additor = self.calculate_additor(N)
        cp_nominal = self.calculate_cp_nominal(input_data.coupon, multiplier, additor)

        if not input_data.spot_date:
            input_data.spot = 1
            input_data.spot_date = input_data.start
            input_data.model_date = input_data.start
            input_data.model_date_fraction = 0

        contract = {
            "currency": "CNY",
            "start_date": DateUtil.to_date_str(input_data.start),
            "tenor": "1Y",
            "expiry_date": DateUtil.to_date_str(schedule[-1]),
            "calendar": "CN",
            "quanto": False,
            "autocall_frequency": input_data.freq,
            "underlyings": input_data.underlyings,
            "autocall_barrier": dict(
                zip(DateUtil.to_date_str(schedule[input_data.offset - 1 :]), ko_barrier)
            ),
            "coupon_barrier": dict(
                zip(DateUtil.to_date_str(schedule[input_data.offset - 1 :]), ko_barrier)
            ),
            "is_memory_coupon": True,
            "ki_barrier": input_data.ki_barrier,
            "ko_coupon": dict(
                zip(
                    DateUtil.to_date_str(schedule[input_data.offset - 1 :]),
                    cp_nominal[0:-1],
                )
            ),
            "d_coupon": cp_nominal[-1],
            "principal": 0,
            "type": "templated",
            "template": "general_snowball",
            "abs_rebate": input_data.abs_rebate,
            "rebate": input_data.rebate,
            "day_counter": "act/365f",
            "barrier_type": input_data.barrier_direction,
            "autocall_schedule": DateUtil.to_date_str(schedule),
            "ko_coupon_is_annual": True,
            "d_coupon_is_annual": True,
            "autocall_start_offset": input_data.offset,
            "coupon_start_offset": input_data.offset,
            # "coupon_bearing_rule": "1,1", # initdeltacal 中没有
            "simple_coupon": True,  # initdeltacal都为True
        }

        if input_data.barrier_direction == "up_out":
            contract["gearing_put"] = 1
            contract["gearing_call"] = 0
            contract["ki_put_strike"] = input_data.vanilla_strike1
            contract["protected_principal"] = 1 - (
                input_data.vanilla_strike1 - input_data.vanilla_strike2
            )
        elif input_data.barrier_direction == "down_out":
            contract["gearing_put"] = 0
            contract["gearing_call"] = 1
            contract["ki_call_strike"] = input_data.vanilla_strike1
            contract["protected_principal"] = 1 - (
                input_data.vanilla_strike2 - input_data.vanilla_strike1
            )

        market_data = {
            "correlations": [],
            "model_date": DateUtil.to_date_str(input_data.model_date),
            "model_date_fraction": input_data.model_date_fraction,
            "products": {
                input_data.underlyings: {
                    "name": input_data.underlyings,
                    "currency": "CNY",
                    "spot": input_data.spot,
                    "spot_date": DateUtil.to_date_str(input_data.spot_date),
                    "type": "equity",
                    "borrow_rate": q,
                    "vol_surface": {
                        "type": "constant_vol_surface",
                        "value": input_data.vol,
                    },
                },
                "CNY": {
                    "name": "CNY",
                    "rate_curve": {
                        "type": "constant_rate_curve",
                        "value": r,
                        "compounding_frequency": "continuous",
                    },
                    "spot": 1,
                    "spot_date": DateUtil.to_date_str(input_data.spot_date),
                    "type": "currency",
                },
            },
            "fixings": {
                input_data.underlyings: {
                    input_data.fixings_date[i]: input_data.fixings_date_value[i]
                    for i in range(len(input_data.fixings_date))
                }
            },
        }

        configs = {
            "valuation_method": "monte_carlo",
            "num_paths": 65535,
            "rate_from_vol_time": True,
        }

        # 根据业务场景判断fixing模式
        if input_data.spot == 1 and input_data.start == input_data.spot_date:
            configs["fixing_mode"] = "post_fixing"
        else:
            configs["fixing_mode"] = "pre_fixing"

        greeks = list(input_data.result_params)

        type = "valuation"

        json_data = {
            "type": type,
            "contract": contract,
            "market_data": market_data,
            "greeks": greeks,
            "configs": configs,
        }

        return json.dumps(json_data, indent=2)

    ################################################
    ############## Contract Type ###################
    ################################################
    def _create_dividend_snowball_json(self, input_data):
        general_snowball_json = json.loads(
            self._create_general_snowball_json(input_data)
        )

        schedule = self.calculate_schedule(input_data)
        N = self.calculate_N(schedule, input_data.offset)
        multiplier = list(np.concatenate([np.ones(N), np.zeros(1)]))
        additor = list(np.concatenate([np.zeros(N), np.ones(1) * input_data.dividend]))
        snowball_json = self.calculate_and_update_coupons(
            input_data, general_snowball_json, multiplier, additor, schedule
        )

        return json.dumps(snowball_json, indent=2)

    def _create_phased_snowball_json(self, input_data):
        general_snowball_json = json.loads(
            self._create_general_snowball_json(input_data)
        )

        schedule = self.calculate_schedule(input_data)
        N = self.calculate_N(schedule, input_data.offset)
        multiplier = list(
            np.concatenate(
                [
                    np.ones(N - input_data.post_periods),
                    np.zeros(input_data.post_periods + 1),
                ]
            )
        )
        additor = list(
            np.concatenate(
                [
                    np.zeros(N - input_data.post_periods),
                    np.ones(input_data.post_periods) * input_data.fixed_coupon,
                    np.ones(1) * input_data.dividend,
                ]
            )
        )
        snowball_json = self.calculate_and_update_coupons(
            input_data, general_snowball_json, multiplier, additor, schedule
        )

        return json.dumps(snowball_json, indent=2)

    def _create_modified_snowball_json(self, input_data):
        general_snowball_json = json.loads(
            self._create_general_snowball_json(input_data)
        )
        schedule = self.calculate_schedule(input_data)
        N = self.calculate_N(schedule, input_data.offset)

        if input_data.m_factor and input_data.p_factor:
            multiplier = input_data.m_factor
            additor = input_data.p_factor
        else:
            multiplier = list(np.ones(N + 1))
            additor = list(
                np.concatenate(
                    [
                        np.linspace(0, N - 1, N) * input_data.coupon_incrementation,
                        np.ones(1) * (N - 1) * input_data.coupon_incrementation,
                    ]
                )
            )
        snowball_json = self.calculate_and_update_coupons(
            input_data, general_snowball_json, multiplier, additor, schedule
        )

        return json.dumps(snowball_json, indent=2), multiplier, additor

    def _create_trigger_snowball_json(self, input_data):
        modified_snowball, m_factor, p_factor = json.loads(
            self._create_modified_snowball_json(input_data)
        )
        modified_snowball_json = json.loads(modified_snowball)
        modified_snowball_json["contract"]["ko_coupon_is_annual"] = False
        modified_snowball_json["contract"]["d_coupon_is_annual"] = False
        modified_snowball_json["contract"]["coupon_bearing_rule"] = None
        modified_snowball_json["contract"]["is_memory_coupon"] = True
        modified_snowball_json["contract"]["simple_coupon"] = False

        return json.dumps(modified_snowball_json, indent=2), m_factor, p_factor

    def _create_forward_snowball_json(self, input_data):
        modified_snowball, m_factor, p_factor = json.loads(
            self._create_modified_snowball_json(input_data)
        )
        modified_snowball_json = json.loads(modified_snowball)
        del modified_snowball_json["contract"]["is_memory_coupon"]
        modified_snowball_json["contract"]["template"] = "snowball_ko_adjust"
        if input_data.barrier_direction == "up_out":
            modified_snowball_json["contract"]["autocall_barrier_postki"] = 99999
            modified_snowball_json["contract"]["gearing_call"] = (
                1 / input_data.vanilla_strike1
            )
            modified_snowball_json["contract"]["gearing_put"] = (
                1 / input_data.vanilla_strike1
            )
            modified_snowball_json["contract"]["protected_principal"] = (
                1
                - (input_data.vanilla_strike1 - input_data.vanilla_strike2)
                / input_data.vanilla_strike1
            )
            modified_snowball_json["contract"][
                "ki_call_strike"
            ] = modified_snowball_json["contract"]["ki_put_strike"]
        elif input_data.barrier_direction == "down_out":
            modified_snowball_json["contract"]["autocall_barrier_postki"] = 0
            modified_snowball_json["contract"]["gearing_call"] = (
                1 / input_data.vanilla_strike1
            )
            modified_snowball_json["contract"]["gearing_put"] = (
                1 / input_data.vanilla_strike1
            )
            modified_snowball_json["contract"]["protected_principal"] = (
                1
                - (input_data.vanilla_strike2 - input_data.vanilla_strike1)
                / input_data.vanilla_strike1
            )
            modified_snowball_json["contract"][
                "ki_put_strike"
            ] = modified_snowball_json["contract"]["ki_call_strike"]

        return json.dumps(modified_snowball_json, indent=2), m_factor, p_factor

    def _create_phoenix_snowball_json(self, input_data):
        general_snowball_json = json.loads(
            self._create_general_snowball_json(input_data)
        )
        schedule = self.calculate_schedule(input_data)
        general_snowball_json["contract"]["is_memory_coupon"] = False
        general_snowball_json["contract"]["coupon_start_offset"] = 1
        general_snowball_json["contract"]["coupon_bearing_rule"] = None
        general_snowball_json["contract"]["simple_coupon"] = True
        general_snowball_json["contract"]["d_coupon"] = None
        general_snowball_json["contract"]["coupon_barrier"] = dict(
            zip(
                DateUtil.to_date_str(schedule),
                list(np.ones(len(schedule)) * input_data.ki_barrier),
            )
        )
        if input_data.m_factor is not None and input_data.p_factor is not None:
            multiplier = input_data.m_factor
            additor = input_data.p_factor
        else:
            multiplier = list(np.ones(len(schedule) + 1))
            additor = list(
                np.concatenate(
                    [
                        np.linspace(0, len(schedule) - 1, len(schedule))
                        * input_data.coupon_incrementation,
                        np.ones(1)
                        * (len(schedule) - 1)
                        * input_data.coupon_incrementation,
                    ]
                )
            )
        cp_nominal = list(input_data.coupon * np.array(multiplier) + additor)
        general_snowball_json["contract"]["ko_coupon"] = dict(
            zip(DateUtil.to_date_str(schedule), cp_nominal[0:-1])
        )
        general_snowball_json["contract"]["d_coupon"] = cp_nominal[-1]
        return json.dumps(general_snowball_json, indent=2), multiplier, additor

    def _create_otm_snowball_json(self, input_data):
        modified_snowball, m_factor, p_factor = json.loads(
            self._create_modified_snowball_json(input_data)
        )
        modified_snowball_json = json.loads(modified_snowball)
        if input_data.barrier_direction == "up_out":
            modified_snowball_json["contract"]["gearing_put"] = (
                1 / input_data.vanilla_strike1
            )
            modified_snowball_json["contract"]["protected_principal"] = (
                1
                - (input_data.vanilla_strike1 - input_data.vanilla_strike2)
                / input_data.vanilla_strike1
            )
        elif input_data.barrier_direction == "down_out":
            modified_snowball_json["contract"]["gearing_call"] = (
                1 / input_data.vanilla_strike1
            )
            modified_snowball_json["contract"]["protected_principal"] = (
                1
                - (input_data.vanilla_strike2 - input_data.vanilla_strike1)
                / input_data.vanilla_strike1
            )

        return json.dumps(modified_snowball_json, indent=2), m_factor, p_factor

    def _create_kiko_snowball_json(self, input_data):
        modified_snowball, m_factor, p_factor = json.loads(
            self._create_modified_snowball_json(input_data)
        )
        modified_snowball_json = json.loads(modified_snowball)
        if input_data.barrier_direction == "up_out":
            modified_snowball_json["contract"]["gearing_put"] = (
                1 / input_data.vanilla_strike1
            )
            modified_snowball_json["contract"]["gearing_call"] = (
                1 / input_data.vanilla_strike1
            )
            modified_snowball_json["contract"]["protected_principal"] = (
                1
                - (input_data.vanilla_strike1 - input_data.vanilla_strike2)
                / input_data.vanilla_strike1
            )
            modified_snowball_json["contract"][
                "ki_call_strike"
            ] = modified_snowball_json["contract"]["ki_put_strike"]
        elif input_data.barrier_direction == "down_out":
            modified_snowball_json["contract"]["gearing_call"] = (
                1 / input_data.vanilla_strike1
            )
            modified_snowball_json["contract"]["gearing_put"] = (
                1 / input_data.vanilla_strike1
            )
            modified_snowball_json["contract"]["protected_principal"] = (
                1
                - (self.vanilla_strike2 - input_data.vanilla_strike1)
                / input_data.vanilla_strike1
            )
            modified_snowball_json["contract"][
                "ki_put_strike"
            ] = modified_snowball_json["contract"]["ki_call_strike"]

        return json.dumps(modified_snowball_json, indent=2), m_factor, p_factor

    def _create_doubleki_snowball_json(self, input_data):
        modified_snowball, m_factor, p_factor = json.loads(
            self._create_modified_snowball_json(input_data)
        )
        modified_snowball_json = json.loads(modified_snowball)
        modified_snowball_json["contract"]["ki_barrier2"] = input_data.ki_barrier2

        return json.dumps(modified_snowball_json, indent=2), m_factor, p_factor

    def _create_lineargain_snowball_json(self, input_data):
        modified_snowball, m_factor, p_factor = json.loads(
            self._create_modified_snowball_json(input_data)
        )
        modified_snowball_json = json.loads(modified_snowball)
        if input_data.barrier_direction == "up_out":
            modified_snowball_json["contract"]["payoff_at_autocall"] = "call"
            modified_snowball_json["contract"][
                "after_ko_participation"
            ] = input_data.after_ko_participation
            modified_snowball_json["contract"][
                "after_ko_strike"
            ] = input_data.after_ko_strike
        elif input_data.barrier_direction == "down_out":
            modified_snowball_json["contract"]["payoff_at_autocall"] = "put"
            modified_snowball_json["contract"][
                "after_ko_participation"
            ] = input_data.after_ko_participation
            modified_snowball_json["contract"][
                "after_ko_strike"
            ] = input_data.after_ko_strike

        return json.dumps(modified_snowball_json, indent=2), m_factor, p_factor

    def _create_trinary_snowball_json(self, input_data):
        modified_snowball, m_factor, p_factor = self._create_modified_snowball_json(
            input_data
        )
        modified_snowball_json = json.loads(modified_snowball)
        modified_snowball_json["contract"]["backend_premium"] = input_data.premium
        modified_snowball_json["contract"]["low_coupon"] = input_data.low_coupon
        modified_snowball_json["contract"]["protected_principal"] = 1

        return json.dumps(modified_snowball_json, indent=2), m_factor, p_factor

    def _create_binary_snowball_json(self, input_data):
        modified_snowball, m_factor, p_factor = self._create_modified_snowball_json(
            input_data
        )
        modified_snowball_json = json.loads(modified_snowball)
        modified_snowball_json["contract"]["backend_premium"] = input_data.premium
        modified_snowball_json["contract"]["low_coupon"] = input_data.low_coupon
        modified_snowball_json["contract"]["protected_principal"] = 1
        if input_data.barrier_direction == "up_out":
            modified_snowball_json["contract"]["ki_barrier"] = 99999
        elif input_data.barrier_direction == "down_out":
            modified_snowball_json["contract"]["ki_barrier"] = 0

        return json.dumps(modified_snowball_json, indent=2)

    def _create_solver_json(self, contract_type_func, input_data, solver_type):
        result, m_factor, p_factor = contract_type_func(input_data)
        json_dict = json.loads(result)

        if solver_type == "regular":
            json_dict["contract"]["contract_type"] = "snowball"
        elif solver_type == "irregular":
            json_dict["contract"]["contract_type"] = "snowball_irregular_coupon"

        json_dict["type"] = "solve_contract"
        json_dict["field"] = "ko_coupon"
        json_dict["result_type"] = "npv"
        json_dict["target_result"] = 0
        json_dict["init_guess"] = 0.2
        json_dict["solve_method"] = "newton"
        json_dict["params"] = {}

        if solver_type == "irregular":
            json_dict["params"]["m_factor"] = m_factor
            json_dict["params"]["p_factor"] = p_factor

        return json.dumps(json_dict, indent=2)

    def _create_regular_coupon_snowball_solver_json(self, input_data):
        return self._create_solver_json(
            self._contract_type_func_map[input_data.contract_type],
            input_data,
            "regular",
        )

    def _create_irregular_coupon_snowball_solver_json(self, input_data):
        return self._create_solver_json(
            self._contract_type_func_map[input_data.contract_type],
            input_data,
            "irregular",
        )
