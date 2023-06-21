import json
import time
from functools import partial

from model.gleba0612.gleba.serializers.op_serializer import (
    OpSerializer,
)


class BasePricerServicer:
    def __init__(self, logger):
        self.logger = logger

    def _handle_request(
        self, input_data, create_input_json_callback, output_data_class, request_type
    ):
        start_time = time.time()
        self.logger.info(f"Message received from client!")
        input_json = create_input_json_callback(input_data)
        model_output_dict = self._run_gleba_model(input_json)
        self._log_input_output(input_json, model_output_dict)

        if request_type == "evaluate":
            result = self._process_eval_result(
                input_data.underlyings,
                input_data.result_params,
                model_output_dict["results"],
            )
        elif request_type == "solve":
            if hasattr(input_data, "field"):
                result = self._process_solver_result(
                    input_data.field, model_output_dict["results"]
                )
            else:
                # snowball solver
                # 2023-06-19 supports ko_coupon only
                result = model_output_dict["results"]["ko_coupon"]

        output_data = output_data_class(result=result)

        end_time = time.time()
        self.logger.info("Message sent to client!")
        self.logger.info("Pricer Time: %s", end_time - start_time)
        return output_data

    def _run_gleba_model(self, input_json):
        ops = OpSerializer.read_json(input_json)
        ops_output = ops.run()
        ops_output_json = ops_output.to_json()
        return json.loads(ops_output_json)

    def _evaluate(self, input_data, create_input_json_callback, output_data_class):
        return self._handle_request(
            input_data,
            create_input_json_callback,
            output_data_class,
            request_type="evaluate",
        )

    def _solve(self, input_data, create_input_json_callback, output_data_class):
        return self._handle_request(
            input_data,
            create_input_json_callback,
            output_data_class,
            request_type="solve",
        )

    def _process_eval_result(self, underlyings, result_params, result_dict):
        result_param_map = {
            "npv": "npv",
            "delta": f"delta[{underlyings}]",
            "delta_cash": f"delta_cash[{underlyings}]",
            "gamma": f"gamma[{underlyings}]",
            "gamma_cash": f"gamma_cash[{underlyings}]",
            "theta": "theta",
            "vega": f"vega[{underlyings}]",
            "rho": "rho[CNY]",
        }
        return [
            result_dict[result_param_map[result_param]]
            for result_param in result_params
            if result_param in result_param_map
        ]

    def _process_solver_result(self, field, result_dict):
        # Note: gleba might return something like:
        # Tolerance of -0.0031264365082823986 reached.
        # Failed to converge after 45 iterations, value is 0.2197575122171585.
        if isinstance(result_dict, str):
            return -999

        return result_dict[field]

    # ==========================================
    # ========= Other Helper Functions =========
    # ==========================================
    def _log_input_output(self, input_json, output_dict):
        self.logger.info(
            "\n====================Input====================\n%s",
            input_json,
        )
        output_json = json.dumps(output_dict, indent=2)
        self.logger.info(
            "\n====================Output====================\n%s", output_json
        )
