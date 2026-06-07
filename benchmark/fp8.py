from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "../models/qwen-1.5b-merged"
quantized_model_dir = "../models/qwen-1.5b-auto-fp8"

quantize_config = BaseQuantizeConfig(
    quant_method="fp8",
    activation_scheme="dynamic"
)

model = AutoFP8ForCausalLM.from_pretrained(
    pretrained_model_dir,
    quantize_config
)

model.quantize()
model.save_quantized(quantized_model_dir)
