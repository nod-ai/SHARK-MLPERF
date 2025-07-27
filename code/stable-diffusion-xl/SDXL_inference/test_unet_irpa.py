from shortfin.python.shortfin_apps.sd.exports import export_sdxl_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hf_model_name", type=str, required=True, help="Hugging Face model name")
parser.add_argument("--component", type=str, default="scheduled_unet", help="Model component to export (e.g., 'unet', 'vae', 'text_encoder')")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for export")
parser.add_argument("--height", type=int, default=1024, help="Height of the model's input image")
parser.add_argument("--width", type=int, default=1024, help="Width of the model's input image")
parser.add_argument("--precision", type=str, default="fp8_ocp", help="Precision for export")
parser.add_argument("--max_length", type=int, default=64, help="Max length for text encoder")
parser.add_argument("--punet_irpa_path", type=str, required=True, help="Path for punet IRPA")

args = parser.parse_args()

res = export_sdxl_model(
    hf_model_name=args.hf_model_name,
    component=args.component,
    batch_size=args.batch_size,
    height=args.height,
    width=args.width,
    precision=args.precision,
    max_length=args.max_length,
    external_weights="irpa",  # Hardcoded
    external_weights_file="./model.irpa",  # Hardcoded
    decomp_attn=False,  # Hardcoded
    quant_path=None,  # Hardcoded
    scheduler_config_path=args.hf_model_name, # Using hf_model_name
    weights_only=False,  # Hardcoded
    punet_irpa_path=args.punet_irpa_path,
)

breakpoint()