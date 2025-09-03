import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from src.model import MambaLiteSR


def export_to_onnx(model, output_path, size=(256, 256)):
    dummy_input = torch.randn(1, 3, *size)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        },
    )


def export_to_torchscript(model, output_path):
    scripted_model = torch.jit.script(model)
    optimized_model = optimize_for_mobile(scripted_model)
    optimized_model._save_for_lite_interpreter(output_path)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    model = MambaLiteSR(scale=4).to(device)
    model.load_state_dict(torch.load("runs/latest/student/best.pt"))
    model.eval()

    # Export formats
    export_to_onnx(model, "mambalitesr.onnx")
    export_to_torchscript(model, "mambalitesr.ptl")

    print("Models exported successfully!")
