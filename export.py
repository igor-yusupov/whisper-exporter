import whisper
import torch
import torch.nn as nn

from src.encoder import AudioEncoder
from src.decoder import TextDecoder


class EncoderModel(nn.Module):
    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor):
        return self.encoder(x)


class DecoderModel(nn.Module):
    def __init__(self, decoder) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        tokens,
        audio_features,
        pos_emb,
        k1,
        v1,
        k2,
        v2,
        k3,
        v3,
        k4,
        v4,
        k5,
        v5,
        k6,
        v6,
    ):
        return self.decoder(
            tokens,
            audio_features,
            pos_emb,
            k1,
            v1,
            k2,
            v2,
            k3,
            v3,
            k4,
            v4,
            k5,
            v5,
            k6,
            v6,
        )


def export_encoder(model, model_name):
    model = EncoderModel(model).eval()

    x = torch.zeros((1, 80, 3000), dtype=torch.float32)
    input_names = ["mel"]

    torch.onnx.export(
        model,
        x,
        model_name,
        verbose=False,
        opset_version=14,
        input_names=input_names,
        # output_names=output_names,
        # dynamic_axes=dynamic_axes,
    )


def export_decoder(model, model_name):
    model = DecoderModel(model).eval()

    (
        tokens,
        audio_features,
        pos_emb,
        k1,
        v1,
        k2,
        v2,
        k3,
        v3,
        k4,
        v4,
        k5,
        v5,
        k6,
        v6,
    ) = (
        torch.zeros((1, 1), dtype=torch.int32),
        torch.rand((1, 1500, 512), dtype=torch.float32),
        torch.zeros((1, 1, 512), dtype=torch.float32),
        torch.zeros((1, 4, 512), dtype=torch.float32),
        torch.zeros((1, 4, 512), dtype=torch.float32),
        torch.zeros((1, 4, 512), dtype=torch.float32),
        torch.zeros((1, 4, 512), dtype=torch.float32),
        torch.zeros((1, 4, 512), dtype=torch.float32),
        torch.zeros((1, 4, 512), dtype=torch.float32),
        torch.zeros((1, 4, 512), dtype=torch.float32),
        torch.zeros((1, 4, 512), dtype=torch.float32),
        torch.zeros((1, 4, 512), dtype=torch.float32),
        torch.zeros((1, 4, 512), dtype=torch.float32),
        torch.zeros((1, 4, 512), dtype=torch.float32),
        torch.zeros((1, 4, 512), dtype=torch.float32),
    )
    input_names = [
        "tokens",
        "audio_features",
        "pos_emb",
        "k1",
        "v1",
        "k2",
        "v2",
        "k3",
        "v3",
        "k4",
        "v4",
        "k5",
        "v5",
        "k6",
        "v6",
    ]
    output_names = [
        "logits",
        "output_k1",
        "output_v1",
        "output_k2",
        "output_v2",
        "output_k3",
        "output_v3",
        "output_k4",
        "output_v4",
        "output_k5",
        "output_v5",
        "output_k6",
        "output_v6",
    ]

    dynamic_axes = {
        "tokens": {0: "batch_size", 1: "token_len"},
        "audio_features": {0: "batch_size"},
        "pos_emb": {0: "batch_size", 1: "token_len"},
        "k1": {0: "batch_size", 1: "offset_len"},
        "k2": {0: "batch_size", 1: "offset_len"},
        "k3": {0: "batch_size", 1: "offset_len"},
        "k4": {0: "batch_size", 1: "offset_len"},
        "k5": {0: "batch_size", 1: "offset_len"},
        "k6": {0: "batch_size", 1: "offset_len"},
        "v1": {0: "batch_size", 1: "offset_len"},
        "v2": {0: "batch_size", 1: "offset_len"},
        "v3": {0: "batch_size", 1: "offset_len"},
        "v4": {0: "batch_size", 1: "offset_len"},
        "v5": {0: "batch_size", 1: "offset_len"},
        "v6": {0: "batch_size", 1: "offset_len"},
        "logits": {0: "batch_size", 1: "token_len"},
        "output_k1": {1: "batch_size"},
        "output_k2": {1: "batch_size"},
        "output_k3": {1: "batch_size"},
        "output_k4": {1: "batch_size"},
        "output_k5": {1: "batch_size"},
        "output_k6": {1: "batch_size"},
        "output_v1": {1: "batch_size"},
        "output_v2": {1: "batch_size"},
        "output_v3": {1: "batch_size"},
        "output_v4": {1: "batch_size"},
        "output_v5": {1: "batch_size"},
        "output_v6": {1: "batch_size"},
    }

    torch.onnx.export(
        model,
        (
            tokens,
            audio_features,
            pos_emb,
            k1,
            v1,
            k2,
            v2,
            k3,
            v3,
            k4,
            v4,
            k5,
            v5,
            k6,
            v6,
        ),
        model_name,
        verbose=False,
        opset_version=14,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def main() -> None:
    model = whisper.load_model("base")
    model = model.eval()
    torch.save(model.encoder.state_dict(), "weights/encoder.pt")
    torch.save(model.decoder.state_dict(), "weights/decoder.pt")

    dims = model.dims

    encoder = AudioEncoder(
        n_mels=dims.n_mels,
        n_ctx=dims.n_audio_ctx,
        n_state=dims.n_audio_state,
        n_head=dims.n_audio_head,
        n_layer=dims.n_audio_layer,
    )
    encoder.load_state_dict(torch.load("weights/encoder.pt"))
    encoder = encoder.eval()

    decoder = TextDecoder(
        n_vocab=dims.n_vocab,
        n_ctx=dims.n_text_ctx,
        n_state=dims.n_text_state,
        n_head=dims.n_text_head,
        n_layer=dims.n_text_layer,
    )
    decoder.load_state_dict(torch.load("weights/decoder.pt"))
    decoder = decoder.eval()

    export_encoder(encoder, "weights/encoder.onnx")
    export_decoder(decoder, "weights/decoder.onnx")


if __name__ == "__main__":
    main()
