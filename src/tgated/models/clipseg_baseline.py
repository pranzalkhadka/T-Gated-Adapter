import torch.nn as nn
from transformers import CLIPSegForImageSegmentation


class CLIPSegBaseline(nn.Module):
    def __init__(self, freeze_vision_layers=4):
        super().__init__()
        self.base_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

        vision_encoder = self.base_model.clip.vision_model.encoder
        for layer_idx, layer in enumerate(vision_encoder.layers):
            for p in layer.parameters():
                p.requires_grad_(layer_idx >= freeze_vision_layers)

        self.base_model.clip.vision_model.embeddings.requires_grad_(True)
        self.base_model.clip.text_model.requires_grad_(True)
        self.base_model.decoder.requires_grad_(True)

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.base_model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        return logits
