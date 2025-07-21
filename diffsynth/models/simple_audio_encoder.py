import torch
import torch.nn as nn

class SimpleAudioEncoder(nn.Module):
    """A minimal audio encoder projecting basic audio features
    to the text embedding dimension used in WanVideo.
    The input is expected to be a tensor of shape [L, F], where
    F is the number of audio features per time step (e.g. mean and std).
    """

    def __init__(self, feature_dim=2, output_dim=4096):
        super().__init__()
        self.proj = nn.Linear(feature_dim, output_dim)

    def forward(self, audio_feats: torch.Tensor) -> torch.Tensor:
        # audio_feats: [L, F] or [B, L, F]
        if audio_feats.dim() == 2:
            audio_feats = audio_feats.unsqueeze(0)
        return self.proj(audio_feats)
