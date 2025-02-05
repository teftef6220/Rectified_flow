import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# パッチ分割/逆分割クラス
# -----------------------
class PatchEmbed(nn.Module):
    """
    画像をパッチに分割して、(B, 3, H, W) -> (B, N, embed_dim) に変換する。
    ここでは patch_size=4, image_size=32 を想定。
    """
    def __init__(self, in_channels=3, patch_size=4, embed_dim=128, image_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)

        # パッチをFlattenしたあとの次元: in_channels * patch_size * patch_size
        self.flat_dim = in_channels * (patch_size * patch_size)

        # Flattenした後、embed_dimに射影
        self.proj = nn.Linear(self.flat_dim, embed_dim)

    def forward(self, x):
        """
        x: (B, 3, 32, 32)
        return: (B, N, embed_dim)
        """
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B, C, H/patch, patch, W/patch, patch)
        #   -> (B, H/patch, W/patch, patch, patch, C)
        #   -> (B, N, patch*patch*C)
        #   -> (B, N, embed_dim)
        p = self.patch_size
        # パッチに分割
        x = x.unfold(2, p, p).unfold(3, p, p)  
        # x.shape = (B, C, H/p, W/p, p, p)

        # 軸を入れ替えて (B, H/p, W/p, C, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5)
        # 最後の3次元(C, p, p)をFlatten
        x = x.reshape(B, -1, self.flat_dim)  # (B, N, flat_dim)

        # 線形層で embed_dim に変換
        x = self.proj(x)  # (B, N, embed_dim)
        return x


class PatchUnembed(nn.Module):
    """
    パッチ列 (B, N, embed_dim) を空間的に並べ直して、(B, 3, H, W) に復元する。
    patch_size=4, image_size=32, embed_dim=128 を想定。
    """
    def __init__(self, out_channels=3, patch_size=4, embed_dim=128, image_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)

        # 一つのパッチに対応する Flatten 次元
        self.flat_dim = out_channels * (patch_size * patch_size)

        # embed_dim -> flat_dim に逆変換
        self.proj = nn.Linear(embed_dim, self.flat_dim)

    def forward(self, x):
        """
        x: (B, N, embed_dim)
        return: (B, out_channels, H, W)
        """
        B, N, _ = x.shape
        p = self.patch_size
        h_patches = self.image_size // p
        w_patches = self.image_size // p

        # embed_dim -> flat_dim
        x = self.proj(x)  # (B, N, flat_dim)
        # flat_dim = out_channels * p * p

        # パッチを (out_channels, p, p) にreshape
        x = x.reshape(B, N, self.out_channels, p, p)

        # N = h_patches * w_patches になるはず
        # (B, h_patches*w_patches, C, p, p) -> (B, C, h_patches, p, w_patches, p)
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, N, p, p)
        x = x.reshape(B, self.out_channels, h_patches, w_patches, p, p)
        # (B, C, h_patches, w_patches, p, p)

        # 最後の軸を結合して (H, W) = (h_patches * p, w_patches * p)
        x = x.permute(0, 1, 2, 4, 3, 5)  # (B, C, h_patches, p, w_patches, p)
        x = x.reshape(B, self.out_channels,
                      h_patches * p,
                      w_patches * p)  # (B, C, 32, 32)
        return x

# -----------------------
# 位置埋め込み用のクラス
# -----------------------
class LearnedPositionalEmbedding(nn.Module):
    """
    (B, N, d_model) 形状のトークン列に対して、(N, d_model) の学習可能埋め込みを加える。
    """
    def __init__(self, num_patches=64, d_model=128):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))

    def forward(self, x):
        """
        x: (B, N, d_model)
        """
        return x + self.pos_embed  # ブロードキャストで加算

# -----------------------
# SinusoidalTimestepEmbedding
# -----------------------
class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self,num_patches=64, embed_dim=128):
        super().__init__()
        self.max_period = 10000
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.silu = nn.SiLU(inplace=True)
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self,x):
        B, num_patches, _ = x.shape

        # Sinusoidal positional embeddings
        half = self.embed_dim // 2
        positions = torch.arange(num_patches).float().cuda()  # (num_patches,)
        freqs = torch.exp(-torch.arange(0, half).float().cuda() * torch.log(torch.tensor(self.max_period)) / half)
        sinusoid_args = positions.unsqueeze(1) * freqs.unsqueeze(0)
        sinusoidal_emb = torch.cat([torch.sin(sinusoid_args), torch.cos(sinusoid_args)], dim=-1)  # (num_patches, embed_dim)

        # Linear transformations
        sinusoidal_emb = sinusoidal_emb.unsqueeze(0).expand(B, -1, -1)  # (B, num_patches, embed_dim)
        x = x + sinusoidal_emb  # Add positional embeddings
        x = self.linear1(x)
        x = self.silu(x)
        x = self.linear2(x)
        return x


# -----------------------
# ViT Encoder
# -----------------------
class ViTEncoder(nn.Module):
    """
    画像をパッチに分割し、ViT Encoder を通して [B, 128, 8, 8] の形に戻す。
    """
    def __init__(
        self,
        in_channels=3,
        patch_size=4,
        embed_dim=128,     # パッチ埋め込み後の次元
        image_size=32,
        n_layers=4,
        n_heads=4,
        dim_feedforward=256,
        dropout=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, patch_size, embed_dim, image_size)
        # self.pos_embed = LearnedPositionalEmbedding(num_patches=(image_size//patch_size)**2,
        #                                             d_model=embed_dim)
        
        self.pos_embed = SinusoidalTimestepEmbedding((image_size//patch_size)**2,
                                                    embed_dim)

        # PyTorch 組み込みの TransformerEncoder を使用
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 入力を (B, N, d_model) とみなす
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size

    def forward(self, x):
        """
        x: (B, 3, 32, 32)
        return: (B, 128, 8, 8)
        """
        # 1) 画像 -> パッチ列 (B, N, embed_dim)
        x = self.patch_embed(x)  # (B, 64, 128)

        # 2) 位置埋め込みを付与
        x = self.pos_embed(x)    # (B, 64, 128)

        # 3) ViT Encoder
        x = self.transformer_encoder(x)  # (B, 64, 128)

        # 4) 空間に戻す (B, 128, 8, 8)
        B, N, D = x.shape
        # N = 8 x 8 = 64
        h = w = int((N)**0.5)  # 8
        x = x.permute(0, 2, 1).reshape(B, D, h, w)  # (B, 128, 8, 8)

        return x

# -----------------------
# Transformer Decoder
# -----------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoder(nn.Module):
    """
    入力: (B, in_embed_dim, 8, 8)  [例: (B, 32, 8, 8)]
    出力: (B, 3, 32, 32)

    内部ではまず 1x1 Conv で in_embed_dim -> embed_dim に変換し、
    その後に Transformer Decoder, patch_unembed で画像再構成します。
    """
    def __init__(
        self,
        out_channels=3,
        patch_size=4,
        # ここで「実際の入力チャネル数」を指定できるようにする
        in_embed_dim=32,  
        # Decoder 内部で使いたい埋め込み次元 (従来どおり128)
        embed_dim=128,
        image_size=32,
        n_layers=4,
        n_heads=4,
        dim_feedforward=256,
        dropout=0.1
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.image_size = image_size

        # --- 1) 1x1 Convで in_embed_dim -> embed_dim に変換 ---
        self.channel_proj = nn.Conv2d(in_embed_dim, embed_dim, kernel_size=1)

        # --- 2) 位置埋め込み ---
        # self.pos_embed = LearnedPositionalEmbedding(
        #     num_patches=(image_size//patch_size)**2,
        #     d_model=embed_dim
        # )
        self.pos_embed = SinusoidalTimestepEmbedding((image_size//patch_size)**2,
                                                    embed_dim)

        # --- 3) TransformerDecoder ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # (B, N, d_model)
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # --- 4) パッチ列 -> 画像に戻す ---
        self.patch_unembed = PatchUnembed(
            out_channels=out_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            image_size=image_size
        )

    def forward(self, x, memory=None):
        """
        x: (B, in_embed_dim, 8, 8)  (例: (64, 32, 8, 8))
        memory: (B, N_enc, embed_dim) もし Encoder-Decoder Attention を使うなら渡す
        """
        B, D, H, W = x.shape  # 例: (64, 32, 8, 8)

        # 1) 1x1 Convで D -> embed_dim (例: 32 -> 128)
        x = self.channel_proj(x)  # (B, 128, 8, 8)

        # 2) (B, 128, 8, 8) -> (B, 64, 128)
        x = x.reshape(B, self.embed_dim, H*W).permute(0, 2, 1)  # (B, 64, 128)

        # 3) 位置埋め込み
        x = self.pos_embed(x)  # (B, 64, 128)

        # 4) Transformer Decoder
        if memory is None:
            # デコーダ内部で自己注意のみ
            x = self.transformer_decoder(x, x)  # (B, 64, 128)
        else:
            # memory (B, N_enc, embed_dim) を参照 (Encoder-Decoder Attention)
            x = self.transformer_decoder(x, memory)  # (B, N_dec, embed_dim)

        # 5) パッチ列 -> 画像 (B, 3, 32, 32)
        x = self.patch_unembed(x)
        x = torch.sigmoid(x)
        return x


# -----------------------
# 例: 全体の簡単なテスト
# -----------------------
if __name__ == "__main__":
    B = 4  # バッチサイズ
    dummy_img = torch.randn(B, 3, 32, 32)

    # --- Encoder ---
    encoder = ViTEncoder(
        in_channels=3,
        patch_size=4,
        embed_dim=128,
        image_size=32,
        n_layers=2,      # 簡易化のため少なめ
        n_heads=4,
        dim_feedforward=256,
        dropout=0.1
    )
    enc_out = encoder(dummy_img)  # 期待形状: (B, 128, 8, 8)
    print("enc_out:", enc_out.shape)

    # --- Decoder ---
    decoder = TransformerDecoder(
        out_channels=3,
        patch_size=4,
        embed_dim=128,
        image_size=32,
        n_layers=2,
        n_heads=4,
        dim_feedforward=256,
        dropout=0.1
    )
    decoder_input = torch.randn(B, 32, 8, 8)
    dec_out = decoder(decoder_input)  # 期待形状: (B, 32, 8, 8)
    print("dec_out:", dec_out.shape)
