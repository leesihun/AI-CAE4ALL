"""Paper-parity GINO decoder for the ShapeNet Car pressure benchmark.

This module is intentionally separate from :mod:`model.gino`.  The suite's
normal ``MeshGINO`` consumes mesh state through an input GNO.  The NeurIPS
2023 CarCFD comparison instead used the decoder half of GINO: a signed-
distance field already sampled on a regular grid is processed by an FNO and
then queried on the car surface by an output GNO.

The implementation below keeps that benchmark-specific contract explicit:

* ``graph.latent_sdf`` is ``[B, R, R, R, 1]`` and is concatenated with the
  three latent-grid coordinates before the FNO lifting layer;
* ``graph.pos`` contains surface queries normalized to ``[-1, 1]^3``;
* Fourier mode counts are *total centered modes* (not modes per FFT corner);
* the output GNO is the paper-era linear kernel transform, averaged over each
  query's latent-grid neighbors;
* no input-mesh quadrature weights are required because there is no input GNO.

It is opt-in via the future ``gino_variant=paper_decoder`` factory hook.  Merely
importing this file does not alter the default training or inference path.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from model.adapters.radius_neighbors import (
    HAS_TORCH_CLUSTER,
    neighbor_stats,
    radius_neighbors_scipy,
    radius_neighbors_torch_cluster,
)
from model.base import OperatorCore


def _triple(value, name: str) -> tuple[int, int, int]:
    if isinstance(value, (int, float)):
        value = [value] * 3
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values, got {value!r}.")
    return tuple(int(v) for v in value)


def _widths(value, default: Sequence[int]) -> tuple[int, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, (int, float)):
        return (int(value),)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"MLP widths must be a number or list, got {value!r}.")
    return tuple(int(v) for v in value)


def tucker_rank_from_fraction(
    shape: Sequence[int], fraction: float
) -> tuple[int, ...]:
    """Resolve an approximately equal Tucker compression fraction.

    The released NeuralOperator GINO uses ``rank=0.4``.  Tensorly resolves a
    scalar compression target by applying a shared scale to every dimension
    and accounting for both the Tucker core and factor matrices.  Keeping the
    resolver local avoids making the benchmark depend on tensorly/tltorch.
    """

    shape = tuple(int(s) for s in shape)
    if not shape or any(s < 1 for s in shape):
        raise ValueError(f"Tucker shape entries must be positive, got {shape}.")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError(f"Tucker rank fraction must be in (0, 1], got {fraction}.")

    dense = math.prod(shape)
    target = float(fraction) * dense

    order = len(shape)
    factor_coefficient = sum(s * s for s in shape)
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        continuous_params = dense * (mid ** order) + factor_coefficient * mid
        if continuous_params <= target:
            lo = mid
        else:
            hi = mid
    return tuple(max(1, min(s, int(round(s * lo)))) for s in shape)


def _complex_parameter(shape: Iterable[int], std: float) -> nn.Parameter:
    parameter = nn.Parameter(torch.empty(*tuple(shape), 2))
    with torch.no_grad():
        nn.init.normal_(parameter, mean=0.0, std=std)
    return parameter


def _as_complex(parameter: torch.Tensor) -> torch.Tensor:
    return torch.view_as_complex(parameter.contiguous())


class TuckerSpectralConv3d(nn.Module):
    """Centered 3-D rFFT convolution with a native complex Tucker weight.

    ``modes=(mx, my, mz)`` follows the paper/released NeuralOperator convention:
    ``mx`` and ``my`` are the total number of centered frequencies retained,
    while the real-FFT axis stores ``mz // 2 + 1`` coefficients.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Sequence[int],
        rank_fraction: float = 0.4,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = _triple(modes, "modes")
        if any(m < 1 for m in self.modes):
            raise ValueError(f"Fourier modes must be positive, got {self.modes}.")
        self.stored_modes = (self.modes[0], self.modes[1], self.modes[2] // 2 + 1)
        self.weight_shape = (
            self.in_channels,
            self.out_channels,
            *self.stored_modes,
        )
        self.rank_fraction = float(rank_fraction)
        self.rank = tucker_rank_from_fraction(self.weight_shape, self.rank_fraction)

        init_std = math.sqrt(2.0 / (self.in_channels + self.out_channels))
        rank_root = math.prod(math.sqrt(r) for r in self.rank)
        factor_std = (init_std / rank_root) ** (1.0 / (len(self.weight_shape) + 1))
        self.core = _complex_parameter(self.rank, factor_std)
        self.factors = nn.ParameterList(
            [_complex_parameter((size, rank), factor_std) for size, rank in zip(self.weight_shape, self.rank)]
        )
        if bias:
            self.bias = nn.Parameter(
                init_std * torch.randn(1, self.out_channels, 1, 1, 1)
            )
        else:
            self.register_parameter("bias", None)

    def reconstructed_weight(self) -> torch.Tensor:
        """Materialize the dense complex weight for tests and small diagnostics."""
        core = _as_complex(self.core)
        factors = [_as_complex(factor) for factor in self.factors]
        return torch.einsum(
            "abcde,ia,ob,xc,yd,ze->ioxyz", core, *factors
        )

    def contract_kept_modes(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the factorized weight without materializing its dense form."""
        core = _as_complex(self.core)
        factors = [_as_complex(factor) for factor in self.factors]
        return torch.einsum(
            "nixyz,abcde,ia,ob,xc,yd,ze->noxyz", x, core, *factors
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != self.in_channels:
            raise ValueError(
                "TuckerSpectralConv3d expects [B, C, X, Y, Z] with "
                f"C={self.in_channels}, got {tuple(x.shape)}."
            )
        spatial = tuple(int(v) for v in x.shape[-3:])
        if self.modes[0] > spatial[0] or self.modes[1] > spatial[1] or self.modes[2] > spatial[2]:
            raise ValueError(
                f"Centered Fourier modes {self.modes} exceed grid resolution {spatial}."
            )

        output_dtype = x.dtype
        spectrum = torch.fft.rfftn(x.float(), dim=(-3, -2, -1), norm="forward")
        spectrum = torch.fft.fftshift(spectrum, dim=(-3, -2))
        mx, my, mz = self.stored_modes
        x0 = (spatial[0] - mx) // 2
        y0 = (spatial[1] - my) // 2
        kept = spectrum[:, :, x0:x0 + mx, y0:y0 + my, :mz]
        transformed = self.contract_kept_modes(kept)

        output_spectrum = spectrum.new_zeros(
            (x.shape[0], self.out_channels, spatial[0], spatial[1], spatial[2] // 2 + 1)
        )
        output_spectrum[:, :, x0:x0 + mx, y0:y0 + my, :mz] = transformed
        output_spectrum = torch.fft.ifftshift(output_spectrum, dim=(-3, -2))
        output = torch.fft.irfftn(
            output_spectrum, s=spatial, dim=(-3, -2, -1), norm="forward"
        )
        if self.bias is not None:
            output = output + self.bias
        return output.to(dtype=output_dtype)


class NeRFSinusoidalEmbedding(nn.Module):
    """NeRF coordinate embedding used by maintained NeuralOperator GNOBlock.

    This mirrors ``SinusoidalEmbedding(..., embedding_type='nerf')`` exactly:
    no raw coordinates are appended and the flattened order interleaves sine
    and cosine for each coordinate/frequency pair.
    """

    def __init__(self, in_channels: int = 3, num_frequencies: int = 16):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_frequencies = int(num_frequencies)
        if self.in_channels < 1 or self.num_frequencies < 1:
            raise ValueError("NeRF embedding dimensions must be positive.")

    @property
    def out_channels(self) -> int:
        return 2 * self.num_frequencies * self.in_channels

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.ndim not in (2, 3) or coordinates.shape[-1] != self.in_channels:
            raise ValueError(
                f"Expected [..., N, {self.in_channels}] coordinates, got {tuple(coordinates.shape)}."
            )
        frequencies = (
            2 ** torch.arange(self.num_frequencies, device=coordinates.device)
        ) * torch.pi
        phases = torch.einsum("...ij,k->...ijk", coordinates, frequencies)
        return torch.stack((phases.sin(), phases.cos()), dim=-1).reshape(
            *coordinates.shape[:-1], self.out_channels
        )


class PaperEraPositionalEmbedding(nn.Module):
    """Coordinate embedding from the public August-2023 GINO implementation.

    The paper-era ``PositionalEmbedding`` flattens scalar coordinates, applies
    logarithmically spaced cosine/sine frequencies, and then reshapes the
    result per point.  ``num_channels=16`` therefore emits 48 values for a
    three-dimensional point, rather than the 96 values emitted by the later
    16-frequency NeRF embedding.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_channels: int = 16,
        max_positions: float = 10000.0,
        endpoint: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_channels = int(num_channels)
        self.max_positions = float(max_positions)
        self.endpoint = bool(endpoint)
        if self.in_channels < 1 or self.num_channels < 2 or self.num_channels % 2:
            raise ValueError("Paper-era positional embedding requires a positive even channel count.")
        if self.max_positions <= 1.0:
            raise ValueError("Paper-era max_positions must be greater than one.")

    @property
    def out_channels(self) -> int:
        return self.in_channels * self.num_channels

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.ndim not in (2, 3) or coordinates.shape[-1] != self.in_channels:
            raise ValueError(
                f"Expected [..., N, {self.in_channels}] coordinates, got {tuple(coordinates.shape)}."
            )
        half = self.num_channels // 2
        denominator = half - (1 if self.endpoint else 0)
        frequencies = torch.arange(
            half, device=coordinates.device, dtype=torch.float32
        ) / denominator
        frequencies = (1.0 / self.max_positions) ** frequencies
        flat = coordinates.reshape(-1, 1)
        phases = flat * frequencies.to(dtype=coordinates.dtype)
        embedded = torch.cat((phases.cos(), phases.sin()), dim=-1)
        return embedded.reshape(*coordinates.shape[:-1], self.out_channels)


class _ChannelMLP3d(nn.Module):
    def __init__(self, channels: int, expansion: float):
        super().__init__()
        inner = max(1, int(round(channels * expansion)))
        self.net = nn.Sequential(
            nn.Conv3d(channels, inner, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(inner, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SoftGating3d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x


class _PaperFNOBlock3d(nn.Module):
    """Post-activation maintained FNOBlocks semantics for one layer."""

    def __init__(
        self,
        channels: int,
        modes: Sequence[int],
        rank: float,
        expansion: float,
        is_final: bool = False,
    ):
        super().__init__()
        self.spectral = TuckerSpectralConv3d(channels, channels, modes, rank)
        self.fno_skip = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.channel_mlp_skip = _SoftGating3d(channels)
        # NeuralOperator's dimension-agnostic InstanceNorm calls functional
        # instance_norm with no weight/bias, so both norms are non-affine.
        self.norm_fno = nn.InstanceNorm3d(channels, affine=False, track_running_stats=False)
        self.norm_channel_mlp = nn.InstanceNorm3d(
            channels, affine=False, track_running_stats=False
        )
        self.channel_mlp = _ChannelMLP3d(channels, expansion)
        self.activation = nn.GELU()
        self.is_final = bool(is_final)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip_fno = self.fno_skip(x)
        x_skip_channel_mlp = self.channel_mlp_skip(x)
        h = self.norm_fno(self.spectral(x)) + x_skip_fno
        # The public paper-era FNO applies this GELU whenever the channel MLP
        # exists, including in the final block. Only the second GELU below is
        # omitted in the final block.
        h = self.activation(h)
        h = self.channel_mlp(h) + x_skip_channel_mlp
        h = self.norm_channel_mlp(h)
        if not self.is_final:
            h = self.activation(h)
        return h


def _build_mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = int(in_dim)
    for width in hidden:
        layers.extend((nn.Linear(current, int(width)), nn.GELU()))
        current = int(width)
    layers.append(nn.Linear(current, int(out_dim)))
    return nn.Sequential(*layers)


class LinearKernelIntegral3d(nn.Module):
    """Paper-era output GNO: neighbor mean of ``k(query, source) * feature``.

    The GINO appendix describes a constant decoder quadrature weight.  The
    public paper-era ``IntegralTransform`` implements that case as a Monte
    Carlo mean over the neighbors of each output query, not as division by all
    points in the latent grid.
    """

    def __init__(
        self,
        channels: int,
        kernel_widths: Sequence[int] = (512, 256),
        radius: float = 0.055,
        coordinate_embedding_dim: int = 16,
        coordinate_embedding_type: str = "nerf",
        use_torch_cluster: bool = True,
        max_num_neighbors: int = 512,
    ):
        super().__init__()
        self.channels = int(channels)
        self.radius = float(radius)
        self.use_torch_cluster = bool(use_torch_cluster)
        self.max_num_neighbors = int(max_num_neighbors)
        self.coordinate_embedding_dim = int(coordinate_embedding_dim)
        self.coordinate_embedding_type = str(coordinate_embedding_type).lower()
        if self.coordinate_embedding_type == "nerf":
            self.position_embedding = NeRFSinusoidalEmbedding(
                in_channels=3, num_frequencies=self.coordinate_embedding_dim
            )
        elif self.coordinate_embedding_type == "paper_2023":
            self.position_embedding = PaperEraPositionalEmbedding(
                in_channels=3, num_channels=self.coordinate_embedding_dim
            )
        else:
            raise ValueError(
                "coordinate_embedding_type must be 'nerf' or 'paper_2023', got "
                f"{coordinate_embedding_type!r}."
            )
        self.kernel = _build_mlp(
            2 * self.position_embedding.out_channels, kernel_widths, self.channels
        )

    def neighbors(self, queries: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
        if self.use_torch_cluster and HAS_TORCH_CLUSTER:
            return radius_neighbors_torch_cluster(
                queries,
                sources,
                self.radius,
                max_num_neighbors=self.max_num_neighbors,
            )
        edge_index = radius_neighbors_scipy(
            queries.detach().cpu().numpy(),
            sources.detach().cpu().numpy(),
            self.radius,
        )
        return torch.from_numpy(edge_index).to(device=queries.device)

    def forward(
        self,
        queries: torch.Tensor,
        sources: torch.Tensor,
        source_features: torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if queries.ndim != 2 or sources.ndim != 2 or queries.shape[1] != 3 or sources.shape[1] != 3:
            raise ValueError("GINO queries and sources must have shape [N, 3].")
        if source_features.shape != (sources.shape[0], self.channels):
            raise ValueError(
                "source_features must have shape "
                f"[{sources.shape[0]}, {self.channels}], got {tuple(source_features.shape)}."
            )
        if edge_index is None:
            edge_index = self.neighbors(queries, sources)

        output = source_features.new_zeros((queries.shape[0], self.channels))
        if edge_index.numel() == 0:
            return output
        query_index, source_index = edge_index[0], edge_index[1]
        query_embedding = self.position_embedding(queries)
        source_embedding = self.position_embedding(sources)
        # Maintained IntegralTransform concatenates y (integration sources)
        # first and x (output queries) second for its linear kernel.
        kernel_input = torch.cat(
            (source_embedding[source_index], query_embedding[query_index]), dim=-1
        )
        kernel = self.kernel(kernel_input)
        messages = kernel * source_features[source_index]
        output.index_add_(0, query_index, messages)
        counts = torch.bincount(
            query_index, minlength=queries.shape[0]
        ).to(device=output.device, dtype=output.dtype)
        output = output / counts.clamp_min(1).unsqueeze(-1)
        return output


def validate_carcfd_config(config, data_spec) -> None:
    if str(config.get("gino_variant", "")).lower() != "paper_decoder":
        raise ValueError("CarCFDGINODecoder requires gino_variant=paper_decoder.")
    if int(data_spec.operator_dim) != 3:
        raise ValueError("CarCFDGINODecoder requires a 3-D dataset.")
    if int(data_spec.output_var) != 1:
        raise ValueError("The CarCFD paper benchmark predicts one pressure channel.")
    resolution = _triple(config.get("gino_grid_resolution", 64), "gino_grid_resolution")
    modes = _triple(config.get("gino_fno_modes", 16), "gino_fno_modes")
    if any(r < 2 for r in resolution):
        raise ValueError(f"gino_grid_resolution entries must be >= 2, got {resolution}.")
    if any(m > r for m, r in zip(modes, resolution)):
        raise ValueError(f"gino_fno_modes {modes} exceed resolution {resolution}.")
    if bool(config.get("gino_use_torch_cluster", True)) and not HAS_TORCH_CLUSTER:
        raise ValueError("gino_use_torch_cluster=True but torch_cluster is not installed.")
    embedding_type = str(config.get("gino_pos_embedding_type", "nerf")).lower()
    if embedding_type not in {"nerf", "paper_2023"}:
        raise ValueError(
            "gino_pos_embedding_type must be 'nerf' or 'paper_2023'."
        )
    if int(config.get("gino_coord_embed_dim", 16)) < 1:
        raise ValueError("gino_coord_embed_dim must be positive.")
    if not bool(config.get("gino_include_grid_coordinates", True)):
        raise ValueError(
            "The paper-era decoder concatenates latent-grid coordinates with the SDF; "
            "gino_include_grid_coordinates must be True."
        )
    domain_padding = float(config.get("gino_domain_padding", 0.0))
    if not 0.0 <= domain_padding < 1.0:
        raise ValueError("gino_domain_padding must be in [0, 1).")


class CarCFDGINODecoder(OperatorCore):
    """Decoder-only GINO core used by the official ShapeNet Car split."""

    model_name = "gino"

    def __init__(self, config, data_spec, coordinate_domain=None):
        super().__init__()
        validate_carcfd_config(config, data_spec)
        self.data_spec = data_spec
        self.variant = "paper_decoder"
        self.output_var = int(data_spec.output_var)
        self.resolution = _triple(config.get("gino_grid_resolution", 64), "gino_grid_resolution")
        self.modes = _triple(config.get("gino_fno_modes", 16), "gino_fno_modes")
        self.hidden = int(config.get("gino_fno_hidden_channels", 64))
        self.n_layers = int(config.get("gino_fno_layers", 4))
        self.tucker_rank = float(config.get("gino_tucker_rank", 0.4))
        self.channel_mlp_expansion = float(config.get("gino_channel_mlp_expansion", 1.0))
        self.out_radius = float(config.get("gino_out_radius", 0.055))
        self.query_chunk_size = int(config.get("gino_query_chunk_size", 1024))
        self.use_torch_cluster = bool(config.get("gino_use_torch_cluster", True))
        self.max_num_neighbors = int(config.get("gino_max_num_neighbors", 512))
        self.pos_embedding_type = str(config.get("gino_pos_embedding_type", "nerf")).lower()
        self.coord_embed_dim = int(config.get("gino_coord_embed_dim", 16))
        self.domain_padding = float(config.get("gino_domain_padding", 0.0))
        self.include_grid_coordinates = bool(
            config.get("gino_include_grid_coordinates", True)
        )
        self.use_checkpointing = bool(config.get("use_checkpointing", True))
        self.kernel_widths = _widths(config.get("gino_kernel_widths"), (512, 256))
        self.projection_widths = _widths(config.get("gino_projection_widths"), (256,))

        self.lifting_hidden = int(
            config.get("gino_lifting_hidden", self.hidden * 2)
        )
        lifting_inputs = 1 + (3 if self.include_grid_coordinates else 0)
        self.lifting = nn.Sequential(
            nn.Conv3d(lifting_inputs, self.lifting_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(self.lifting_hidden, self.hidden, kernel_size=1),
        )
        self.fno_blocks = nn.ModuleList(
            [
                _PaperFNOBlock3d(
                    self.hidden,
                    self.modes,
                    self.tucker_rank,
                    self.channel_mlp_expansion,
                    is_final=layer_index == self.n_layers - 1,
                )
                for layer_index in range(self.n_layers)
            ]
        )
        self.output_gno = LinearKernelIntegral3d(
            channels=self.hidden,
            kernel_widths=self.kernel_widths,
            radius=self.out_radius,
            coordinate_embedding_dim=self.coord_embed_dim,
            coordinate_embedding_type=self.pos_embedding_type,
            use_torch_cluster=self.use_torch_cluster,
            max_num_neighbors=self.max_num_neighbors,
        )
        self.projection = _build_mlp(self.hidden, self.projection_widths, self.output_var)

        axes = [torch.linspace(-1.0, 1.0, r) for r in self.resolution]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        self.register_buffer("latent_points", grid.reshape(-1, 3).contiguous(), persistent=False)

    def _latent_sdf(self, graph) -> torch.Tensor:
        if not hasattr(graph, "latent_sdf"):
            raise ValueError(
                "gino_variant=paper_decoder requires graph.latent_sdf "
                "with shape [B, R, R, R, 1]."
            )
        sdf = graph.latent_sdf
        if sdf.ndim == 4 and sdf.shape[-1] == 1:
            sdf = sdf.unsqueeze(0)
        expected = (*self.resolution, 1)
        if sdf.ndim != 5 or tuple(sdf.shape[1:]) != expected:
            raise ValueError(
                f"graph.latent_sdf must have shape [B, {expected}], got {tuple(sdf.shape)}."
            )
        return sdf

    def _block(self, block: nn.Module, h: torch.Tensor) -> torch.Tensor:
        return block(h)

    def encode_operator(self, graph) -> torch.Tensor:
        sdf = self._latent_sdf(graph).permute(0, 4, 1, 2, 3).contiguous()
        if self.include_grid_coordinates:
            coordinate_grid = self.latent_points.reshape(*self.resolution, 3)
            coordinate_grid = coordinate_grid.permute(3, 0, 1, 2).unsqueeze(0)
            coordinate_grid = coordinate_grid.expand(sdf.shape[0], -1, -1, -1, -1)
            sdf = torch.cat((sdf, coordinate_grid.to(dtype=sdf.dtype)), dim=1)
        h = self.lifting(sdf)
        padding = tuple(round(self.domain_padding * size) for size in self.resolution)
        if any(padding):
            px, py, pz = padding
            h = F.pad(h, (0, pz, 0, py, 0, px), mode="constant")
        for block in self.fno_blocks:
            if self.use_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        if any(padding):
            rx, ry, rz = self.resolution
            h = h[..., :rx, :ry, :rz]
        return h

    def _decode_one(
        self, latent_grid: torch.Tensor, queries: torch.Tensor
    ) -> torch.Tensor:
        latent_features = latent_grid.reshape(self.hidden, -1).T.contiguous()
        chunks = []
        chunk_size = self.query_chunk_size if self.query_chunk_size > 0 else queries.shape[0]
        for start in range(0, queries.shape[0], max(1, chunk_size)):
            end = min(start + max(1, chunk_size), queries.shape[0])
            integral = self.output_gno(
                queries[start:end], self.latent_points, latent_features
            )
            chunks.append(self.projection(integral))
        if not chunks:
            return latent_features.new_empty((0, self.output_var))
        return torch.cat(chunks, dim=0)

    def decode_queries(self, encoded, graph, start: int, end: int):
        if not hasattr(graph, "ptr"):
            raise ValueError("CarCFDGINODecoder requires graph.ptr for ragged batches.")
        outputs = []
        for graph_index in range(graph.ptr.numel() - 1):
            graph_start = int(graph.ptr[graph_index])
            graph_end = int(graph.ptr[graph_index + 1])
            lo, hi = max(start, graph_start), min(end, graph_end)
            if lo < hi:
                outputs.append(self._decode_one(encoded[graph_index], graph.pos[lo:hi]))
        if not outputs:
            return encoded.new_empty((0, self.output_var))
        return torch.cat(outputs, dim=0)

    def forward(self, graph) -> torch.Tensor:
        if not hasattr(graph, "ptr"):
            raise ValueError("CarCFDGINODecoder requires graph.ptr for ragged batches.")
        encoded = self.encode_operator(graph)
        expected_graphs = graph.ptr.numel() - 1
        if encoded.shape[0] != expected_graphs:
            raise ValueError(
                f"latent_sdf batch has {encoded.shape[0]} samples but graph.ptr describes "
                f"{expected_graphs} graphs."
            )
        return self.decode_queries(encoded, graph, 0, int(graph.ptr[-1]))

    def supports_query_chunking(self) -> bool:
        return True

    def coverage_preflight(self, graph) -> dict:
        reports = []
        for graph_index in range(graph.ptr.numel() - 1):
            start, end = int(graph.ptr[graph_index]), int(graph.ptr[graph_index + 1])
            edges = self.output_gno.neighbors(graph.pos[start:end], self.latent_points)
            stats = neighbor_stats(edges.detach().cpu().numpy(), end - start)
            reports.append({"graph": graph_index, "output_gno": stats})
            if stats["empty_fraction"] > 0.0:
                raise ValueError(
                    f"GINO paper-decoder coverage failed on graph {graph_index}: "
                    f"{stats['empty_fraction']:.4f} of output queries have no latent-grid "
                    "neighbor. Increase gino_out_radius or the grid resolution."
                )
        return {"reports": reports}

    def export_model_config(self) -> dict:
        return {
            "model_name": self.model_name,
            "gino_variant": self.variant,
            "gino_grid_resolution": list(self.resolution),
            "gino_fno_modes": list(self.modes),
            "gino_fno_hidden_channels": self.hidden,
            "gino_fno_layers": self.n_layers,
            "gino_tucker_rank": self.tucker_rank,
            "gino_channel_mlp_expansion": self.channel_mlp_expansion,
            "gino_lifting_hidden": self.lifting_hidden,
            "gino_out_radius": self.out_radius,
            "gino_pos_embedding_type": self.pos_embedding_type,
            "gino_coord_embed_dim": self.coord_embed_dim,
            "gino_domain_padding": self.domain_padding,
            "gino_include_grid_coordinates": self.include_grid_coordinates,
            "gino_kernel_widths": list(self.kernel_widths),
            "gino_projection_widths": list(self.projection_widths),
            "gino_query_chunk_size": self.query_chunk_size,
            "gino_max_num_neighbors": self.max_num_neighbors,
            "gino_use_torch_cluster": self.use_torch_cluster,
            "use_checkpointing": self.use_checkpointing,
            "benchmark_target_mean_relative_l2": 0.0712,
            "paper_era_architecture_commit": "957f0b0fe540bf167f6138494297073d8aa97d98",
            "maintained_recipe_commit": "86a8bc7812a31b42c4f7895693cf4ac11521c066",
        }
