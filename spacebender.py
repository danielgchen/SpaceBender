"""
SpaceBender: spatially-aware ambient RNA removal for spatial transcriptomics (subcellular- and spot-resolution)
"""

from __future__ import annotations
import argparse
import logging
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from typing import Dict, List, Optional, Tuple
# import pytorch packages
import torch
import torch.nn as nn
from torch.distributions import constraints
# import pyro packages
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
# import cellbender functions
import SpaceBender.remove_background.consts as consts
from SpaceBender.remove_background.data.dataprep import DataLoader
from SpaceBender.remove_background.distributions.NegativeBinomialPoissonConvApprox import NegativeBinomialPoissonConvApprox as NBPCapprox
from SpaceBender.remove_background.vae.decoder import Decoder
from SpaceBender.remove_background.vae.encoder import CompositeEncoder, EncodeNonZLatents, EncodeZ


# instantiate logger
logger = logging.getLogger('spatial_cellbender')


# define helper functions
def compute_spatial_chi_ambient(
        count_matrix: sp.csr_matrix,
        spatial_coords: np.ndarray,
        k_neighbors: int = 6,
        radius: Optional[float] = None,
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    compute a (spatially) local ambient RNA profile.

    @param count_matrix: raw counts matrix (e.g. [obs x genes])
    @param spatial_coords: array of tissue coordinates (e.g. [x, y])
    @param k_neighbors: # of nearest-neighbors to use for averaging
    @param radius: alternative method to define nearest-neighbors by distance
    @param eps: value added before normalization to prevent division by zero errors

    @returns spatially-local ambient RNA profile
    """
    # retrieve the number of spot and genes
    n_spots, n_genes = count_matrix.shape
    # build a KD-tree for fast nearest-neighbor (nn) look-up
    tree = cKDTree(spatial_coords)

    # collect neighbor indices for every spot
    # > distance-based method for nn calculation
    if radius is not None:
        neighbor_lists = tree.query_ball_point(spatial_coords, r=radius)
        neighbor_lists = [
            [j for j in nbrs if j != i] or
            tree.query(spatial_coords[i], k=k_neighbors + 1)[1][1:].tolist()
            for i, nbrs in enumerate(neighbor_lists)
        ]
    # > kNN-based method for nn calculation
    else:
        _, indices = tree.query(spatial_coords, k=k_neighbors + 1)
        neighbor_lists = indices[:, 1:].tolist()

    # take the weighted average (by # of counts) for ambient RNA profile calculation
    count_matrix_dense = np.array(count_matrix.todense(), dtype=np.float32)  # (S, G)
    chi_ambient = np.zeros((n_spots, n_genes), dtype=np.float32)
    for s in range(n_spots):
        nbrs = neighbor_lists[s]
        if len(nbrs) == 0:
            # if a spot is isolated use the global mean
            nbrs = list(range(n_spots))
        neighbor_counts = count_matrix_dense[nbrs, :]    # (k, G)
        neighbor_totals = neighbor_counts.sum(axis=1)    # (k,)
        # weight neighbors by # of counts
        weights = neighbor_totals / (neighbor_totals.sum() + eps)  # (k,)
        local_profile = (weights[:, None] * neighbor_counts).sum(axis=0)  # (G,)
        row_sum = local_profile.sum()
        chi_ambient[s] = local_profile / (row_sum + eps)

    return torch.tensor(chi_ambient, dtype=torch.float32)


class SpatialTxDataset:
    """
    class for spatial transcriptomics data

    @param adata: anndata object with count matrix (.X),
                  in-tissue observation (.obs['in_tissue']),
                  and spatial coordinates (.obsm['spatial'])
    @param k_neighbors: # of neighbors to use for ambient RNA profile calculation
    @param ambient_radius: distance-based method to calculate ambient RNA profiles
    @param low_count_threshold: provided if user wants to threshold counts
                                (rarely used since we fall back on "in_tissue" labels)
    @param gene_blacklist: list of genes to exclude from analysis
    @param excluded_feature_types: list of feature types to exclude
    """

    def __init__(
            self,
            adata,
            k_neighbors: int = 6,
            ambient_radius: Optional[float] = None,
            low_count_threshold: int = 0,
            gene_blacklist: List[int] = [],
            excluded_feature_types: List[str] = [],
    ):
        # assign inputs to attributes
        self.adata = adata
        self.k_neighbors = k_neighbors
        self.ambient_radius = ambient_radius

        # process count matrix
        if sp.issparse(adata.X):
            matrix = adata.X.tocsr().astype(np.float32)
        else:
            matrix = sp.csr_matrix(adata.X.astype(np.float32))
        self.n_spots, self.n_genes_raw = matrix.shape

        # process metadata
        if 'in_tissue' not in adata.obs.columns:
            raise ValueError("adata.obs must contain an 'in_tissue' column.")
        self.in_tissue = adata.obs['in_tissue'].values.astype(bool)  # (S,)
        self.in_tissue_inds = np.where(self.in_tissue)[0]
        self.out_tissue_inds = np.where(~self.in_tissue)[0]

        # process spatial coordinates
        if 'spatial' not in adata.obsm:
            if 'array_row' in adata.obs.columns and 'array_col' in adata.obs.columns:
                self.spatial_coords = adata.obs[['array_row', 'array_col']].values
            else:
                raise ValueError(
                    "adata.obsm['spatial'] or adata.obs[['array_row','array_col']] is not found."
                )
        else:
            self.spatial_coords = adata.obsm['spatial']  # (S, 2)

        # perform gene filtering
        gene_logic = np.ones(self.n_genes_raw, dtype=bool)
        gene_counts = np.array(matrix.sum(axis=0)).squeeze()
        gene_logic &= gene_counts > 0
        if len(gene_blacklist) > 0:
            bl = set(gene_blacklist)
            bl_logic = np.array([i not in bl for i in range(self.n_genes_raw)])
            gene_logic &= bl_logic
        # exclude feature types of disinterest
        if len(excluded_feature_types) > 0 and 'feature_types' in adata.var.columns:
            feat = adata.var['feature_types'].values
            keep = np.array([f not in excluded_feature_types for f in feat])
            gene_logic &= keep
        self.analyzed_gene_inds = np.where(gene_logic)[0]
        self.n_genes = len(self.analyzed_gene_inds)
        logger.info(f"Retaining {self.n_genes} / {self.n_genes_raw} genes after filtering.")
        # perform actual filtering
        self.count_matrix = matrix[:, self.analyzed_gene_inds]  # (S, G)

        # calculate model priors
        self.priors = self._estimate_priors(low_count_threshold)
        # calculate spatial ambient RNA profiles
        if ambient_radius is not None:
            logger.info(f"Computing spatial ambient RNA profiles (radius={ambient_radius})...")
        else:
            logger.info(f"Computing spatial ambient RNA profiles (k={k_neighbors})...")
        self.chi_ambient_spatial = compute_spatial_chi_ambient(
            count_matrix=self.count_matrix,
            spatial_coords=self.spatial_coords,
            k_neighbors=k_neighbors,
            radius=ambient_radius,
        )

        # reassign gene names and barcodes
        self.gene_names = (adata.var_names.values
                           if hasattr(adata.var_names, 'values')
                           else np.array(adata.var_names.tolist()))
        self.barcodes = (adata.obs_names.values
                         if hasattr(adata.obs_names, 'values')
                         else np.array(adata.obs_names.tolist()))
        logger.info(f"SpatialTxDataset instantiated with {self.n_spots} spots and {self.n_genes} genes: {self.in_tissue_inds.size} observations in-tissue and {self.out_tissue_inds.size} observations out-of-tissue")

    def _estimate_priors(self, low_count_threshold: int) -> Dict:
        """
        estimate priors for the model

        @param low_count_threshold: provided if user wants to threshold counts
                                    (rarely used since we fall back on "in_tissue" labels)

        @returns dictionary of model priors
        """
        # compute UMI counts for each spot (in and out of tissue)
        umi = np.array(self.count_matrix.sum(axis=1)).squeeze()
        in_umi = umi[self.in_tissue_inds]
        out_umi = umi[self.out_tissue_inds]
        # take the mean of in-tissue UMI counts as cell count prior
        cell_counts = float(np.exp(np.mean(np.log1p(in_umi[in_umi > low_count_threshold]))))
        # compute the standard deviation of said counts (division by 5 is a heuristic from cellbender)
        log_in = np.log1p(in_umi[in_umi > low_count_threshold])
        d_std = float(np.std(log_in)) / 5.0
        # compute the median of out-of-tissue UMI for the empty prior
        if len(out_umi) > 0 and out_umi.max() > 0:
            empty_counts = float(np.median(out_umi[out_umi > low_count_threshold])
                                 if (out_umi > low_count_threshold).any()
                                 else np.median(out_umi))
        else:
            # if there are NOT any out-of-tissue spots use the 10th percentile of in-tissue counts
            empty_counts = float(np.percentile(in_umi, 10))
        # as per above, heuristics are taken from cellbender
        d_empty_std = 0.01
        log_counts_crossover = (np.log1p(empty_counts) + np.log1p(cell_counts)) / 2.0
        surely_empty_counts = empty_counts * 2.0

        # compute a global chi_ambient to be used as needed
        eps = np.finfo(np.float32).eps
        if len(self.out_tissue_inds) > 0:
            out_mat = np.array(self.count_matrix[self.out_tissue_inds, :].sum(axis=0)).squeeze()
        else:
            out_mat = np.array(self.count_matrix.sum(axis=0)).squeeze()
        chi_ambient_global = out_mat + eps
        chi_ambient_global = torch.tensor(
            chi_ambient_global / chi_ambient_global.sum(), dtype=torch.float32
        )
        chi_bar_vals = np.array(self.count_matrix.sum(axis=0)).squeeze() + eps
        chi_bar = torch.tensor(chi_bar_vals / chi_bar_vals.sum(), dtype=torch.float32)

        return {
            'cell_counts': cell_counts,
            'd_std': max(d_std, 0.01),
            'empty_counts': empty_counts,
            'd_empty_std': d_empty_std,
            'log_counts_crossover': log_counts_crossover,
            'surely_empty_counts': surely_empty_counts,
            'chi_ambient': chi_ambient_global,
            'chi_bar': chi_bar,
        }

    def get_dataloader(
            self,
            device: str = 'cpu',
            batch_size: int = 128,
            shuffle: bool = True,
            in_tissue_only: bool = False,
    ) -> DataLoader:
        """
        derive a dataloader for the spatial transcriptomics dataset

        @param device: where to compute and place data (e.g. cpu, cuda)
        @param batch_size: number of observations per batch
        @param shuffle: whether to shuffle the training data
        @param in_tissue_only: whether to use only in-tissue spots for training
                               (recommend this to be false to help the model learn w/ more info)
        """
        if in_tissue_only:
            mat = self.count_matrix[self.in_tissue_inds, :]
        else:
            mat = self.count_matrix

        return DataLoader(
            mat,
            empty_drop_dataset=None,
            batch_size=batch_size,
            fraction_empties=0.0,
            shuffle=shuffle,
            device=device,
        )


class SpatialRemoveBackgroundModel(nn.Module):
    """
    pyro model to remove amibient RNA in a spatially-aware manner

    @param encoder: CompositeEncoder instance
    @param decoder: Decoder instance
    @param priors: dictionary of model priors
    @param n_genes: # of genes
    @param n_spots: # of spots
    @param epsilon_prior: epsilon gamma prior
    @param phi_loc_prior: overdispersion prior (mean)
    @param phi_scale_prior: overdispersion prior (standard deviation)
    """

    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            priors: Dict,
            n_genes: int,
            n_spots: int,
            device: str = 'cpu',
            epsilon_prior: float = consts.EPSILON_PRIOR,
            phi_loc_prior: float = consts.PHI_LOC_PRIOR,
            phi_scale_prior: float = consts.PHI_SCALE_PRIOR,
    ):
        super().__init__()
        # set model attributes
        self.encoder = encoder
        self.decoder = decoder
        self.n_genes = n_genes
        self.n_spots = n_spots
        self.device = device
        self.z_dim = decoder.input_dim

        # move all parameters to the target device
        self.to(device)
        for v in self.encoder.values():
            v.to(device)
        def _t(x, dtype=torch.float32):
            return torch.tensor(x, dtype=dtype).to(self.device)

        # set model priors
        self.d_cell_loc_prior = _t(np.log1p(priors['cell_counts']))
        self.d_cell_scale_prior = _t(priors['d_std'])
        self.d_empty_loc_prior = _t(np.log1p(priors['empty_counts']))
        self.d_empty_scale_prior = _t(priors['d_empty_std'])
        self.z_loc_prior = torch.zeros(self.z_dim, device=self.device)
        self.z_scale_prior = torch.ones(self.z_dim, device=self.device)
        self.epsilon_prior = _t(epsilon_prior)
        phi_conc = (phi_loc_prior ** 2) / (phi_scale_prior ** 2)
        phi_rate = phi_loc_prior / (phi_scale_prior ** 2)
        self.phi_conc_prior = _t(phi_conc)
        self.phi_rate_prior = _t(phi_rate)
        self.phi_loc_prior_val = _t(phi_loc_prior)
        self.phi_scale_prior_val = _t(phi_scale_prior)

        # track losses during training and testing
        self.loss = {'train': {'epoch': [], 'elbo': []},
                     'test':  {'epoch': [], 'elbo': []},
                     'learning_rate': {'epoch': [], 'value': []}}

    def model(
            self,
            x: torch.Tensor,
            chi_ambient_batch: torch.Tensor,
            in_tissue_batch: torch.Tensor,
    ):
        """
        derive a pyro model for the given batch of data

        @param x: raw counts for the given set of observations
        @param chi_ambient_batch: spatially-aware ambient RNA profile (per observation)
        @param in_tissue_batch: indication for whether each spot is in- or out-of-tissue
        """
        pyro.module("decoder", self.decoder, update_module_params=True)
        B = x.shape[0]
        phi = pyro.sample(
            "phi",
            dist.Gamma(self.phi_conc_prior, self.phi_rate_prior)
        )
        alpha = phi.reciprocal()

        # compute per spot latents
        with pyro.plate("data", B):
            z = pyro.sample(
                "z",
                dist.Normal(self.z_loc_prior, self.z_scale_prior)
                .expand_by([B]).to_event(1)
            )
            chi = pyro.deterministic("chi", self.decoder(z))  # (B, G) on simplex

            # compute d_cell (per observation depth)
            d_cell = pyro.sample(
                "d_cell",
                dist.LogNormal(self.d_cell_loc_prior, self.d_cell_scale_prior)
                .expand_by([B])
            )
            # repeat for the background
            d_empty = pyro.sample(
                "d_empty",
                dist.LogNormal(self.d_empty_loc_prior, self.d_empty_scale_prior)
                .expand_by([B])
            )
            # compute capture efficiency
            epsilon = pyro.sample(
                "epsilon",
                dist.Gamma(self.epsilon_prior, self.epsilon_prior)
                .expand_by([B])
            )
            # compute the expected counts
            mu = (
                in_tissue_batch.unsqueeze(-1)
                * epsilon.unsqueeze(-1)
                * d_cell.unsqueeze(-1)
                * chi
            )
            # compute the noise rate
            lam = (
                epsilon.unsqueeze(-1)
                * d_empty.unsqueeze(-1)
                * chi_ambient_batch
            )

            # sample observations from the distribution
            pyro.sample(
                "obs",
                NBPCapprox(
                    mu=mu + consts.NBPC_MU_EPS_SAFEGAURD,
                    alpha=alpha + consts.NBPC_ALPHA_EPS_SAFEGAURD,
                    lam=lam + consts.NBPC_LAM_EPS_SAFEGAURD,
                ).to_event(1),
                obs=x,
            )

            # correct the data based on the ambient expression
            out_mask = (in_tissue_batch < 0.5)
            if out_mask.any().item():
                with poutine.mask(mask=out_mask):
                    with poutine.scale(scale=consts.REG_SCALE_AMBIENT_EXPRESSION):
                        lam_out = (
                            epsilon.unsqueeze(-1)
                            * d_empty.unsqueeze(-1)
                            * chi_ambient_batch
                        )
                        pyro.sample(
                            "obs_empty",
                            dist.Poisson(lam_out + consts.POISSON_EPS_SAFEGAURD).to_event(1),
                            obs=x,
                        )

        return {
            'mu': mu,
            'lam': lam,
            'alpha': alpha,
        }

    def guide(
            self,
            x: torch.Tensor,
            chi_ambient_batch: torch.Tensor,
            in_tissue_batch: torch.Tensor,
    ):
        """
        compute variational posterior

        @param x: raw counts for the given set of observations
        @param chi_ambient_batch: spatially-aware ambient RNA profile (per observation)
        @param in_tissue_batch: indication for whether each spot is in- or out-of-tissue
        """
        # register encoders
        for name, module in self.encoder.items():
            pyro.module("encoder_" + name, module, update_module_params=True)

        B = x.shape[0]

        # define model parameters
        phi_loc = pyro.param(
            "phi_loc",
            self.phi_loc_prior_val * torch.ones(1, device=self.device),
            constraint=constraints.positive,
        )
        phi_scale = pyro.param(
            "phi_scale",
            self.phi_scale_prior_val * torch.ones(1, device=self.device),
            constraint=constraints.positive,
        )
        phi_conc = phi_loc.pow(2) / phi_scale.pow(2)
        phi_rate = phi_loc / phi_scale.pow(2)
        pyro.sample("phi", dist.Gamma(phi_conc, phi_rate))
        d_cell_scale = pyro.param(
            "d_cell_scale",
            torch.tensor([consts.D_CELL_SCALE_INIT], device=self.device),
            constraint=constraints.positive,
        )
        d_empty_loc = pyro.param(
            "d_empty_loc",
            self.d_empty_loc_prior * torch.ones(torch.Size([]), device=self.device),
            constraint=constraints.positive,
        )
        d_empty_scale = pyro.param(
            "d_empty_scale",
            self.d_empty_scale_prior * torch.ones(torch.Size([]), device=self.device),
            constraint=constraints.positive,
        )

        # encode observations, not all sampling is necessary but may be helpful for future iterations
        with pyro.plate("data", B):
            enc = self.encoder(
                x=x,
                chi_ambient=None,
                cell_prior_log=self.d_cell_loc_prior,
            )
            d_empty = pyro.sample(
                "d_empty",
                dist.LogNormal(d_empty_loc, d_empty_scale).expand_by([B])
            )
            z = pyro.sample(
                "z",
                dist.Normal(enc['z']['loc'], enc['z']['scale']).to_event(1)
            )
            t = in_tissue_batch.detach()
            d_cell_loc_gated = (
                t * enc['d_loc'] + (1.0 - t) * self.d_cell_loc_prior
            )
            d_cell = pyro.sample(
                "d_cell",
                dist.LogNormal(d_cell_loc_gated, d_cell_scale)
            )
            eps_gated = t * enc['epsilon'] + (1.0 - t) * 1.0
            epsilon = pyro.sample(
                "epsilon",
                dist.Gamma(
                    eps_gated * self.epsilon_prior,
                    self.epsilon_prior,
                )
            )


def build_spatial_model(
        dataset: SpatialTxDataset,
        z_dim: int = 64,
        hidden_dims: List[int] = [512, 512],
        device: str = 'cpu',
) -> SpatialRemoveBackgroundModel:
    """
    overall script to build the spatial denoising model

    @param dataset: spatial transcriptomics dataset
    @param z_dim: number of latent dimensions
    @param hidden_dims: hidden layer sizes
    @param device: where to compute and place data (e.g. cpu, cuda)

    @returns SpatialRemoveBackgroundModel instance
    """
    n_genes = dataset.n_genes

    encoder_z = EncodeZ(
        input_dim=n_genes,
        hidden_dims=hidden_dims,
        output_dim=z_dim,
        input_transform='log',
    )
    encoder_other = EncodeNonZLatents(
        n_genes=n_genes,
        z_dim=z_dim,
        log_count_crossover=float(dataset.priors['log_counts_crossover']),
        prior_log_cell_counts=float(np.log1p(dataset.priors['cell_counts'])),
        empty_log_count_threshold=float(np.log1p(dataset.priors['surely_empty_counts'])),
        prior_logit_cell_prob=0.0,
        input_transform='log_normalize',
    )
    encoder = CompositeEncoder({'z': encoder_z, 'other': encoder_other})

    decoder = Decoder(
        input_dim=z_dim,
        hidden_dims=hidden_dims,
        output_dim=n_genes,
    )

    model = SpatialRemoveBackgroundModel(
        encoder=encoder,
        decoder=decoder,
        priors=dataset.priors,
        n_genes=n_genes,
        n_spots=dataset.n_spots,
        device=device,
    )

    return model


def train_spatial_model(
        model: SpatialRemoveBackgroundModel,
        dataset: SpatialTxDataset,
        n_epochs: int = 150,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        device: str = 'cpu',
        train_in_tissue_only: bool = False,
) -> SpatialRemoveBackgroundModel:
    """
    train the spatial denoising model

    @param model: SpatialRemoveBackgroundModel instance
    @param dataset: spatial transcriptomics dataset 
    @param n_epochs: number of training epochs
    @param batch_size: number of observations per batch
    @param learning_rate: learning rate during training
    @param device: where to compute and place data (e.g. cpu, cuda)
    @param train_in_tissue_only: whether to use only in-tissue spots for training
                                 (we do not recommend this, but provide it just in case)

    @returns trained SpatialRemoveBackgroundModel
    """
    # precompute ambient RNA profiles
    chi_amb_full = dataset.chi_ambient_spatial.to(device)
    in_tissue_full = torch.tensor(
        dataset.in_tissue.astype(np.float32), device=device
    )

    # define filters for training
    if train_in_tissue_only:
        train_inds = dataset.in_tissue_inds
        logger.info("Training on in-tissue spots only.")
    else:
        train_inds = np.arange(dataset.n_spots)
        logger.info("Training on all spots.")

    # prepare helper objects for training
    pyro.clear_param_store()
    optimizer = ClippedAdam({'lr': learning_rate, 'clip_norm': 10.0})
    elbo = Trace_ELBO()
    svi = SVI(model.model, model.guide, optimizer, loss=elbo)

    # train the model
    n_train = len(train_inds)
    count_matrix = dataset.count_matrix
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        perm = np.random.permutation(n_train)
        shuffled_inds = train_inds[perm]

        for start in range(0, n_train, batch_size):
            batch_global_inds = shuffled_inds[start:start + batch_size]
            # skip tiny little batches so they do not mess up normalizations
            if len(batch_global_inds) < 2:
                continue

            # perform training
            x_np = np.array(count_matrix[batch_global_inds, :].todense(),
                            dtype=np.float32)
            x_batch = torch.tensor(x_np, device=device)
            chi_batch = chi_amb_full[batch_global_inds, :]
            t_batch = in_tissue_full[batch_global_inds]
            loss = svi.step(
                x=x_batch,
                chi_ambient_batch=chi_batch,
                in_tissue_batch=t_batch,
            )
            epoch_loss += loss

        avg_loss = epoch_loss / n_train
        model.loss['train']['epoch'].append(epoch)
        model.loss['train']['elbo'].append(-avg_loss)

        # report the loss every few epochs
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:4d}/{n_epochs} | ELBO: {-avg_loss:.2f}")

    model.eval()
    return model


@torch.no_grad()
def sample_denoised_counts(
        model: SpatialRemoveBackgroundModel,
        dataset: SpatialTxDataset,
        n_samples: int = 25,
        batch_size: int = 64,
        device: str = 'cpu',
        lambda_multiplier: float = 1.0,
        use_map: bool = True,
) -> sp.csr_matrix:
    """
    sample denoised counts from the trained model

    @param model: trained SpatialRemoveBackgroundModel
    @param dataset: spatial transcriptomics dataset
    @param n_samples: number of posterior samples to draw
    @param batch_size: number of observations per batch
    @param device: where to compute and place data (e.g. cpu, cuda)
    @param lambda_multiplier: multiplier for the noise rate (lam) to utilize for denoising
    @param use_map: whether to use the MAP estimate of noise counts or noise-count distribution

    @returns denoised count matrix
    """
    # prepare model for evaluation mode
    model.eval()
    model.to(device)
    # gather parameter information
    chi_amb_full = dataset.chi_ambient_spatial.to(device)
    in_tissue_full = torch.tensor(
        dataset.in_tissue.astype(np.float32), device=device
    )
    count_matrix = dataset.count_matrix
    n_spots = dataset.n_spots
    n_genes = dataset.n_genes

    # assemble denoised matrix
    all_rows, all_cols, all_vals = [], [], []
    # encoder cannot process a batch of one so fold into the previous batch
    bounds = [(s, min(s + batch_size, n_spots)) for s in range(0, n_spots, batch_size)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < 2:
        bounds[-2] = (bounds[-2][0], bounds[-1][1])
        bounds.pop()
    for start, end in bounds:
        batch_inds = np.arange(start, end)
        B = len(batch_inds)
        x_np = np.array(count_matrix[batch_inds, :].todense(), dtype=np.float32)
        # skip computation if the batch is completely empty (cannot remove noise from nothing)
        if not x_np.any():
            logger.info(f"  Skipping empty batch: spots {start}-{end - 1} have no counts.")
            continue
        x_batch = torch.tensor(x_np, device=device)
        chi_batch = chi_amb_full[batch_inds, :]
        t_batch = in_tissue_full[batch_inds]
        log_noise_pdf_BGC = None
        noise_offset_BG = None

        for s in range(1, n_samples + 1):
            guide_trace = pyro.poutine.trace(model.guide).get_trace(
                x=x_batch,
                chi_ambient_batch=chi_batch,
                in_tissue_batch=t_batch,
            )
            replayed_model = pyro.poutine.replay(model.model, guide_trace)
            out = replayed_model(
                x=x_batch,
                chi_ambient_batch=chi_batch,
                in_tissue_batch=t_batch,
            )
            mu_s = out['mu']
            lam_s = out['lam']
            alpha_s = out['alpha']

            # estimate the noise
            log_probs_BGC, offset_BG = _log_prob_noise_count_tensor(
                data=x_batch,
                mu_est=mu_s + 1e-30,
                lam_est=lam_s * lambda_multiplier + 1e-30,
                alpha_est=alpha_s + 1e-30,
            )
            log_probs_BGC = log_probs_BGC - torch.logsumexp(
                log_probs_BGC, dim=-1, keepdim=True
            )
            if s == 1:
                log_noise_pdf_BGC = log_probs_BGC
                noise_offset_BG = offset_BG
            else:
                log_noise_pdf_BGC = torch.logaddexp(
                    log_noise_pdf_BGC
                    + torch.log(torch.tensor(1.0 - 1.0 / s, device=device)),
                    log_probs_BGC
                    + torch.log(torch.tensor(1.0 / s, device=device)),
                )
                noise_offset_BG = offset_BG

        # obtain noise counts
        if use_map:
            noise_count_increment_BG = log_noise_pdf_BGC.argmax(dim=-1)  # (B, G)
        else:
            noise_count_increment_BG = dist.Categorical(
                logits=log_noise_pdf_BGC
            ).sample()  # (B, G)
        noise_counts_BG = (noise_count_increment_BG + noise_offset_BG).long()  # (B, G)

        # denoise based on said noise counts
        denoised_BG = torch.clamp(x_batch - noise_counts_BG.float(), min=0.0)
        denoised_np = denoised_BG.cpu().numpy().astype(np.uint32)
        nz_spots, nz_genes = np.nonzero(denoised_np)
        all_rows.extend((nz_spots + start).tolist())
        all_cols.extend(nz_genes.tolist())
        all_vals.extend(denoised_np[nz_spots, nz_genes].tolist())

        if start % (batch_size * 10) == 0:
            logger.info(f"  Posterior sampling: {end}/{n_spots} spots done.")

    denoised_matrix = sp.csr_matrix(
        (np.array(all_vals, dtype=np.uint32),
         (np.array(all_rows, dtype=np.int32),
          np.array(all_cols, dtype=np.int32))),
        shape=(n_spots, n_genes),
    )
    logger.info(f"Denoised matrix: {denoised_matrix.nnz} non-zero entries, "
                f"shape {denoised_matrix.shape}.")
    return denoised_matrix


def _log_prob_noise_count_tensor(
        data: torch.Tensor,
        mu_est: torch.Tensor,
        lam_est: torch.Tensor,
        alpha_est: torch.Tensor,
        n_counts_max: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    compute the probability of a given noise count

    @param data: observed counts
    @param mu_est: estimated signal mean
    @param lam_est: estimated noise mean
    @param alpha_est: dispersion parameter
    @param n_counts_max: maximum count to use for noise

    @returns probability for a given noise counts
    """
    device = data.device
    n = min(n_counts_max, data.max().item())
    poisson_values_low = (lam_est.detach() - n / 2).int()
    poisson_values_low = torch.clamp(
        torch.min(poisson_values_low, (data - n + 1).int()),
        min=0,
    ).float()
    noise_count_tensor = (
        torch.arange(start=0, end=n, device=device)
        .expand([data.shape[0], data.shape[1], -1])
        .float()
    )
    noise_count_tensor = noise_count_tensor + poisson_values_low.unsqueeze(-1)
    logits = (mu_est.log() - alpha_est.log()).unsqueeze(-1)
    c_cell_clamped = torch.clamp(data.unsqueeze(-1) - noise_count_tensor, min=0)
    log_prob_tensor = (
        dist.Poisson(lam_est.unsqueeze(-1), validate_args=False)
        .log_prob(noise_count_tensor)
        + dist.NegativeBinomial(
            total_count=alpha_est.unsqueeze(-1),
            logits=logits,
            validate_args=False,
        ).log_prob(c_cell_clamped) 
    )
    neg_inf = torch.ones_like(log_prob_tensor) * -np.inf
    log_prob_tensor = torch.where(
        noise_count_tensor <= data.unsqueeze(-1),
        log_prob_tensor,
        neg_inf,
    )

    return log_prob_tensor, poisson_values_low

def write_spatial_output(
        dataset: SpatialTxDataset,
        denoised_matrix: sp.csr_matrix,
        output_file: str,
        latents: Optional[Dict] = None,
) -> None:
    """
    write the denoised output to an H5AD file

    @param dataset: spatial transcriptomics dataset
    @param denoised_matrix: denoised count matrix
    @param output_file: where to write the data
    @param latents: optional dictionary of latent variables to include in the output
    """
    # assemble the anndata object
    import anndata as ad
    raw_matrix = (dataset.adata.X.tocsr().astype(np.float32)
                  if sp.issparse(dataset.adata.X)
                  else sp.csr_matrix(dataset.adata.X.astype(np.float32)))

    output_full = raw_matrix.copy().tolil()
    output_full[:, dataset.analyzed_gene_inds] = denoised_matrix.tolil()
    output_full = output_full.tocsr()
    output_full = output_full.astype(np.uint32)
    obs = dataset.adata.obs.copy()
    var = dataset.adata.var.copy()
    var['analyzed'] = False
    var.iloc[dataset.analyzed_gene_inds, var.columns.get_loc('analyzed')] = True
    # add in the spatial and latents if needed
    obsm = {}
    if 'spatial' in dataset.adata.obsm:
        obsm['spatial'] = dataset.adata.obsm['spatial']
    if latents is not None and 'z' in latents:
        obsm['X_cellbender_z'] = latents['z']
    # perform the final assembly
    out_adata = ad.AnnData(
        X=output_full,
        obs=obs,
        var=var,
        obsm=obsm if obsm else None,
    )
    out_adata.obs.index = out_adata.obs.index.astype('object')
    out_adata.var.index = out_adata.var.index.astype('object')
    for col in ['gene_ids','feature_types','genome']:
        try: out_adata.var[col] = out_adata.var[col].astype('object')
        except: pass
    # write the data
    out_adata.write_h5ad(output_file)
    logger.info(f"Saved denoised output to {output_file}")


def run_spacebender(
        input_file: str,
        output_file: str,
        k_neighbors: int = 6,
        ambient_radius: Optional[float] = None,
        n_epochs: int = 150,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        z_dim: int = 64,
        hidden_dims: List[int] = [512, 512],
        n_posterior_samples: int = 25,
        device: str = 'cpu',
        train_in_tissue_only: bool = False,
        random_seed: int = 1234,
        lambda_multiplier: float = 1.0,
        use_map: bool = True,
) -> None:
    """
    pipeline to run spacebender

    @param input_file: path to the input H5AD
    @param output_file: path for where to write H5AD
    @param k_neighbors: # of neighbors for spatial RNA profile
    @param ambient_radius: distance to use for spatial RNA profile (if not using kNN)
    @param n_epochs: # of epochs to use for training
    @param batch_size: # of observations per batch for training
    @param learning_rate: learning rate for model training
    @param z_dim: # of latent dimensions for the model
    @param hidden_dims: sizes of the model hidden layers
    @param n_posterior_samples: # of posterior samples to draw for denoising
    @param device: where to compute and place data (e.g. cpu, cuda)
    @param train_in_tissue_only: whether to only use in-tissue spots for training (not recommended, but provided just in case)
    @param random_seed: random seed for reproducibility
    @param lambda_multiplier: balance factor for noise removal aggressivity
    @param use_map: whether to use the MAP estimate or noise-count distribution
    """
    # instantiate loggers and random seeds along with devices
    import anndata as ad
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    pyro.util.set_rng_seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    elif device == 'mps' and torch.backends.mps.is_available():
        torch.mps.manual_seed(random_seed)

    # read in data and build dataset
    logger.info(f"Loading data from {input_file}")
    adata = ad.read_h5ad(input_file)
    dataset = SpatialTxDataset(
        adata=adata,
        k_neighbors=k_neighbors,
        ambient_radius=ambient_radius,
    )

    # build and train model
    logger.info("Building SpatialRemoveBackgroundModel")
    model = build_spatial_model(
        dataset=dataset,
        z_dim=z_dim,
        hidden_dims=hidden_dims,
        device=device,
    )
    logger.info("Training")
    model = train_spatial_model(
        model=model,
        dataset=dataset,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        train_in_tissue_only=train_in_tissue_only,
    )

    # sample for denoised counts
    logger.info("Sampling denoised counts")
    denoised_matrix = sample_denoised_counts(
        model=model,
        dataset=dataset,
        n_samples=n_posterior_samples,
        batch_size=batch_size,
        device=device,
        lambda_multiplier=lambda_multiplier,
        use_map=use_map,
    )

    # write the denoised matrix
    logger.info("Writing output")
    write_spatial_output(
        dataset=dataset,
        denoised_matrix=denoised_matrix,
        output_file=output_file,
    )
    logger.info("SpaceBender finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SpaceBender: remove ambient RNA from ST data'
    )
    parser.add_argument('--input',  required=True, help='Input H5AD')
    parser.add_argument('--output', required=True, help='Output H5AD')
    parser.add_argument('--k-neighbors', type=int, default=6, help='# of neighbors for spatial ambient RNA profile (kNN-method)')
    parser.add_argument('--ambient-radius', type=float, default=None, help='Radius (in coordinate units) for spatial ambient RNA profile (distance-method)')
    parser.add_argument('--epochs', type=int, default=150, help="# of epochs for training")
    parser.add_argument('--batch-size', type=int, default=128, help="Batch size for training")
    parser.add_argument('--learning-rate', type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument('--z-dim', type=int, default=64, help="# of dimensions for the latent space")
    parser.add_argument('--n-posterior-samples', type=int, default=25, help="# of posterior samples to draw")
    parser.add_argument('--lambda-multiplier', type=float, default=1.0, help='Ambient noise rate scaling factor for posterior sampling')
    parser.add_argument('--use-sample', action='store_true', help='Draw a stochastic posterior sample rather than the MAP (argmax) estimate.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help="Where to perform computations")
    parser.add_argument('--train-in-tissue-only', action='store_true', help='Train on in tissue spots only (not recommended)')
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    run_spacebender(
        input_file=args.input,
        output_file=args.output,
        k_neighbors=args.k_neighbors,
        ambient_radius=args.ambient_radius,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        z_dim=args.z_dim,
        n_posterior_samples=args.n_posterior_samples,
        device=args.device,
        train_in_tissue_only=args.train_in_tissue_only,
        random_seed=args.seed,
        lambda_multiplier=args.lambda_multiplier,
        use_map=not args.use_sample,
    )
