import torch
import torch.nn.functional as F


class FacePatternCoordinator:
    """Derives facial feature set points from bioelectric activity."""

    def __init__(self, latticeDims, gene_names=None, device='cpu', dtype=torch.float64):
        self.num_rows, self.num_cols = latticeDims
        self.num_cells = self.num_rows * self.num_cols
        self.device = device
        self.dtype = dtype
        default_genes = ['rx', 'six3', 'pax6', 'lhx2', 'alx', 'dlx', 'hand2', 'runx2']
        self.gene_names = gene_names if gene_names is not None else default_genes
        self.feature_gene_map = {
            1: {'rx': 1.0, 'six3': 0.95, 'pax6': 0.95, 'lhx2': 0.9},   # eyes
            2: {'alx': 0.95},                                         # nose
            3: {'dlx': 0.95, 'hand2': 0.9},                           # jaw
            0: {'runx2': 0.95},                                       # bone
        }
        self.baseline = 0.05
        self.detail_gain = 1.5
        self._eps = 1e-6
        self._feature_pref_template = torch.tensor([0.5, 0.0, -0.5], dtype=self.dtype)
        self._build_coordinate_grids()
        self._precompute_templates()

    def _normalize(self, tensor):
        min_vals = tensor.amin(dim=(1, 2), keepdim=True)
        max_vals = tensor.amax(dim=(1, 2), keepdim=True)
        range_vals = torch.clamp(max_vals - min_vals, min=1e-9)
        return (tensor - min_vals) / range_vals

    def _build_coordinate_grids(self):
        row_axis = torch.linspace(0.0, 1.0, self.num_rows, dtype=self.dtype)
        col_axis = torch.linspace(0.0, 1.0, self.num_cols, dtype=self.dtype)
        self.row_grid = row_axis.view(self.num_rows, 1).expand(self.num_rows, self.num_cols)
        self.col_grid = col_axis.view(1, self.num_cols).expand(self.num_rows, self.num_cols)

    def _gaussian(self, center_row, center_col, sigma_row, sigma_col):
        row_term = ((self.row_grid - center_row) ** 2) / (2 * sigma_row ** 2)
        col_term = ((self.col_grid - center_col) ** 2) / (2 * sigma_col ** 2)
        return torch.exp(-(row_term + col_term))

    def _build_eye_template(self):
        left_eye = self._gaussian(0.32, 0.32, 0.06, 0.05)
        right_eye = self._gaussian(0.32, 0.68, 0.06, 0.05)
        return left_eye + right_eye

    def _build_nose_template(self):
        return self._gaussian(0.55, 0.5, 0.08, 0.04)

    def _build_jaw_template(self):
        mouth = self._gaussian(0.82, 0.5, 0.08, 0.2)
        chin = self._gaussian(0.9, 0.5, 0.05, 0.25)
        return mouth + 0.6 * chin

    def _precompute_templates(self):
        eye = self._build_eye_template()
        nose = self._build_nose_template()
        jaw = self._build_jaw_template()
        templates = torch.stack([eye, nose, jaw], dim=0)
        # normalization = templates.sum(0, keepdim=True) + self._eps
        normalization = templates.max(1,keepdim=True).values.max(2,keepdim=True).values
        self.base_templates = templates / normalization

    def _compute_detail(self, vmem_grid):
        vnorm = self._normalize(vmem_grid)
        blurred = F.avg_pool2d(vnorm.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        detail = vnorm - blurred
        denom = detail.abs().amax(dim=(1, 2), keepdim=True) + self._eps
        return detail / denom

    def _feature_preferences(self, device):
        return self._feature_pref_template.to(device)

    def _compute_feature_mask(self, vmem_grid):
        num_samples = vmem_grid.shape[0]
        device = vmem_grid.device
        detail = self._compute_detail(vmem_grid)  # shape (samples, rows, cols)
        templates = self.base_templates.to(device).unsqueeze(0).expand(num_samples, -1, -1, -1)

        pos_thresh = 0.35
        neg_thresh = -0.35
        template_thresh = 0.15

        eye_template = templates[:, 0]
        nose_template = templates[:, 1]
        jaw_template = templates[:, 2]

        eye_mask = (detail >= pos_thresh) & (eye_template > template_thresh)
        jaw_mask = (detail <= neg_thresh) & (jaw_template > template_thresh)
        nose_mask = (~eye_mask) & (~jaw_mask) & (nose_template > template_thresh / 2)

        feature_mask = torch.zeros_like(detail, dtype=self.dtype)
        feature_mask[eye_mask] = 1
        feature_mask[nose_mask] = 2
        feature_mask[jaw_mask] = 3

        return feature_mask

    def _build_gene_targets(self, feature_mask):
        num_samples = feature_mask.shape[0]
        grid_targets = torch.full(
            (num_samples, self.num_rows, self.num_cols, len(self.gene_names)),
            self.baseline,
            device=self.device,
            dtype=self.dtype,
        )
        for gene_idx, gene_name in enumerate(self.gene_names):
            gene_targets = grid_targets[..., gene_idx]
            for feature_id, gene_map in self.feature_gene_map.items():
                if gene_name not in gene_map:
                    continue
                target_value = gene_map[gene_name]
                mask = feature_mask == feature_id
                gene_targets = torch.where(mask, torch.full_like(gene_targets, target_value), gene_targets)
            grid_targets[..., gene_idx] = gene_targets
        return grid_targets

    def derive_set_point(self, vmem):
        """
        Args:
            vmem: Tensor of shape (numSamples, numCells, 1)

        Returns:
            dict with keys:
                'feature_mask' -> (numSamples, numCells, 1)
                'gene_targets' -> (numSamples, numCells * numGenes, 1)
        """
        if vmem.ndim != 3:
            raise ValueError("vmem must have shape (numSamples, numCells, 1)")
        num_samples, num_cells, _ = vmem.shape
        if num_cells != self.num_cells:
            raise ValueError(f"Expected {self.num_cells} cells, received {num_cells}")
        vmem_grid = vmem.view(num_samples, self.num_rows, self.num_cols)
        feature_mask = self._compute_feature_mask(vmem_grid)
        gene_targets_grid = self._build_gene_targets(feature_mask)

        feature_mask_flat = feature_mask.view(num_samples, num_cells, 1)
        gene_targets_flat = gene_targets_grid.view(num_samples, -1, 1)

        return {
            'feature_mask': feature_mask_flat,
            'feature_mask_grid': feature_mask,
            'gene_targets': gene_targets_flat,
            'gene_targets_grid': gene_targets_grid,
            'gene_names': self.gene_names,
        }
