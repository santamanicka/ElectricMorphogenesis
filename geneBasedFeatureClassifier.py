"""
Gene-Based Feature Classifier
Classifies facial features (eye, nose, mouth, bone) based on gene expression levels ONLY.
NO voltage thresholds - features emerge purely from gene combinations.
"""

import torch
import torch.nn.functional as F


class GeneBasedFeatureClassifier:
    """
    Classifies facial features from gene expression patterns.

    Features are determined by gene combinations:
    - Eye: high(pax6 × lhx2)
    - Nose: high(alx)
    - Mouth: high(dlx × hand2)
    - Bone: high(runx2) OR low(all others)
    """

    def __init__(self, grid_size, gene_names=None, device='cpu', dtype=torch.float64):
        """
        Args:
            grid_size: Size of spatial grid (assumes square)
            gene_names: List of gene names (default: facial genes)
            device: 'cpu' or 'cuda'
            dtype: torch data type
        """
        self.grid_size = grid_size
        self.device = device
        self.dtype = dtype

        # Default facial gene battery
        if gene_names is None:
            self.gene_names = ['rx', 'six3', 'pax6', 'lhx2', 'alx', 'dlx', 'hand2', 'runx2']
        else:
            self.gene_names = gene_names

        # Feature labels
        self.feature_names = ['bone', 'eye', 'nose', 'mouth']
        self.num_features = len(self.feature_names)

        # Softmax temperature for probabilistic classification
        # Lower temperature → sharper boundaries
        # Higher temperature → softer boundaries
        self.temperature = torch.tensor(0.5, device=device, dtype=dtype)

        # Minimum expression thresholds (prevent noise from being classified as features)
        # LOWER thresholds to allow morphogen-driven patterns to emerge
        # MUCH HIGHER mouth threshold to create narrow horizontal stripe matching IdealFace.png
        self.min_eye_expr = 0.30
        self.min_nose_expr = 0.10
        self.min_mouth_expr = 0.85  # Raised from 0.30 to 0.85 for small horizontal mouth stripe
        self.min_bone_expr = 0.20

    def compute_feature_scores(self, gene_grids):
        """
        Compute feature scores from gene expression.

        Args:
            gene_grids: dict with keys = gene names, values = (grid_size, grid_size) tensors

        Returns:
            dict with keys:
                'eye_score': (grid_size, grid_size)
                'nose_score': (grid_size, grid_size)
                'mouth_score': (grid_size, grid_size)
                'bone_score': (grid_size, grid_size)
        """
        # Extract genes
        pax6 = gene_grids.get('pax6', torch.zeros(self.grid_size, self.grid_size, device=self.device, dtype=self.dtype))
        lhx2 = gene_grids.get('lhx2', torch.zeros_like(pax6))
        six3 = gene_grids.get('six3', torch.zeros_like(pax6))
        rx = gene_grids.get('rx', torch.zeros_like(pax6))
        alx = gene_grids.get('alx', torch.zeros_like(pax6))
        dlx = gene_grids.get('dlx', torch.zeros_like(pax6))
        hand2 = gene_grids.get('hand2', torch.zeros_like(pax6))
        runx2 = gene_grids.get('runx2', torch.zeros_like(pax6))

        # Compute feature scores using SINGLE representative genes
        # Simplified from gene products to allow morphogen gradients to create features

        # Eye: pax6 only (most representative eye marker)
        eye_score = pax6
        eye_score = torch.where(eye_score > self.min_eye_expr, eye_score, torch.zeros_like(eye_score))

        # Nose: alx only (nose-specific marker)
        nose_score = alx
        nose_score = torch.where(nose_score > self.min_nose_expr, nose_score, torch.zeros_like(nose_score))

        # Mouth: hand2 only (jaw/mouth marker)
        mouth_score = hand2
        mouth_score = torch.where(mouth_score > self.min_mouth_expr, mouth_score, torch.zeros_like(mouth_score))

        # Bone: default state when ALL other features are below threshold (i.e., all zero)
        # Do NOT use runx2 explicitly - it's ubiquitous and would dominate
        # Bone gets score of 1.0 only if all other features are zero (below threshold)
        all_features_zero = (eye_score == 0.0) & (nose_score == 0.0) & (mouth_score == 0.0)
        bone_score = torch.where(all_features_zero, torch.ones_like(eye_score), torch.zeros_like(eye_score))

        return {
            'eye': eye_score,
            'nose': nose_score,
            'mouth': mouth_score,
            'bone': bone_score
        }

    def classify_features_hard(self, feature_scores):
        """
        Hard classification: assign each cell to the feature with highest score.

        Args:
            feature_scores: dict from compute_feature_scores

        Returns:
            feature_grid: (grid_size, grid_size) with integer labels
                0 = bone, 1 = eye, 2 = nose, 3 = mouth
        """
        # Stack scores: (grid_size, grid_size, num_features)
        scores = torch.stack([
            feature_scores['bone'],
            feature_scores['eye'],
            feature_scores['nose'],
            feature_scores['mouth']
        ], dim=-1)

        # Argmax along feature dimension
        feature_grid = torch.argmax(scores, dim=-1)

        return feature_grid

    def classify_features_soft(self, feature_scores):
        """
        Soft classification: probabilistic assignment using softmax.

        Args:
            feature_scores: dict from compute_feature_scores

        Returns:
            feature_probs: (grid_size, grid_size, num_features) - probabilities
        """
        # Stack scores
        scores = torch.stack([
            feature_scores['bone'],
            feature_scores['eye'],
            feature_scores['nose'],
            feature_scores['mouth']
        ], dim=-1)

        # Softmax with temperature
        feature_probs = F.softmax(scores / self.temperature, dim=-1)

        return feature_probs

    def classify(self, gene_grids, mode='hard'):
        """
        Classify features from gene expression.

        Args:
            gene_grids: dict of gene expression grids
            mode: 'hard' (argmax) or 'soft' (probabilistic)

        Returns:
            dict with keys:
                'features': (grid_size, grid_size) - hard labels (0-3)
                'scores': dict - feature scores
                'probabilities': (grid_size, grid_size, 4) - soft probabilities (if mode='soft')
        """
        # Compute scores
        scores = self.compute_feature_scores(gene_grids)

        # Hard classification
        features = self.classify_features_hard(scores)

        result = {
            'features': features,
            'scores': scores,
            'feature_names': self.feature_names
        }

        # Soft classification (optional)
        if mode == 'soft' or mode == 'both':
            probs = self.classify_features_soft(scores)
            result['probabilities'] = probs

        return result

    def get_feature_masks(self, feature_grid):
        """
        Get binary masks for each feature type.

        Args:
            feature_grid: (grid_size, grid_size) with integer labels

        Returns:
            dict with keys 'bone', 'eye', 'nose', 'mouth' → binary masks
        """
        masks = {}
        for idx, name in enumerate(self.feature_names):
            masks[name] = (feature_grid == idx).to(dtype=self.dtype)

        return masks

    def summarize_features(self, feature_grid):
        """
        Count cells assigned to each feature.

        Args:
            feature_grid: (grid_size, grid_size) with integer labels

        Returns:
            dict with feature counts
        """
        unique, counts = torch.unique(feature_grid, return_counts=True)
        count_dict = {}

        for idx, name in enumerate(self.feature_names):
            # Find count for this feature
            mask = (unique == idx)
            if mask.any():
                count_dict[name] = int(counts[mask].item())
            else:
                count_dict[name] = 0

        return count_dict
