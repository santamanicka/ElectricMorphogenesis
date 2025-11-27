#!/bin/bash
# Run script for learnRefinedFacialIntegration.py
# Learns bioelectric-morphogen-gene parameters to match target facial features

# =============================================================================
# Configuration
# =============================================================================

# Grid and simulation parameters
GRID_SIZE=11
NUM_SIM_ITERS=1000        # Stigmergic bioelectric simulation iterations
NUM_GRN_ITERS=2000        # GRN iterations per learning step
NUM_LEARN_ITERS=10000       # Number of learning iterations

# Learning parameters
LEARNING_RATE=0.02
LOSS_METHOD="featureMap"  # Options: featureMap, featureMapMSE, accuracy

# File paths
STIGMERGIC_PARAMS="data/StigmergicModelParameters.dat"
IDEAL_FACE="IdealFace.png"

# Output control
VERBOSE="True"

# =============================================================================
# Learning Configurations
# =============================================================================

# Configuration 1: Bioelectric gating only (most impactful)
config_bioelectric() {
    FILE_NUMBER=$1
    LEARNED_PARAMS="['ca_threshold_percentile','ca_sensitivity','and_threshold']"

    echo "=========================================="
    echo "Configuration 1: Bioelectric Gating"
    echo "Learning: Ca threshold, sensitivity, AND threshold"
    echo "=========================================="

    python learnRefinedFacialIntegration.py \
        --gridSize $GRID_SIZE \
        --numSimIters $NUM_SIM_ITERS \
        --numGRNIters $NUM_GRN_ITERS \
        --numLearnIters $NUM_LEARN_ITERS \
        --lr $LEARNING_RATE \
        --lossMethod $LOSS_METHOD \
        --learnedParameters "$LEARNED_PARAMS" \
        --idealFacePath $IDEAL_FACE \
        --stigmergicParamsPath $STIGMERGIC_PARAMS \
        --fileNumber $FILE_NUMBER \
        --verbose $VERBOSE
}

# Configuration 2: Bioelectric + AND gate sharpness
config_bioelectric_plus_sharpness() {
    FILE_NUMBER=$1
    LEARNED_PARAMS="['ca_threshold_percentile','ca_sensitivity','and_threshold','and_sharpness']"

    echo "=========================================="
    echo "Configuration 2: Bioelectric + Sharpness"
    echo "Learning: Ca threshold, sensitivity, AND threshold + sharpness"
    echo "=========================================="

    python learnRefinedFacialIntegration.py \
        --gridSize $GRID_SIZE \
        --numSimIters $NUM_SIM_ITERS \
        --numGRNIters $NUM_GRN_ITERS \
        --numLearnIters $NUM_LEARN_ITERS \
        --lr $LEARNING_RATE \
        --lossMethod $LOSS_METHOD \
        --learnedParameters "$LEARNED_PARAMS" \
        --idealFacePath $IDEAL_FACE \
        --stigmergicParamsPath $STIGMERGIC_PARAMS \
        --fileNumber $FILE_NUMBER \
        --verbose $VERBOSE
}

# Configuration 3: Bioelectric + FGF8 morphogen
config_bioelectric_plus_fgf8() {
    FILE_NUMBER=$1
    LEARNED_PARAMS="['ca_threshold_percentile','ca_sensitivity','and_threshold','fgf8_strength','fgf8_degradation_factor']"

    echo "=========================================="
    echo "Configuration 3: Bioelectric + FGF8"
    echo "Learning: Bioelectric gating + FGF8 dynamics"
    echo "=========================================="

    python learnRefinedFacialIntegration.py \
        --gridSize $GRID_SIZE \
        --numSimIters $NUM_SIM_ITERS \
        --numGRNIters $NUM_GRN_ITERS \
        --numLearnIters $NUM_LEARN_ITERS \
        --lr $LEARNING_RATE \
        --lossMethod $LOSS_METHOD \
        --learnedParameters "$LEARNED_PARAMS" \
        --idealFacePath $IDEAL_FACE \
        --stigmergicParamsPath $STIGMERGIC_PARAMS \
        --fileNumber $FILE_NUMBER \
        --verbose $VERBOSE
}

# Configuration 4: Full model (bioelectric + morphogen + gene dynamics)
config_full() {
    FILE_NUMBER=$1
    LEARNED_PARAMS="['ca_threshold_percentile','ca_sensitivity','and_threshold','and_sharpness','fgf8_strength','fgf8_degradation_factor','k_activation','k_degradation']"

    echo "=========================================="
    echo "Configuration 4: Full Model"
    echo "Learning: Bioelectric + morphogen + gene dynamics"
    echo "=========================================="

    python learnRefinedFacialIntegration.py \
        --gridSize $GRID_SIZE \
        --numSimIters $NUM_SIM_ITERS \
        --numGRNIters $NUM_GRN_ITERS \
        --numLearnIters $NUM_LEARN_ITERS \
        --lr $LEARNING_RATE \
        --lossMethod $LOSS_METHOD \
        --learnedParameters "$LEARNED_PARAMS" \
        --idealFacePath $IDEAL_FACE \
        --stigmergicParamsPath $STIGMERGIC_PARAMS \
        --fileNumber $FILE_NUMBER \
        --verbose $VERBOSE
}

# Configuration 5: Feature classification threshold
config_feature_threshold() {
    FILE_NUMBER=$1
    LEARNED_PARAMS="['ca_threshold_percentile','ca_sensitivity','and_threshold','min_mouth_expr']"

    echo "=========================================="
    echo "Configuration 5: Bioelectric + Feature Threshold"
    echo "Learning: Bioelectric gating + mouth classification threshold"
    echo "=========================================="

    python learnRefinedFacialIntegration.py \
        --gridSize $GRID_SIZE \
        --numSimIters $NUM_SIM_ITERS \
        --numGRNIters $NUM_GRN_ITERS \
        --numLearnIters $NUM_LEARN_ITERS \
        --lr $LEARNING_RATE \
        --lossMethod $LOSS_METHOD \
        --learnedParameters "$LEARNED_PARAMS" \
        --idealFacePath $IDEAL_FACE \
        --stigmergicParamsPath $STIGMERGIC_PARAMS \
        --fileNumber $FILE_NUMBER \
        --verbose $VERBOSE
}

# Configuration 6: Quick test (fewer iterations)
config_quick_test() {
    FILE_NUMBER=$1
    LEARNED_PARAMS="['ca_threshold_percentile','ca_sensitivity','and_threshold']"

    echo "=========================================="
    echo "Configuration 6: Quick Test"
    echo "Learning: Bioelectric gating (fast)"
    echo "=========================================="

    python learnRefinedFacialIntegration.py \
        --gridSize $GRID_SIZE \
        --numSimIters 10 \
        --numGRNIters 10 \
        --numLearnIters 10 \
        --lr 0.05 \
        --lossMethod $LOSS_METHOD \
        --learnedParameters "$LEARNED_PARAMS" \
        --idealFacePath $IDEAL_FACE \
        --stigmergicParamsPath $STIGMERGIC_PARAMS \
        --fileNumber $FILE_NUMBER \
        --verbose $VERBOSE
}

# Configuration 7: Long training (more iterations for convergence)
config_long_training() {
    FILE_NUMBER=$1
    LEARNED_PARAMS="['ca_threshold_percentile','ca_sensitivity','and_threshold','and_sharpness','fgf8_strength']"

    echo "=========================================="
    echo "Configuration 7: Long Training"
    echo "Learning: Bioelectric + FGF8 (extended)"
    echo "=========================================="

    python learnRefinedFacialIntegration.py \
        --gridSize $GRID_SIZE \
        --numSimIters $NUM_SIM_ITERS \
        --numGRNIters 8000 \
        --numLearnIters 200 \
        --lr 0.01 \
        --lossMethod $LOSS_METHOD \
        --learnedParameters "$LEARNED_PARAMS" \
        --idealFacePath $IDEAL_FACE \
        --stigmergicParamsPath $STIGMERGIC_PARAMS \
        --fileNumber $FILE_NUMBER \
        --verbose $VERBOSE
}

# Configuration 8: GRN-only with morphogen shapes + gene dynamics (no bioelectric gating)
config_grn_only() {
    FILE_NUMBER=$1
    LEARNED_PARAMS="['shh_decay_length','fgf8_decay_length','edn1_decay_length',
                     'fgf8_strength','fgf8_degradation_factor','edn1_strength','edn1_degradation_factor',
                     'diffusion_rate','k_activation','k_degradation','K_self','n_self']"

    echo "=========================================="
    echo "Configuration 8: GRN-Only (No Bioelectric)"
    echo "Learning: Morphogen shapes + dynamics + self-maintenance"
    echo "=========================================="

    python learnRefinedFacialIntegration.py \
        --gridSize $GRID_SIZE \
        --numSimIters $NUM_SIM_ITERS \
        --numGRNIters $NUM_GRN_ITERS \
        --numLearnIters $NUM_LEARN_ITERS \
        --lr $LEARNING_RATE \
        --lossMethod $LOSS_METHOD \
        --learnedParameters "$LEARNED_PARAMS" \
        --idealFacePath $IDEAL_FACE \
        --stigmergicParamsPath $STIGMERGIC_PARAMS \
        --fileNumber $FILE_NUMBER \
        --verbose $VERBOSE \
        --grnOnly True
}

# =============================================================================
# Main Execution
# =============================================================================

# Parse command line arguments
if [ $# -eq 0 ]; then
    echo "Usage: bash runLearnRefinedFacialIntegration.sh <config> [file_number]"
    echo ""
    echo "Available configurations:"
    echo "  1 | bioelectric         - Learn Ca gating parameters only (fastest)"
    echo "  2 | bioelectric_sharp   - Learn Ca gating + AND sharpness"
    echo "  3 | bioelectric_fgf8    - Learn Ca gating + FGF8 morphogen"
    echo "  4 | full                - Learn bioelectric + morphogen + gene dynamics"
    echo "  5 | feature_threshold   - Learn Ca gating + mouth threshold"
    echo "  6 | test                - Quick test run (20 iterations)"
    echo "  7 | long                - Long training run (200 iterations)"
    echo "  8 | grn_only            - GRN-only mode with morphogen shape learning"
    echo "  all                     - Run all configurations sequentially"
    echo ""
    echo "Optional: Specify file_number (default: 0)"
    echo ""
    echo "Examples:"
    echo "  bash runLearnRefinedFacialIntegration.sh 1           # Config 1, file 0"
    echo "  bash runLearnRefinedFacialIntegration.sh bioelectric 5  # Config 1, file 5"
    echo "  bash runLearnRefinedFacialIntegration.sh test        # Quick test"
    echo "  bash runLearnRefinedFacialIntegration.sh grn_only 8  # GRN-only mode"
    echo "  bash runLearnRefinedFacialIntegration.sh all         # Run all configs"
    exit 1
fi

CONFIG=$1
FILE_NUMBER=${2:-0}  # Default to 0 if not specified

echo "=============================================="
echo "REFINED FACIAL INTEGRATION PARAMETER LEARNING"
echo "=============================================="
echo "Configuration: $CONFIG"
echo "File number: $FILE_NUMBER"
echo "Grid size: ${GRID_SIZE}x${GRID_SIZE}"
echo "Learning rate: $LEARNING_RATE"
echo "Loss method: $LOSS_METHOD"
echo "=============================================="
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Run selected configuration
case $CONFIG in
    1|bioelectric)
        config_bioelectric $FILE_NUMBER
        ;;
    2|bioelectric_sharp)
        config_bioelectric_plus_sharpness $FILE_NUMBER
        ;;
    3|bioelectric_fgf8)
        config_bioelectric_plus_fgf8 $FILE_NUMBER
        ;;
    4|full)
        config_full $FILE_NUMBER
        ;;
    5|feature_threshold)
        config_feature_threshold $FILE_NUMBER
        ;;
    6|test)
        config_quick_test $FILE_NUMBER
        ;;
    7|long)
        config_long_training $FILE_NUMBER
        ;;
    8|grn_only)
        config_grn_only $FILE_NUMBER
        ;;
    all)
        echo "Running all configurations sequentially..."
        echo ""
        config_quick_test 0
        echo ""
        config_bioelectric 1
        echo ""
        config_bioelectric_plus_sharpness 2
        echo ""
        config_bioelectric_plus_fgf8 3
        echo ""
        config_feature_threshold 4
        echo ""
        config_full 5
        echo ""
        echo "=============================================="
        echo "✅ All configurations complete!"
        echo "=============================================="
        echo ""
        echo "Results saved to:"
        echo "  data/bestLearnedFacialParams_0.dat (quick test)"
        echo "  data/bestLearnedFacialParams_1.dat (bioelectric)"
        echo "  data/bestLearnedFacialParams_2.dat (bioelectric + sharpness)"
        echo "  data/bestLearnedFacialParams_3.dat (bioelectric + FGF8)"
        echo "  data/bestLearnedFacialParams_4.dat (feature threshold)"
        echo "  data/bestLearnedFacialParams_5.dat (full model)"
        echo ""
        echo "Visualizations saved to:"
        echo "  learned_facial_comparison_*.png"
        ;;
    *)
        echo "Error: Unknown configuration '$CONFIG'"
        echo "Run without arguments to see usage."
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "✅ Learning complete!"
echo "=============================================="
echo ""
echo "Output files:"
echo "  Parameters: data/bestLearnedFacialParams_${FILE_NUMBER}.dat"
echo "  Visualization: learned_facial_comparison_${FILE_NUMBER}.png"
echo ""

#sbatch --export=ALL,grnOnly=True --time 1-00:00:00 -p batch --array 1-100 -e Error_%A_%a.err --mem 1G runLearnRefinedFacialntegration.sh