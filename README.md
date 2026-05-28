The main files implementing the bioelectric field model are:
`cellularFieldNetwork.py`, `embryo.py`, `simulateTrainedModel.py`, and `simulateCellularFieldNetwork.py`.

To simulate the Mosaic and Stigmergic models, set the `Model` variable in
`simulateTrainedModel.py` to either `"Stigmergic"` or `"Mosaic"` and run the file.

To simulate arbitrary models, set the appropriate variables in
`simulateCellularFieldNetwork.py` and run the file.

To learn model parameters, run `bash runLearnCellularFieldNetwork.sh`.

To analyze parameter sweeps and avalanche/criticality properties, use the
`analyze*.py` and `compute*.py` scripts, or run `bash runPlotAnalysisData.sh`.