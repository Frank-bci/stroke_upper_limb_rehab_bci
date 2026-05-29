from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stroke_bci_mvp.features import BandpowerTransformer, FBCSPTransformer


def build_model(config: dict, sfreq: float) -> Pipeline:
    model_cfg = config["model"]
    model_type = model_cfg.get("type", "fbcsp_lda")
    bands = [tuple(map(float, band)) for band in model_cfg.get("bands", [(8, 12), (12, 16), (16, 24), (24, 30)])]

    if model_type == "fbcsp_lda":
        return Pipeline(
            steps=[
                (
                    "fbcsp",
                    FBCSPTransformer(
                        sfreq=sfreq,
                        bands=bands,
                        n_components=int(model_cfg.get("csp_components", 4)),
                    ),
                ),
                ("scaler", StandardScaler()),
                ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]
        )

    if model_type == "bandpower_logreg":
        return Pipeline(
            steps=[
                ("bandpower", BandpowerTransformer(sfreq=sfreq, bands=bands)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )

    raise ValueError(f"Unsupported model type: {model_type}")

