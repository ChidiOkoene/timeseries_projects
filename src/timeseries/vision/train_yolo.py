"""Train YOLOv8 on annotated candlestick chart patterns."""

from __future__ import annotations

from pathlib import Path

from ultralytics.models import YOLO

from timeseries.paths import ARTIFACTS_DIR, CODES_DIR, ensure_runtime_dirs


def train_chart_yolo(
    data_yaml: str | Path | None = None,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    weights: str = "yolov8n.pt",
    project: str | Path | None = None,
    name: str = "xrp_patterns",
    run_inference: bool = False,
    inference_source: str | Path | None = None,
):
    """
    Train YOLOv8n on the local chart dataset.

    Defaults point at the legacy ``codes/chart_dataset/`` tree on disk
    (gitignored). Override paths as needed.
    """
    ensure_runtime_dirs()
    if data_yaml is None:
        data_yaml = CODES_DIR / "chart_dataset" / "data.yaml"
    data_yaml = Path(data_yaml)
    if project is None:
        project = ARTIFACTS_DIR / "chart_yolo"
    project = Path(project)
    project.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project),
        name=name,
        exist_ok=True,
    )

    if run_inference:
        if inference_source is None:
            inference_source = CODES_DIR / "chart_dataset" / "images_with_boxes"
        model.predict(
            source=str(inference_source),
            conf=0.25,
            save=True,
            project=str(project),
            name="inference_results",
        )

    best_path = project / "best.pt"
    model.save(str(best_path))
    return results


if __name__ == "__main__":
    train_chart_yolo(run_inference=True)
