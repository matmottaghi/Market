import argparse
import json
import os
from pathlib import Path

from .config import Settings
from .service import MarketService, schedule_updates
from .storage import atomic_bytes, excel_bytes


def main():
    parser = argparse.ArgumentParser(description="Iran market DSS: Excel, LSTM and options analytics")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    pipeline = commands.add_parser("pipeline")
    pipeline.add_argument("--symbols", nargs="+")
    pipeline.add_argument("--no-live", action="store_true")
    pipeline.add_argument("--data-only", action="store_true")
    pipeline.add_argument("--retrain", action="store_true")
    importer = commands.add_parser("import")
    importer.add_argument("workbook", type=Path)
    importer.add_argument("--mapping", type=Path, help="Optional per-sheet JSON column mapping")
    inspector = commands.add_parser("inspect")
    inspector.add_argument("workbook", type=Path)
    updater = commands.add_parser("update")
    updater.add_argument("--history", action="store_true")
    updater.add_argument(
        "--symbols", nargs="+", help="Any stock/ETF symbols; omit for leveraged ETF discovery"
    )
    trainer = commands.add_parser("train")
    trainer.add_argument("symbol", help="Symbol, stable identifier, or 'all'")
    trainer.add_argument("--horizon", type=int, default=21)
    trainer.add_argument("--epochs", type=int, default=60)
    trainer.add_argument("--window", type=int, default=30)
    trainer.add_argument("--device", default="cpu")
    trainer.add_argument("--attention", action="store_true")
    predictor = commands.add_parser("predict")
    predictor.add_argument("symbol")
    predictor.add_argument("--model", type=Path)
    predictor.add_argument("--horizon", type=int, default=21)
    predictor.add_argument("--paths", type=int, default=2000)
    walk = commands.add_parser("walk-forward")
    walk.add_argument("symbol")
    walk.add_argument("--horizon", type=int, default=5)
    walk.add_argument("--initial", type=int, default=300)
    walk.add_argument("--step", type=int, default=21)
    walk.add_argument("--epochs", type=int, default=30)
    commands.add_parser("validate")
    exporter = commands.add_parser("export")
    exporter.add_argument("destination", type=Path)
    scheduled = commands.add_parser("schedule")
    scheduled.add_argument("--history", action="store_true")
    scheduled.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    service = MarketService(Settings.from_env())
    if args.command == "init":
        print("Initialized empty DSS storage; no market observations generated.")
    elif args.command == "pipeline":
        from .pipeline import MarketPipelineService
        report = MarketPipelineService(service).run_full_pipeline(
            symbols=args.symbols, update_live=not args.no_live,
            update_forecasts=not args.data_only, retrain_policy="always" if args.retrain else "auto")
        print(json.dumps({"run_id": report["run_id"], "status": report["status"], "counts": report["counts"],
                          "stages": report["stages"]}, ensure_ascii=True, indent=2))
    elif args.command == "inspect":
        from .ingestion import CollectorAdapter

        print(json.dumps(CollectorAdapter().inspect(args.workbook), ensure_ascii=False, indent=2))
    elif args.command == "import":
        mapping = json.loads(args.mapping.read_text(encoding="utf-8")) if args.mapping else None
        print(json.dumps(service.import_file(args.workbook, mapping), ensure_ascii=False, indent=2))
    elif args.command == "update":
        print(json.dumps(service.update(args.history, tickers=args.symbols), ensure_ascii=False, indent=2))
    elif args.command in ("train", "predict", "walk-forward"):
        from .forecasting import ForecastEngine, ModelConfig, walk_forward_forecasts

        if args.command == "train":
            symbols = service.instruments().symbol.tolist() if args.symbol == "all" else [args.symbol]
            if not symbols:
                parser.error("No instruments; import a workbook or update first")
            for symbol in symbols:
                try:
                    model = ForecastEngine(
                        ModelConfig(
                            horizon=args.horizon,
                            window=args.window,
                            epochs=args.epochs,
                            device=args.device,
                            attention=args.attention,
                            annual_sessions=service.settings.annual_sessions,
                        )
                    ).fit(service.history(symbol))
                    print(model.save(service.settings.model_dir))
                    print(model.metrics.to_string(index=False))
                except ValueError as exc:
                    if args.symbol != "all":
                        raise
                    print(f"Skipped {symbol}: {exc}")
        elif args.command == "predict":
            history = service.history(args.symbol)
            if history.empty:
                parser.error("No historical observations for this instrument")
            versions = service.models(history.instrument_id.iloc[-1])
            if not args.model and not versions:
                parser.error("No trained model; run train first")
            model = ForecastEngine.load(args.model or versions[0].parent)
            forecast = model.predict(history, args.horizon, args.paths, calendar=service.calendar)
            destination = (
                service.settings.data_dir
                / "forecasts"
                / f"{history.instrument_id.iloc[-1]}_{model.metadata['version']}.xlsx"
            )
            forecast.export(destination)
            print(destination)
        else:
            result = walk_forward_forecasts(
                service.history(args.symbol),
                ModelConfig(horizon=args.horizon, epochs=args.epochs),
                args.initial,
                args.step,
            )
            destination = service.settings.data_dir / "backtests" / "forecast_walk_forward.xlsx"
            atomic_bytes(destination, excel_bytes({"predictions": result}))
            print(destination)
    elif args.command == "validate":
        print(service.quality().to_string(index=False))
    elif args.command == "export":
        service.repository.export(args.destination)
    elif args.command == "schedule":
        schedule_updates(
            service,
            int(os.getenv("DSS_UPDATE_HOUR", "16")),
            int(os.getenv("DSS_UPDATE_MINUTE", "30")),
            args.history,
            args.retrain,
        )


if __name__ == "__main__":
    main()
