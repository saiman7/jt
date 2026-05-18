"""Run the MT5 FastAPI gateway (uvicorn)."""

import uvicorn


def main() -> None:
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
