import logfire
from fastapi import FastAPI


def setup_telemetry(app: FastAPI) -> None:
    """
    Configure Logfire tracing for the FastAPI app.
    Instruments all requests automatically.
    """
    logfire.configure(
        # token read from LOGFIRE_TOKEN env var automatically
        # if not set, logs to console in dev mode
        send_to_logfire="if-token-present",
        service_name="raglab",
    )

    # auto-instruments all FastAPI routes — adds trace per request
    logfire.instrument_fastapi(app)
