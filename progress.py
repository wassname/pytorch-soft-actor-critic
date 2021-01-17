from rich.progress import (
    ProgressColumn,
    BarColumn,
    DownloadColumn,
    TextColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
    Progress,
    TaskID,
    TimeElapsedColumn,
    SpinnerColumn,
    Text
)

class SpeedColumn(ProgressColumn):
    """Renders human readable transfer speed."""

    def render(self, task: "Task") -> Text:
        """Show data transfer speed."""
        speed = task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        return Text(f"{speed:2.2f} it/s", style="progress.data.speed")

def RichTQDM():
    return Progress(
    SpinnerColumn(),
    "[progress.description]{task.description}",
    BarColumn(),
    TextColumn("{task.completed}/{task.total}"),
    "[",
    TimeElapsedColumn(),
    "<",
    TimeRemainingColumn(),
    ',',
    SpeedColumn(),
    ']',
    refresh_per_second=.1, speed_estimate_period=30
    )
