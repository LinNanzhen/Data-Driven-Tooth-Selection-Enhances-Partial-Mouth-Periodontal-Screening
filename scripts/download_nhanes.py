import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
URL_TEMPLATE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"
TIMEOUT_SECONDS = 120
XPORT_MAGIC = b"HEADER RECORD*******LIBRARY HEADER RECORD"
MANIFEST_HEADER = "cycle\trole\tcomponent\tfile\turl\tbytes\tsha256\tdownloaded_utc"

FILES = (
    ("2009-2010", "development", "periodontal", "OHXPER_F"),
    ("2009-2010", "development", "dentition", "OHXDEN_F"),
    ("2009-2010", "development", "demographics", "DEMO_F"),
    ("2011-2012", "development", "periodontal", "OHXPER_G"),
    ("2011-2012", "development", "dentition", "OHXDEN_G"),
    ("2011-2012", "development", "demographics", "DEMO_G"),
    ("2013-2014", "external_validation", "periodontal", "OHXPER_H"),
    ("2013-2014", "external_validation", "dentition", "OHXDEN_H"),
    ("2013-2014", "external_validation", "demographics", "DEMO_H"),
)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [MANIFEST_HEADER]

    for cycle, role, component, name in FILES:
        url = URL_TEMPLATE.format(year=cycle[:4], name=name)
        path = OUT_DIR / f"{name}.xpt"
        source = path
        if not path.exists() or path.stat().st_size == 0:
            print(f"Downloading {name}.xpt", flush=True)
            source = path.with_suffix(".xpt.part")
            subprocess.run(
                ["curl", "--fail", "--location", "--silent", "--show-error",
                 "--max-time", str(TIMEOUT_SECONDS), "--output", str(source), url],
                check=True,
            )

        data = source.read_bytes()
        if not data.startswith(XPORT_MAGIC):
            raise ValueError(f"{source} has an invalid SAS XPORT header")

        if source != path:
            source.replace(path)

        digest = hashlib.sha256(data).hexdigest()
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        rows.append("\t".join(
            (cycle, role, component, name, url, str(len(data)), digest,
             modified.strftime("%Y-%m-%dT%H:%M:%SZ"))
        ))

    manifest = OUT_DIR / "manifest.tsv"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"{len(FILES)} XPT files ready. Manifest written to {manifest}")


if __name__ == "__main__":
    main()
