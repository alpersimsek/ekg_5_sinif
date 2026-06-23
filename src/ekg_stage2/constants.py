from typing import Final

LABELS: Final[tuple[str, ...]] = ("NORMAL", "AFIB", "AFL", "LBBB", "RBBB")
LEADS: Final[tuple[str, ...]] = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)
EXPECTED_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "subject_id",
    "study_id",
    "file_name",
    "path",
    *LABELS,
)

