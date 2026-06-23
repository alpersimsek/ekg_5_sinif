from pathlib import Path

import pytest

from ekg_stage2.constants import LEADS
from ekg_stage2.data.wfdb_io import assess_signal_quality, load_record


@pytest.mark.integration
def test_known_real_record_loads_in_canonical_shape() -> None:
    record = Path(
        "/mnt/d/EKG_WORK/5_Class_Data/files/p1000/p10000032/s40689238/40689238"
    )
    if not record.with_suffix(".hea").exists():
        pytest.skip("Real dataset is unavailable")
    signal = load_record(record, expected_leads=LEADS)
    assert signal.shape == (12, 5000)
    assert assess_signal_quality(signal, 500).valid

