"""ユーティリティ関数"""


def kmh_to_mps(v_kmh: float) -> float:
    """km/h を m/s に変換"""
    return float(v_kmh) / 3.6


def mps_to_kmh(v_mps: float) -> float:
    """m/s を km/h に変換"""
    return float(v_mps) * 3.6
