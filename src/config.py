from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "flipkart_customer_service.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

# Set this explicitly if automatic target detection does not find your label.
TARGET_COLUMN = None

RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_TEXT_FEATURES = 5000
MIN_TEXT_LENGTH = 3

TARGET_CANDIDATES = [
    "CSAT Score", "CSAT_Score", "CSAT", "csat_score",
    "satisfaction", "satisfaction_score", "customer_satisfaction"
]

TEXT_CANDIDATES = [
    "Customer Remarks", "Customer_Remarks", "remarks",
    "feedback", "comment", "comments", "review",
    "chat", "chat_transcript", "transcript"
]
