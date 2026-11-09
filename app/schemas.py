from pydantic import BaseModel, Field, ConfigDict

class AssetReading(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset_id: str = Field(min_length=2, max_length=40)
    temperature_c: float = Field(ge=-20, le=200)
    vibration_mm_s: float = Field(ge=0, le=50)
    pressure_bar: float = Field(ge=0, le=30)
    energy_kw: float = Field(ge=0, le=500)
    operating_hours: int = Field(ge=0, le=2_000_000)
    work_orders_90d: int = Field(ge=0, le=100)
    days_since_maintenance: int = Field(ge=0, le=5000)
    asset_criticality: int = Field(ge=1, le=5)
